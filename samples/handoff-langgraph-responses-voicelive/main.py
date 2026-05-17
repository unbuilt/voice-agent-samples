# Copyright (c) Microsoft. All rights reserved.

"""LangGraph Handoff — Hosted agent with Responses protocol.

Multi-agent customer support handoff demo using LangGraph, served through
the Responses protocol for voice-first scenarios.

This sample demonstrates:
  - Multi-agent orchestration with handoff (triage → refund → order agents)
  - Using the Responses protocol with ``ResponseEventStream``
  - Streaming status updates as agents hand off and tools execute
  - Voice-optimized responses (short, conversational)

Required environment variables:
    FOUNDRY_PROJECT_ENDPOINT: Foundry project endpoint (auto-injected in hosted containers)
    OPENAI_API_KEY: API key for the LLM provider
    OPENAI_BASE_URL: Base URL for the LLM provider (default: OpenAI)
    MODEL: Model name (default: gpt-4o)

Usage::

    # Set environment variables
    export FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
    export OPENAI_API_KEY="sk-..."

    # Start the agent
    python main.py

    # Invoke the agent (streaming)
    curl -sS -N -X POST http://localhost:8088/responses \\
        -H "Content-Type: application/json" \\
        -d '{"input": "I need a refund for order 12345", "stream": true}'
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import uuid
from collections.abc import AsyncIterable
from typing import Any

from azure.ai.agentserver.responses import (
    CreateResponse,
    ResponseContext,
    ResponseEventStream,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
)
import httpx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool
from langchain_openai import ChatOpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03dZ [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

if not os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
    logger.warning(
        "APPLICATIONINSIGHTS_CONNECTION_STRING not set — traces will not be sent to "
        "Application Insights. Set it to enable local telemetry."
    )

_endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
if not _endpoint:
    raise EnvironmentError(
        "FOUNDRY_PROJECT_ENDPOINT environment variable is not set. "
        "Set it to your Foundry project endpoint, or use 'azd ai agent run' "
        "which sets it automatically."
    )

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
_proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
_http_client = httpx.Client(proxy=_proxy_url) if _proxy_url else None
_http_async_client = httpx.AsyncClient(proxy=_proxy_url) if _proxy_url else None

llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    model=os.getenv("MODEL", "gpt-4o"),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    http_client=_http_client,
    http_async_client=_http_async_client,
)

# Voice instruction prepended so the LLM keeps responses short and TTS-friendly.
_VOICE_INSTRUCTION = (
    "[Voice mode] Respond in short, natural sentences as if speaking aloud. "
    "No bullet points, numbered lists, markdown, code blocks, or special formatting. "
    "Keep it conversational and concise — a few sentences at most, no more than one question. "
    "Start with a brief acknowledgment (under 5 words), then continue if needed.\n\n"
)

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@langchain_tool
def lookup_order_details(order_id: str) -> dict[str, str]:
    """Return synthetic order details for a given order ID."""
    normalized = "".join(ch for ch in order_id if ch.isdigit()) or order_id
    rng = random.Random(normalized)
    catalog = [
        "Wireless Headphones",
        "Mechanical Keyboard",
        "Gaming Mouse",
        "27-inch Monitor",
        "USB-C Dock",
        "Bluetooth Speaker",
        "Laptop Stand",
    ]
    return {
        "order_id": normalized,
        "item_name": catalog[rng.randrange(len(catalog))],
        "amount": f"${rng.randint(39, 349)}.{rng.randint(0, 99):02d}",
        "currency": "USD",
        "purchase_date": f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "status": "delivered",
    }


@langchain_tool
def submit_refund(refund_description: str, amount: str, order_id: str) -> str:
    """Process a refund request."""
    return f"Refund recorded for order {order_id} (amount: {amount}): {refund_description}"


@langchain_tool
def submit_replacement(order_id: str, shipping_preference: str, replacement_note: str) -> str:
    """Process a replacement request."""
    return f"Replacement recorded for order {order_id} (shipping: {shipping_preference}): {replacement_note}"


# ---------------------------------------------------------------------------
# Agent definitions (system prompts + tools + handoff targets)
# ---------------------------------------------------------------------------

AGENTS: dict[str, dict[str, Any]] = {
    "triage_agent": {
        "system": (
            "You are the customer support triage agent.\n"
            "Routing policy:\n"
            "1. Route refund-related requests to refund_agent using the handoff_to_refund_agent tool.\n"
            "2. Route replacement/shipping requests to order_agent using the handoff_to_order_agent tool.\n"
            "3. If user wants both refund and replacement, route to refund_agent first.\n"
            "4. Do not force replacement if the user asked for refund only.\n"
            "5. When the issue is fully resolved, send a warm farewell and end with exactly: Case complete.\n"
            "\n"
            "CRITICAL RULES:\n"
            "- You MUST use a handoff tool to transfer. NEVER just describe a transfer in text.\n"
            "- If the user asks for something outside your scope (new orders, general questions),\n"
            "  politely explain you can only help with refunds and replacements, then ask if there's\n"
            "  anything else. If not, say farewell and end with: Case complete."
        ),
        "tools": [],
        "handoffs": ["refund_agent", "order_agent"],
    },
    "refund_agent": {
        "system": (
            "You are the refund specialist.\n"
            "Workflow policy:\n"
            "1. If order_id is missing, ask only for order_id.\n"
            "2. Once order_id is available, call lookup_order_details(order_id).\n"
            "3. Do not ask the customer how much they paid.\n"
            "4. If user intent is ambiguous, ask: refund only, replacement only, or both.\n"
            "5. If the user wants a refund, call submit_refund with order_id, amount, and description.\n"
            "6. After successful refund:\n"
            "   - If user also wants replacement, call handoff_to_order_agent immediately.\n"
            "   - If refund only, call handoff_to_triage_agent immediately for farewell.\n"
            "7. If replacement only, call handoff_to_order_agent directly.\n"
            "8. Never say 'Case complete.' yourself.\n"
            "\n"
            "CRITICAL RULES:\n"
            "- You MUST use a handoff tool to transfer. NEVER just describe a transfer in text.\n"
            "- After submit_refund succeeds, you MUST call a handoff tool in the same turn.\n"
            "  Do NOT ask 'anything else?' — just handoff.\n"
            "- If a user asks for something unrelated to refunds, call handoff_to_triage_agent."
        ),
        "tools": [lookup_order_details, submit_refund],
        "handoffs": ["order_agent", "triage_agent"],
    },
    "order_agent": {
        "system": (
            "You are the order specialist.\n"
            "Only handle replacement/exchange/shipping tasks.\n"
            "1. If shipping preference is missing, ask: standard or expedited.\n"
            "2. If order_id is missing, ask for it.\n"
            "3. Once ready, call submit_replacement(order_id, shipping_preference, replacement_note).\n"
            "4. After success, call handoff_to_triage_agent immediately for farewell.\n"
            "5. Never say 'Case complete.' yourself.\n"
            "If user wants refund only, call handoff_to_refund_agent.\n"
            "\n"
            "CRITICAL RULES:\n"
            "- You MUST use a handoff tool to transfer. NEVER just describe a transfer in text.\n"
            "- After submit_replacement succeeds, you MUST call handoff_to_triage_agent in the same turn.\n"
            "- If a user asks for something unrelated to replacements, call handoff_to_triage_agent."
        ),
        "tools": [lookup_order_details, submit_replacement],
        "handoffs": ["triage_agent", "refund_agent"],
    },
}


def _build_handoff_tool_schemas(agent_name: str) -> list[dict]:
    """Build handoff tool schemas (OpenAI format) for an agent."""
    return [
        {
            "type": "function",
            "function": {
                "name": f"handoff_to_{target}",
                "description": f"Transfer the conversation to {target}.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        for target in AGENTS[agent_name]["handoffs"]
    ]


# ---------------------------------------------------------------------------
# Simple turn-based orchestrator (per-session state)
# ---------------------------------------------------------------------------

_session_state: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    if session_id not in _session_state:
        _session_state[session_id] = {"messages": [], "active_agent": "triage_agent"}
    return _session_state[session_id]


async def run_agent_turn(
    session_id: str, user_text: str
) -> AsyncIterable[dict[str, Any]]:
    """Run one turn, yielding status/content events as they happen.

    Yields dicts with:
      - {"type": "status", "text": "..."} for status updates
      - {"type": "content", "text": "..."} for final response text
      - {"type": "agent", "name": "..."} when agent changes
    """
    session = _get_session(session_id)
    session["messages"].append(HumanMessage(content=user_text))

    active_agent = session["active_agent"]
    working_messages = list(session["messages"])
    max_iterations = 15

    for _ in range(max_iterations):
        agent_def = AGENTS[active_agent]
        system_msg = {"role": "system", "content": _VOICE_INSTRUCTION + agent_def["system"]}

        real_tools = agent_def["tools"]
        handoff_schemas = _build_handoff_tool_schemas(active_agent)
        all_tools = real_tools + [s["function"] for s in handoff_schemas]
        llm_with_tools = llm.bind_tools(all_tools)

        response = await llm_with_tools.ainvoke([system_msg] + working_messages)

        if not response.tool_calls:
            text = response.content or ""
            session["active_agent"] = active_agent
            session["messages"].append(AIMessage(content=text))
            yield {"type": "content", "text": text}
            return

        working_messages.append(response)

        for tc in response.tool_calls:
            tool_name = tc["name"]

            if tool_name.startswith("handoff_to_"):
                target = tool_name.replace("handoff_to_", "")
                if target in AGENTS:
                    working_messages.append(
                        ToolMessage(content=f"Transferred to {target}.", tool_call_id=tc["id"])
                    )
                    active_agent = target
                    session["active_agent"] = active_agent
                    yield {"type": "status", "text": f"Transferring to {target.replace('_', ' ')}..."}
                    yield {"type": "agent", "name": active_agent}
                else:
                    working_messages.append(
                        ToolMessage(content=f"Unknown agent: {target}", tool_call_id=tc["id"])
                    )
            else:
                yield {"type": "status", "text": f"Running {tool_name.replace('_', ' ')}..."}
                tool_fn = next((t for t in real_tools if t.name == tool_name), None)
                if tool_fn:
                    try:
                        result = await tool_fn.ainvoke(tc["args"])
                        result_str = json.dumps(result) if not isinstance(result, str) else result
                    except Exception as e:
                        result_str = f"Error: {e}"
                else:
                    result_str = f"Unknown tool: {tool_name}"
                working_messages.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))

    # Safety fallback
    session["active_agent"] = active_agent
    fallback = "I'm having trouble processing your request. Please try again."
    session["messages"].append(AIMessage(content=fallback))
    yield {"type": "content", "text": fallback}


# ---------------------------------------------------------------------------
# Responses protocol handler
# ---------------------------------------------------------------------------

app = ResponsesAgentServerHost(
    options=ResponsesServerOptions(default_fetch_history_count=20),
)


@app.response_handler
async def handle_response(
    request: CreateResponse,
    context: ResponseContext,
    cancellation_signal: asyncio.Event,
) -> AsyncIterable[dict[str, Any]]:
    """Stream LangGraph handoff output via the Responses protocol."""

    user_input = await context.get_input_text() or "Hello!"

    logger.info(
        "Processing request %s (input length=%d): %.100s",
        context.response_id, len(user_input), user_input,
    )

    stream = ResponseEventStream(
        response_id=context.response_id,
        model=getattr(request, "model", None),
    )

    yield stream.emit_created()
    yield stream.emit_in_progress()

    # Use session ID from context or generate one
    session_id = os.environ.get("FOUNDRY_AGENT_SESSION_ID", str(uuid.uuid4()))

    try:
        async for event in run_agent_turn(session_id, user_input):
            if cancellation_signal.is_set():
                yield stream.emit_incomplete(reason="cancelled")
                return

            if event["type"] == "status":
                # Emit status as a separate output item
                s_item = stream.add_output_item_message()
                yield s_item.emit_added()
                s_tc = s_item.add_text_content()
                yield s_tc.emit_added()
                yield s_tc.emit_delta(event["text"] + "\n")
                yield s_tc.emit_text_done()
                yield s_tc.emit_done()
                yield s_item.emit_done()

            elif event["type"] == "content":
                # Emit the actual response content
                content_item = stream.add_output_item_message()
                yield content_item.emit_added()
                tc = content_item.add_text_content()
                yield tc.emit_added()

                # Stream in chunks for natural feel
                text = event["text"]
                chunk_size = 20
                for i in range(0, len(text), chunk_size):
                    yield tc.emit_delta(text[i:i + chunk_size])

                yield tc.emit_text_done()
                yield tc.emit_done()
                yield content_item.emit_done()

            # "agent" type events are just for internal tracking

    except Exception as exc:
        logger.exception("Agent turn failed")
        err_item = stream.add_output_item_message()
        yield err_item.emit_added()
        err_tc = err_item.add_text_content()
        yield err_tc.emit_added()
        yield err_tc.emit_delta(f"Sorry, something went wrong: {exc}")
        yield err_tc.emit_text_done()
        yield err_tc.emit_done()
        yield err_item.emit_done()

    yield stream.emit_completed()

    logger.info("Request %s completed", context.response_id)


if __name__ == "__main__":
    logger.info("Starting LangGraph Handoff Responses agent")
    app.run()
