# Copyright (c) Microsoft. All rights reserved.

"""Microsoft Agent Framework Handoff — Hosted agent with Responses protocol.

Multi-agent customer support handoff demo using Microsoft Agent Framework's
HandoffBuilder, served through the Responses protocol for voice-first scenarios.

This sample demonstrates:
  - Multi-agent orchestration with HandoffBuilder (triage → refund → order)
  - Using the Responses protocol with ``ResponseEventStream``
  - Streaming status updates as agents hand off and tools execute
  - Voice-optimized responses (short, conversational)
  - No tool approval (auto-approved for voice-first scenarios)

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
import logging
import os
import random
import sys
from collections.abc import AsyncIterable
from typing import Any

import httpx
from agent_framework import (
    Agent,
    Message,
    Workflow,
    WorkflowBuilder,
    WorkflowContext,
    executor,
    tool,
)
from agent_framework.openai import OpenAIChatCompletionClient as OpenAIChatClient
from agent_framework.orchestrations import HandoffBuilder
from azure.ai.agentserver.responses import (
    CreateResponse,
    ResponseContext,
    ResponseEventStream,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
)
from dotenv import load_dotenv

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
# LLM Client
# ---------------------------------------------------------------------------

_proxy_url = (
    os.environ.get("HTTPS_PROXY")
    or os.environ.get("https_proxy")
    or os.environ.get("ALL_PROXY")
    or os.environ.get("all_proxy")
)
_http_client = httpx.Client(proxy=_proxy_url) if _proxy_url else None
_http_async_client = httpx.AsyncClient(proxy=_proxy_url) if _proxy_url else None

_client_kwargs: dict[str, Any] = {
    "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "model": os.getenv("MODEL", "gpt-4o"),
    "api_key": os.getenv("OPENAI_API_KEY", ""),
}

# Voice instruction prepended so the LLM keeps responses short and TTS-friendly.
_VOICE_INSTRUCTION = (
    "[Voice mode] Respond in short, natural sentences as if speaking aloud. "
    "No bullet points, numbered lists, markdown, code blocks, or special formatting. "
    "Keep it conversational and concise — a few sentences at most, no more than one question. "
    "Start with a brief acknowledgment (under 5 words), then continue if needed.\n\n"
)

# ---------------------------------------------------------------------------
# Tools (no approval required for voice-first)
# ---------------------------------------------------------------------------


@tool(approval_mode="never_require")
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


@tool(approval_mode="never_require")
def submit_refund(refund_description: str, amount: str, order_id: str) -> str:
    """Process a refund request."""
    return f"Refund recorded for order {order_id} (amount: {amount}): {refund_description}"


@tool(approval_mode="never_require")
def submit_replacement(order_id: str, shipping_preference: str, replacement_note: str) -> str:
    """Process a replacement request."""
    return f"Replacement recorded for order {order_id} (shipping: {shipping_preference}): {replacement_note}"


# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------


def _create_client() -> OpenAIChatClient:
    return OpenAIChatClient(**_client_kwargs)


def create_agents() -> tuple[Agent, Agent, Agent]:
    """Create triage, refund, and order agents for the handoff workflow."""

    client = _create_client()

    triage = Agent(
        id="triage_agent",
        name="triage_agent",
        require_per_service_call_history_persistence=True,
        instructions=(
            _VOICE_INSTRUCTION
            + "You are the customer support triage agent.\n"
            "Routing policy:\n"
            "1. Route refund-related requests to refund_agent.\n"
            "2. Route replacement/shipping requests to order_agent.\n"
            "3. If user wants both refund and replacement, route to refund_agent first.\n"
            "4. Do not force replacement if the user asked for refund only.\n"
            "5. When the issue is fully resolved, send a warm farewell and end with exactly: Case complete.\n"
            "\n"
            "CRITICAL RULES:\n"
            "- If the user asks for something outside your scope (new orders, general questions),\n"
            "  politely explain you can only help with refunds and replacements, then ask if there's\n"
            "  anything else. If not, say farewell and end with: Case complete."
        ),
        client=client,
    )

    refund = Agent(
        id="refund_agent",
        name="refund_agent",
        require_per_service_call_history_persistence=True,
        instructions=(
            _VOICE_INSTRUCTION
            + "You are the refund specialist.\n"
            "Workflow policy:\n"
            "1. If order_id is missing, ask only for order_id.\n"
            "2. Once order_id is available, call lookup_order_details(order_id).\n"
            "3. Do not ask the customer how much they paid.\n"
            "4. If user intent is ambiguous, ask: refund only, replacement only, or both.\n"
            "5. If the user wants a refund, call submit_refund with order_id, amount, and description.\n"
            "6. After successful refund:\n"
            "   - If user also wants replacement, handoff to order_agent.\n"
            "   - If refund only, handoff to triage_agent for farewell.\n"
            "7. If replacement only, handoff to order_agent directly.\n"
            "8. Never say 'Case complete.' yourself."
        ),
        client=client,
        tools=[lookup_order_details, submit_refund],
    )

    order = Agent(
        id="order_agent",
        name="order_agent",
        require_per_service_call_history_persistence=True,
        instructions=(
            _VOICE_INSTRUCTION
            + "You are the order specialist.\n"
            "Only handle replacement/exchange/shipping tasks.\n"
            "1. If shipping preference is missing, ask: standard or expedited.\n"
            "2. If order_id is missing, ask for it.\n"
            "3. Once ready, call submit_replacement(order_id, shipping_preference, replacement_note).\n"
            "4. After success, handoff to triage_agent for farewell.\n"
            "5. Never say 'Case complete.' yourself.\n"
            "If user wants refund only, handoff to refund_agent."
        ),
        client=client,
        tools=[lookup_order_details, submit_replacement],
    )

    return triage, refund, order


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


def _termination_condition(conversation: list[Message]) -> bool:
    """Stop when any assistant emits an explicit completion marker."""
    for message in reversed(conversation):
        if message.role != "assistant":
            continue
        if (message.text or "").strip().lower().endswith("case complete."):
            return True
    return False


def create_handoff_workflow() -> Workflow:
    """Build the HandoffBuilder workflow."""

    triage, refund, order = create_agents()
    builder = HandoffBuilder(
        name="handoff_maf_voicelive",
        participants=[triage, refund, order],
        termination_condition=_termination_condition,
    )

    (
        builder
        .add_handoff(triage, [refund], description="Route for refunds or damaged-item claims.")
        .add_handoff(triage, [order], description="Route for replacement, exchange, or shipping.")
        .add_handoff(refund, [order], description="Route after refund if replacement is also needed.")
        .add_handoff(refund, [triage], description="Route back for final closure after refund-only.")
        .add_handoff(order, [triage], description="Route back after replacement tasks are complete.")
        .add_handoff(order, [refund], description="Route if user pivots from replacement to refund.")
    )

    return builder.with_start_agent(triage).build()


# ---------------------------------------------------------------------------
# Responses protocol handler
# ---------------------------------------------------------------------------

app = ResponsesAgentServerHost(
    options=ResponsesServerOptions(default_fetch_history_count=20),
)

# Per-session workflow state
_sessions: dict[str, Any] = {}


@app.response_handler
async def handle_response(
    request: CreateResponse,
    context: ResponseContext,
    cancellation_signal: asyncio.Event,
) -> AsyncIterable[dict[str, Any]]:
    """Stream Agent Framework handoff output via the Responses protocol."""

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

    session_id = os.environ.get("FOUNDRY_AGENT_SESSION_ID", "default")

    # Create or reuse workflow for this session
    if session_id not in _sessions:
        workflow = create_handoff_workflow()
        _sessions[session_id] = {"workflow": workflow, "completed": False}

    session = _sessions[session_id]

    if session["completed"]:
        done_item = stream.add_output_item_message()
        yield done_item.emit_added()
        done_tc = done_item.add_text_content()
        yield done_tc.emit_added()
        yield done_tc.emit_delta("This case has been resolved. Please start a new conversation.")
        yield done_tc.emit_text_done()
        yield done_tc.emit_done()
        yield done_item.emit_done()
        yield stream.emit_completed()
        return

    workflow: Workflow = session["workflow"]

    try:
        # Run the workflow with streaming events
        response_stream = workflow.run(message=user_input, stream=True)
        full_text = ""

        async for event in response_stream:
            if cancellation_signal.is_set():
                yield stream.emit_incomplete(reason="cancelled")
                return

            # Handle output events (final outputs from executors)
            if event.type == "output" and event.data is not None:
                text = ""
                if isinstance(event.data, str):
                    text = event.data
                elif hasattr(event.data, "text"):
                    text = event.data.text or ""
                elif hasattr(event.data, "content"):
                    text = str(event.data.content)

                if not text:
                    continue

                full_text += text

                content_item = stream.add_output_item_message()
                yield content_item.emit_added()
                tc = content_item.add_text_content()
                yield tc.emit_added()

                chunk_size = 20
                for i in range(0, len(text), chunk_size):
                    yield tc.emit_delta(text[i:i + chunk_size])

                yield tc.emit_text_done()
                yield tc.emit_done()
                yield content_item.emit_done()

            # Handle handoff events as status updates
            elif event.type == "handoff_sent":
                target = getattr(event.data, "target_agent", None) or "agent"
                s_item = stream.add_output_item_message()
                yield s_item.emit_added()
                s_tc = s_item.add_text_content()
                yield s_tc.emit_added()
                yield s_tc.emit_delta(f"Transferring to {target}...\n")
                yield s_tc.emit_text_done()
                yield s_tc.emit_done()
                yield s_item.emit_done()

        # Check if case is complete
        if full_text.strip().lower().endswith("case complete."):
            session["completed"] = True

    except Exception as exc:
        logger.exception("Workflow execution failed")
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
    logger.info("Starting Agent Framework Handoff Responses agent")
    app.run()
