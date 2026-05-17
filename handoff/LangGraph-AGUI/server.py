"""LangGraph-based handoff workflow demo.

Turn-by-turn execution — no interrupts for user questions.
No tool approval — suitable for voice-first scenarios.
Uses a simple manual orchestrator to avoid langgraph-swarm's
chat history validation issues with handoff tool_calls.

Also exposes an AG-UI compatible endpoint for reuse with the existing frontend.
"""

from __future__ import annotations

import json
import os
import random
import uuid
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    model=os.getenv("MODEL", "qwen-max"),
    api_key=os.getenv("OPENAI_API_KEY", ""),
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
# Simple turn-based orchestrator
# ---------------------------------------------------------------------------

_thread_state: dict[str, dict] = {}


def _get_thread(thread_id: str) -> dict:
    if thread_id not in _thread_state:
        _thread_state[thread_id] = {"messages": [], "active_agent": "triage_agent"}
    return _thread_state[thread_id]


async def run_agent_turn(thread_id: str, user_text: str) -> tuple[str, str]:
    """Run one turn. Returns (response_text, active_agent_after_turn).

    Within a single turn, the agent may:
    - Call real tools (lookup, submit) — results fed back, loop continues
    - Call handoff tools — switch active agent, loop continues with new agent
    - Produce text without tool calls — turn ends, text returned to user
    """
    thread = _get_thread(thread_id)
    thread["messages"].append(HumanMessage(content=user_text))

    active_agent = thread["active_agent"]
    # Working messages for this turn (includes intra-turn tool calls)
    working_messages = list(thread["messages"])
    max_iterations = 15

    for _ in range(max_iterations):
        agent_def = AGENTS[active_agent]
        system_msg = {"role": "system", "content": agent_def["system"]}

        # Build tool list for LLM
        real_tools = agent_def["tools"]
        handoff_schemas = _build_handoff_tool_schemas(active_agent)

        # Bind real tools + handoff schemas
        all_tools = real_tools + [s["function"] for s in handoff_schemas]
        llm_with_tools = llm.bind_tools(all_tools)

        # Invoke LLM
        response = await llm_with_tools.ainvoke([system_msg] + working_messages)

        if not response.tool_calls:
            # Final text response — turn done
            text = response.content or ""
            thread["active_agent"] = active_agent
            thread["messages"].append(AIMessage(content=text))
            return text, active_agent

        # Process tool calls
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
                else:
                    working_messages.append(
                        ToolMessage(content=f"Unknown agent: {target}", tool_call_id=tc["id"])
                    )
            else:
                # Execute real tool
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
    thread["active_agent"] = active_agent
    fallback = "I'm having trouble processing your request. Please try again."
    thread["messages"].append(AIMessage(content=fallback))
    return fallback, active_agent


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    thread_id: str
    message: str


def create_app() -> FastAPI:
    app = FastAPI(title="LangGraph Handoff Demo")

    cors_origins = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://127.0.0.1:5173").split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    completed_threads: set[str] = set()

    # ------------------------------------------------------------------
    # AG-UI compatible endpoint
    # ------------------------------------------------------------------

    def ag_ui_event(event_type: str, **kwargs) -> str:
        payload = {"type": event_type, **kwargs}
        return f"data: {json.dumps(payload)}\n\n"

    @app.post("/handoff_demo")
    async def ag_ui_endpoint(request: Request):
        body = await request.json()
        thread_id = body.get("thread_id", str(uuid.uuid4()))
        run_id = body.get("run_id", str(uuid.uuid4()))

        # Extract user message
        user_text = None
        messages = body.get("messages", [])
        resume = body.get("resume")

        if resume and isinstance(resume, dict):
            interrupts = resume.get("interrupts", [])
            if interrupts:
                interrupt_value = interrupts[0].get("value")
                if isinstance(interrupt_value, list):
                    for item in interrupt_value:
                        if isinstance(item, dict):
                            for c in item.get("contents", []):
                                if isinstance(c, dict) and c.get("type") == "text":
                                    user_text = c.get("text", "")
                                    break
                elif isinstance(interrupt_value, dict):
                    approved = interrupt_value.get("approved", True)
                    user_text = "Yes, approved." if approved else "No, cancel."

        if not user_text and messages:
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_text = msg.get("content", "")
                    break

        if not user_text:
            user_text = "hello"

        if thread_id in completed_threads:
            async def closed_stream():
                yield ag_ui_event("RUN_STARTED", thread_id=thread_id, run_id=run_id)
                msg_id = str(uuid.uuid4())
                yield ag_ui_event("TEXT_MESSAGE_START", message_id=msg_id, role="assistant")
                yield ag_ui_event("TEXT_MESSAGE_CONTENT", message_id=msg_id, delta="This case has been resolved. Please start a new conversation.")
                yield ag_ui_event("TEXT_MESSAGE_END", message_id=msg_id)
                yield ag_ui_event("RUN_FINISHED", thread_id=thread_id, run_id=run_id)
            return StreamingResponse(closed_stream(), media_type="text/event-stream")

        async def ag_ui_stream():
            yield ag_ui_event("RUN_STARTED", thread_id=thread_id, run_id=run_id)

            try:
                response_text, active_agent = await run_agent_turn(thread_id, user_text)
            except Exception as e:
                msg_id = str(uuid.uuid4())
                yield ag_ui_event("TEXT_MESSAGE_START", message_id=msg_id, role="assistant")
                yield ag_ui_event("TEXT_MESSAGE_CONTENT", message_id=msg_id, delta=f"Error: {e}")
                yield ag_ui_event("TEXT_MESSAGE_END", message_id=msg_id)
                yield ag_ui_event("RUN_FINISHED", thread_id=thread_id, run_id=run_id)
                return

            yield ag_ui_event("STEP_STARTED", step_name=active_agent)

            if response_text:
                msg_id = str(uuid.uuid4())
                yield ag_ui_event("TEXT_MESSAGE_START", message_id=msg_id, role="assistant")
                chunk_size = 20
                for i in range(0, len(response_text), chunk_size):
                    yield ag_ui_event("TEXT_MESSAGE_CONTENT", message_id=msg_id, delta=response_text[i:i+chunk_size])
                yield ag_ui_event("TEXT_MESSAGE_END", message_id=msg_id)

            if response_text.strip().lower().endswith("case complete."):
                completed_threads.add(thread_id)

            yield ag_ui_event("RUN_FINISHED", thread_id=thread_id, run_id=run_id)

        return StreamingResponse(ag_ui_stream(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Simple REST endpoints
    # ------------------------------------------------------------------

    @app.post("/chat")
    async def chat(req: ChatRequest) -> dict:
        if req.thread_id in completed_threads:
            return {"thread_id": req.thread_id, "active_agent": "closed", "response": "Case resolved."}

        response_text, active_agent = await run_agent_turn(req.thread_id, req.message)

        if response_text.strip().lower().endswith("case complete."):
            completed_threads.add(req.thread_id)

        return {"thread_id": req.thread_id, "active_agent": active_agent, "response": response_text}

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        if req.thread_id in completed_threads:
            async def closed():
                yield "data: [closed] Case resolved.\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(closed(), media_type="text/event-stream")

        async def gen():
            response_text, active_agent = await run_agent_turn(req.thread_id, req.message)
            yield f"data: [{active_agent}] {response_text}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8892"))
    print(f"Handoff demo running at http://{host}:{port}")
    print("Endpoints:")
    print("  AG-UI: POST /handoff_demo  (compatible with existing frontend)")
    print("  REST:  POST /chat, POST /chat/stream, GET /healthz")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
