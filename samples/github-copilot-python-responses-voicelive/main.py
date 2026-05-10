# Copyright (c) Microsoft. All rights reserved.

"""GitHub Copilot — Bring Your Own Responses agent with status updates.

Hosted agent that forwards user input to GitHub Copilot via the Copilot SDK
and returns the reply through the Responses protocol, emitting status updates
(tool execution, reasoning progress, etc.) as the Copilot model works.

This sample demonstrates:
  - Using the Responses protocol with ``ResponseEventStream`` for full
    control over streaming lifecycle events.
  - Integrating the GitHub Copilot SDK (``CopilotClient``) as the AI backend.
  - Surfacing real-time status updates (tool calls, reasoning, progress) so
    callers can display progress while the model is working.

Required environment variables:
    FOUNDRY_PROJECT_ENDPOINT: Foundry project endpoint (auto-injected in hosted containers)
    GITHUB_TOKEN: GitHub personal access token for the Copilot SDK

Optional environment variables:
    GITHUB_COPILOT_MODEL: Model to use with Copilot (default: SDK default)

Usage::

    # Set environment variables
    export FOUNDRY_PROJECT_ENDPOINT="https://<account>.services.ai.azure.com/api/projects/<project>"
    export GITHUB_TOKEN="ghp_..."

    # Start the agent
    python main.py

    # Invoke the agent (non-streaming)
    curl -sS -X POST http://localhost:8088/responses \\
        -H "Content-Type: application/json" \\
        -d '{"input": "What is Microsoft Foundry?", "stream": false}' | jq .

    # Invoke the agent (streaming with status updates)
    curl -sS -N -X POST http://localhost:8088/responses \\
        -H "Content-Type: application/json" \\
        -d '{"input": "What is Microsoft Foundry?", "stream": true}'
"""

import asyncio
import logging
import os
import pathlib
import random
import sys
import time
import re
import uuid
from collections.abc import AsyncIterable
from types import SimpleNamespace
from typing import Any

from azure.ai.agentserver.responses import (
    CreateResponse,
    ResponseContext,
    ResponseEventStream,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
)

from copilot import CopilotClient, SubprocessConfig
from copilot.generated.session_events import SessionEvent, SessionEventType

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
        "Application Insights. Set it to enable local telemetry. "
        "(This variable is auto-injected in hosted Foundry containers — do not declare it in agent.manifest.yaml.)"
    )

_endpoint = os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
if not _endpoint:
    raise EnvironmentError(
        "FOUNDRY_PROJECT_ENDPOINT environment variable is not set. "
        "Set it to your Foundry project endpoint, or use 'azd ai agent run' "
        "which sets it automatically."
    )

_github_token = os.environ.get("GITHUB_TOKEN")
if not _github_token:
    raise EnvironmentError(
        "GITHUB_TOKEN environment variable is not set. "
        "Set it to a GitHub personal access token with Copilot access."
    )

# ── Copilot SDK helpers ─────────────────────────────────────────────────────

# Voice instruction prepended to each user message so Copilot knows
# to keep responses short and TTS-friendly.
_VOICE_INSTRUCTION = (
    "You are working in a Voice output mode. "
    "Please feedback the user input with a short (under 5 words) message first. then continue messages if needed. "
    "Keep the messages concise and informative, suitable for text-to-speech readout. Don't ask many questions in a message. "
    "\n"
    "User input: "
)


def _approve_all(request, context):
    """Auto-approve all permission requests (no interactive user in container)."""
    from copilot.session import PermissionRequestResult
    return PermissionRequestResult(kind="approve-once")


# Human-friendly labels for well-known Copilot tool names.
_TOOL_LABELS: dict[str, str] = {
    "view": "Reading files",
    "glob": "Searching files",
    "grep": "Searching code",
    "powershell": "Running command",
    "bash": "Running command",
    "python": "Running Python",
    "node": "Running Node.js",
    "create": "Creating file",
    "edit": "Editing file",
    "report_intent": "Planning",
}

# Max characters for readout before truncation.
_MAX_READOUT_CHARS = 300


def _friendly_tool_label(name: str) -> str:
    """Return a short user-facing label for a tool name."""
    return _TOOL_LABELS.get(name, f"Using {name}")


def _make_stream_event_handler(
    queue: "asyncio.Queue[SimpleNamespace | Exception | None]",
    loop: asyncio.AbstractEventLoop,
):
    """Build an event handler that maps Copilot SDK events to queued chunks.

    Each queued item is a ``SimpleNamespace`` with:
      - ``text``: the content to stream
      - ``is_status`` (bool): True for status/progress updates, False for
        actual assistant output

    Status events are filtered to brief, user-friendly summaries:
      - Tool start → short label (e.g. "Searching files...")
      - Tool completion, reasoning deltas, turn starts → suppressed
    """
    seen_tools: set[str] = set()  # de-duplicate repeated tool calls
    last_label: list[str] = [""]  # track last tool label for consecutive de-dup
    last_label_time: list[float] = [0.0]  # monotonic time of last emitted label
    got_deltas: list[bool] = [False]  # track if we received streaming deltas

    # Allow the same consecutive label through after this many seconds.
    _DEDUP_RESET_SECS = 5.0

    def _enqueue(item: SimpleNamespace | Exception | None) -> None:
        """Thread-safe enqueue — the Copilot SDK may fire events from a
        background thread, so we must use call_soon_threadsafe."""
        loop.call_soon_threadsafe(queue.put_nowait, item)

    def _tool_name(event_data) -> str:
        return (
            getattr(event_data, "tool_name", None)
            or getattr(event_data, "mcp_tool_name", None)
            or "tool"
        )

    def handler(event: SessionEvent) -> None:
        etype = event.type

        # Log event type + truncated data for debugging
        try:
            data_str = str({
                k: getattr(event.data, k)
                for k in dir(event.data)
                if not k.startswith("_")
            })
            logger.info("COPILOT_EVENT type=%s data=%.100s", etype, data_str)
        except Exception:
            logger.info("COPILOT_EVENT type=%s", etype)

        # ── Assistant content deltas — stream tokens immediately ────────
        if etype == SessionEventType.ASSISTANT_MESSAGE_DELTA:
            if event.data.delta_content:
                got_deltas[0] = True
                logger.debug(
                    "CONTENT [assistant_delta] (queued): %.200s",
                    event.data.delta_content,
                )
                _enqueue(
                    SimpleNamespace(text=event.data.delta_content, is_status=False)
                )

        # ── Complete assistant message — signal to close the current item
        elif etype == SessionEventType.ASSISTANT_MESSAGE:
            if got_deltas[0]:
                # Deltas were streamed; signal the response loop to close
                # the current output item.
                logger.debug("CONTENT [assistant_message] -> message_done signal")
                _enqueue(SimpleNamespace(text=None, is_status=False, message_done=True))
                got_deltas[0] = False  # reset for next turn
            else:
                # No deltas (non-streaming fallback) — send full content
                content = getattr(event.data, "content", None) or ""
                if content.strip():
                    logger.info(
                        "CONTENT [assistant_message] (queued, is_status=False): %.200s",
                        content,
                    )
                    _enqueue(
                        SimpleNamespace(text=content, is_status=False)
                    )
                    # Also signal close right after
                    _enqueue(SimpleNamespace(text=None, is_status=False, message_done=True))
                else:
                    logger.debug("CONTENT [assistant_message] (skipped, empty)")

        # ── Tool start → announce to user (readable, de-dup consecutive) ──
        elif etype == SessionEventType.TOOL_EXECUTION_START:
            name = _tool_name(event.data)
            label = _friendly_tool_label(name)
            now = time.monotonic()
            logger.info("CONTENT [tool_start] tool=%s label=%s", name, label)
            if label != last_label[0] or (now - last_label_time[0]) >= _DEDUP_RESET_SECS:
                last_label[0] = label
                last_label_time[0] = now
                _enqueue(
                    SimpleNamespace(text=f"{label}...\n", is_status=True)
                )
            else:
                logger.debug("CONTENT [tool_start] (skipped, same as previous)")

        elif etype == SessionEventType.TOOL_EXECUTION_PROGRESS:
            msg = getattr(event.data, "progress_message", None)
            logger.info("CONTENT [tool_progress] (suppressed): %s", msg)
            _enqueue(SimpleNamespace(text=None, is_status=False, activity=True))

        elif etype == SessionEventType.TOOL_EXECUTION_COMPLETE:
            name = _tool_name(event.data)
            call_id = getattr(event.data, "tool_call_id", None)
            logger.info(
                "CONTENT [tool_complete] (suppressed) tool=%s call_id=%s",
                name, call_id,
            )
            _enqueue(SimpleNamespace(text=None, is_status=False, activity=True))

        # ── Reasoning deltas (suppressed from output) ────────────────────
        elif etype == SessionEventType.ASSISTANT_REASONING_DELTA:
            delta = getattr(event.data, "delta_content", None)
            logger.info(
                "CONTENT [reasoning_delta] (suppressed): %.200s",
                delta or "(empty)",
            )

        # ── Turn boundaries (suppressed from output) ─────────────────────
        elif etype == SessionEventType.ASSISTANT_TURN_START:
            logger.info("CONTENT [turn_start] (suppressed)")

        # ── Skill execution → brief status ──────────────────────────────
        elif etype == SessionEventType.SKILL_INVOKED:
            name = getattr(event.data, "tool_name", None) or "skill"
            logger.info(
                "CONTENT [skill_invoked] (queued, is_status=True): %s", name
            )
            _enqueue(
                SimpleNamespace(text=f"Using skill {name}...\n", is_status=True)
            )

        # ── Session lifecycle ────────────────────────────────────────────
        elif etype == SessionEventType.SESSION_IDLE:
            logger.info("CONTENT [session_idle] -> end of stream")
            _enqueue(None)  # signals end of stream
        elif etype == SessionEventType.SESSION_ERROR:
            error_msg = getattr(event.data, "message", None) or "Session error"
            logger.error("CONTENT [session_error]: %s", error_msg)
            _enqueue(RuntimeError(error_msg))

        else:
            logger.info("CONTENT [unhandled] type=%s (ignored)", etype)

    return handler


# ── Copilot SDK session — single shared session per container ────────────────

_copilot_client: CopilotClient | None = None
_session = None
_session_lock = asyncio.Lock()
_skills_dir = str(pathlib.Path(__file__).parent / "skills")


async def _reset_session() -> None:
    """Stop the current session in the background and clear the global.

    The old session's ``stop()`` runs as a fire-and-forget task so the
    caller returns immediately.  The next call to ``_ensure_session()``
    will create a fresh session.
    """
    global _session
    old_session = _session
    _session = None
    if old_session is not None:

        async def _bg_stop():
            try:
                await old_session.stop()
                logger.info("Old session stopped")
            except Exception:
                logger.debug("Old session stop failed (ignored)", exc_info=True)

        asyncio.create_task(_bg_stop())


async def _ensure_session():
    """Resume a persisted session or create a new one (lazy, runs once)."""
    global _copilot_client, _session
    if _session is not None:
        return _session
    async with _session_lock:
        if _session is not None:
            return _session

        config = SubprocessConfig(github_token=_github_token)
        _copilot_client = CopilotClient(config, auto_start=False)
        logger.info("Starting CopilotClient (github_token length=%d)...", len(_github_token))
        await _copilot_client.start()
        logger.info("CopilotClient started")

        session_id = os.environ.get("FOUNDRY_AGENT_SESSION_ID")
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.warning(
                "FOUNDRY_AGENT_SESSION_ID not set, using: %s", session_id)

        home_dir = os.environ.get("HOME", "/home")
        # Windows
        if os.name == "nt":
            home_dir = os.environ.get("USERPROFILE", home_dir)
        working_dir = os.environ.get(
            "HOME", home_dir)  # Copilot needs a working directory, even if we don't use it)
        copilot_model = os.environ.get("GITHUB_COPILOT_MODEL")

        session_kwargs: dict[str, Any] = {
            "session_id": session_id,
            "on_permission_request": _approve_all,
            "streaming": True,
            "skill_directories": [_skills_dir],
            "working_directory": working_dir,
        }
        if copilot_model:
            session_kwargs["model"] = copilot_model

        logger.info(
            "Session config: id=%s, working_dir=%s, model=%s, skills_dir=%s",
            session_id, working_dir, copilot_model or "(default)", _skills_dir,
        )

        try:
            _session = await _copilot_client.resume_session(
                session_id,
                on_permission_request=_approve_all,
                streaming=True,
                skill_directories=[_skills_dir],
                working_directory=working_dir,
            )
            logger.info("Resumed session: %s", session_id)
        except Exception:
            _session = await _copilot_client.create_session(**session_kwargs)
            logger.info("Created session: %s", session_id)

        return _session


# ── Responses protocol handler ───────────────────────────────────────────────

app = ResponsesAgentServerHost(
    options=ResponsesServerOptions(default_fetch_history_count=20),
)


# Filler phrases sent immediately to reduce perceived latency.
# TTS will start speaking one of these while Copilot is still thinking.
_FILLER_PHRASES = [
    "Hmm.",
    "Let me see.",
    "One moment.",
    "Just a sec.",
    "Alright.",
]


@app.response_handler
async def handle_response(
    request: CreateResponse,
    context: ResponseContext,
    cancellation_signal: asyncio.Event,
) -> AsyncIterable[dict[str, Any]]:
    """Stream Copilot output with status updates via the Responses protocol.

    Uses ``ResponseEventStream`` to emit structured lifecycle events:
      1. ``response.created`` — response accepted
      2. ``response.in_progress`` — Copilot is working
      3. Text deltas interleaved with status annotations
      4. ``response.completed`` — all done
    """
    # Kick off session creation early (no-op if already created).
    _schedule_warmup()

    user_input = await context.get_input_text() or "Hello!"

    logger.info(
        "Processing request %s (input length=%d): %.100s",
        context.response_id, len(user_input),
        user_input,
    )

    # Build the Responses protocol event stream
    stream = ResponseEventStream(
        response_id=context.response_id,
        model=getattr(request, "model", None),
    )

    # ── Lifecycle: response created & in progress ────────────────────────
    yield stream.emit_created()
    yield stream.emit_in_progress()

    # ── Filler phrase — speak immediately to reduce perceived latency ────
    filler = random.choice(_FILLER_PHRASES)
    filler_item = stream.add_output_item_message()
    yield filler_item.emit_added()
    fc = filler_item.add_text_content()
    yield fc.emit_added()
    yield fc.emit_delta(filler)
    yield fc.emit_text_done()
    yield fc.emit_done()
    yield filler_item.emit_done()

    # ── Await the session (warmup may have already completed it) ─────────
    session = await _ensure_session()

    # ── Stream from the shared Copilot session ───────────────────────────
    queue: asyncio.Queue[SimpleNamespace | Exception | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    unsubscribe = session.on(_make_stream_event_handler(queue, loop))

    full_text = ""
    has_status = False
    chunk_count = 0

    # Current open output item (created lazily on first delta, closed on
    # message_done signal).  None when no item is open.
    cur_item = None
    cur_tc = None

    async def _open_item():
        """Open a new output item for streaming content."""
        nonlocal cur_item, cur_tc
        cur_item = stream.add_output_item_message()
        yield cur_item.emit_added()
        cur_tc = cur_item.add_text_content()
        yield cur_tc.emit_added()

    async def _close_item():
        """Close the current output item."""
        nonlocal cur_item, cur_tc
        if cur_tc is not None:
            yield cur_tc.emit_text_done()
            yield cur_tc.emit_done()
        if cur_item is not None:
            yield cur_item.emit_done()
        cur_item = None
        cur_tc = None

    # Heartbeat: send periodic updates when nothing has been emitted for a while.
    _HEARTBEAT_SECS = 10.0  # seconds of silence before sending a heartbeat
    _HEARTBEAT_PHRASES = [
        "Still working...",
        "Hang on...",
        "Working on it...",
        "Bear with me...",
        "Just a moment...",
        "Processing...",
    ]
    heartbeat_idx = 0
    last_emit_time = time.monotonic()

    try:
        logger.info("Sending user input to Copilot session")
        send_task = asyncio.create_task(
            session.send(_VOICE_INSTRUCTION + user_input, mode="immediate")
        )

        while True:
            if cancellation_signal.is_set():
                send_task.cancel()
                unsubscribe()
                # Just unsubscribe — the next request will use
                # mode="immediate" which interrupts any in-progress work.
                logger.info("Request %s cancelled by client", context.response_id)
                async for evt in _close_item():
                    yield evt
                yield stream.emit_incomplete(reason="cancelled")
                return

            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if send_task.done() and send_task.exception():
                    raise send_task.exception()
                # ── Heartbeat: emit diverse status after silence ──
                if (time.monotonic() - last_emit_time) >= _HEARTBEAT_SECS:
                    phrase = _HEARTBEAT_PHRASES[heartbeat_idx % len(_HEARTBEAT_PHRASES)]
                    heartbeat_idx += 1
                    hb_item = stream.add_output_item_message()
                    yield hb_item.emit_added()
                    hb_tc = hb_item.add_text_content()
                    yield hb_tc.emit_added()
                    yield hb_tc.emit_delta(f"{phrase}\n")
                    yield hb_tc.emit_text_done()
                    yield hb_tc.emit_done()
                    yield hb_item.emit_done()
                    last_emit_time = time.monotonic()
                    logger.info("Heartbeat sent: %s", phrase)
                continue

            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            # ── message_done signal: close current item ──────────────────
            if getattr(item, "message_done", False):
                async for evt in _close_item():
                    yield evt
                continue

            # ── activity marker: reset heartbeat but emit nothing ────────
            if getattr(item, "activity", False):
                last_emit_time = time.monotonic()
                continue

            chunk = item.text
            if not chunk:
                continue

            last_emit_time = time.monotonic()  # reset heartbeat timer

            if item.is_status:
                has_status = True
                # Status as its own self-contained output item
                s_item = stream.add_output_item_message()
                yield s_item.emit_added()
                s_tc = s_item.add_text_content()
                yield s_tc.emit_added()
                yield s_tc.emit_delta(chunk + "\n")
                yield s_tc.emit_text_done()
                yield s_tc.emit_done()
                yield s_item.emit_done()
            else:
                # Open a new item if none is open
                if cur_item is None:
                    async for evt in _open_item():
                        yield evt

                full_text += chunk
                chunk_count += 1
                yield cur_tc.emit_delta(chunk)

        # Ensure send completed without error
        await send_task

    except Exception as exc:
        logger.exception("Copilot streaming failed")
        error_msg = f"\n\nError: {exc}"
        full_text += error_msg
        if cur_item is None:
            async for evt in _open_item():
                yield evt
        yield cur_tc.emit_delta(error_msg)
    finally:
        unsubscribe()

    # ── Lifecycle: finalize ──────────────────────────────────────────────
    # Close any still-open item
    async for evt in _close_item():
        yield evt
    yield stream.emit_completed()

    logger.info(
        "Request %s completed (%d chars, %d chunks, status=%s)",
        context.response_id, len(full_text), chunk_count, has_status,
    )


_warmup_task: asyncio.Task | None = None


def _schedule_warmup() -> None:
    """Schedule eager session creation on the running event loop.

    Safe to call multiple times — only the first call has an effect.
    Must be called when an asyncio event loop is running.
    """
    global _warmup_task
    if _warmup_task is not None:
        return

    async def _do_warmup():
        try:
            await _ensure_session()
            logger.info("Copilot session warmed up")
        except Exception:
            logger.warning(
                "Session warmup failed; will retry on first request",
                exc_info=True,
            )

    _warmup_task = asyncio.ensure_future(_do_warmup())


# Hook into the ASGI app's lifespan so warmup runs as soon as the server
# starts, NOT on the first request.  The underlying app object is an ASGI
# application served by Hypercorn.
_original_app_call = app.__call__.__func__ if hasattr(app.__call__, '__func__') else None


async def _warmup_asgi_wrapper(self, scope, receive, send):
    """ASGI wrapper that triggers warmup on the first call of any type."""
    _schedule_warmup()
    # Replace ourselves with the original to avoid the check on every call
    if _original_app_call:
        type(self).__call__ = _original_app_call
    return await _original_app_call(self, scope, receive, send) if _original_app_call else None


# Try to monkey-patch; if the app structure doesn't support it, fall back
# to warmup-on-first-request (which is what _schedule_warmup in the handler does).
if _original_app_call is not None:
    type(app).__call__ = _warmup_asgi_wrapper


if __name__ == "__main__":
    logger.info("Starting GitHub Copilot Responses agent")
    app.run()
