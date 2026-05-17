# LangGraph Swarm Handoff Demo

Turn-by-turn multi-agent handoff using LangGraph Swarm — no interrupts for user questions, no tool approval. Equivalent behavior to the Agent Framework `HandoffBuilder` demo but with simpler execution model suitable for voice-first scenarios.

## Architecture

```
User message → POST /chat → route to active_agent → agent turn (ReAct loop) → response
Next message → POST /chat → same or new active_agent → agent turn → response
```

- **No interrupt/resume** — each request completes naturally
- **No tool approval** — tools execute immediately
- **Active agent tracked** in LangGraph checkpointed state

## Agents & Topology

```
triage_agent ──→ refund_agent ──→ order_agent
     ↑               │                │
     └───────────────┴────────────────┘
```

- **triage_agent**: Routes to refund or order specialist, delivers farewell
- **refund_agent**: Handles refund workflow (lookup → submit_refund → handoff)
- **order_agent**: Handles replacements (shipping preference → submit_replacement → handoff)

## Setup

```bash
cp .env.example .env
# Edit .env with your API key

pip install -r requirements.txt
python server.py
```

## API

### POST /chat
```json
{"thread_id": "abc", "message": "I need a refund for order 12345"}
```
Response:
```json
{"thread_id": "abc", "active_agent": "refund_agent", "response": "..."}
```

### POST /chat/stream
Same request body, returns SSE stream:
```
data: [refund_agent] I found your order...
data: [DONE]
```

### GET /healthz
```json
{"status": "ok"}
```
