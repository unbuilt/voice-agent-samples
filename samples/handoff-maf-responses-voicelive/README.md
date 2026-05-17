# Agent Framework Handoff — Responses Protocol (VoiceLive)

Multi-agent customer support handoff demo using Microsoft Agent Framework's `HandoffBuilder`, served as a hosted agent through the Responses protocol. Optimized for voice-first scenarios.

## Architecture

Three specialized agents with explicit handoff topology via `HandoffBuilder`:

- **Triage Agent** — Routes requests to the appropriate specialist
- **Refund Agent** — Handles refund requests (lookup order, submit refund)
- **Order Agent** — Handles replacements/exchanges (submit replacement)

Tools are auto-approved (no interrupts) for seamless voice interaction.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FOUNDRY_PROJECT_ENDPOINT` | Yes | Foundry project endpoint (auto-injected in hosted containers) |
| `OPENAI_API_KEY` | Yes | API key for the LLM provider |
| `OPENAI_BASE_URL` | No | Base URL for the LLM (default: `https://api.openai.com/v1`) |
| `MODEL` | No | Model name (default: `gpt-4o`) |

## Local Development

```bash
cp .env.example .env
# Edit .env with your values
pip install -r requirements.txt
python main.py
```

## Test

```bash
# Streaming
curl -sS -N -X POST http://localhost:8088/responses \
    -H "Content-Type: application/json" \
    -d '{"input": "I need a refund for order 12345", "stream": true}'
```

## Deploy

```bash
azd ai agent run
```
