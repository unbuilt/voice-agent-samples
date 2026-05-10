# Copyright (c) Microsoft. All rights reserved.

import os
import sys
from datetime import datetime, timezone

from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from agent_framework.openai import OpenAIChatCompletionClient
from agent_framework_foundry_hosting import ResponsesHostServer
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()

from azure.identity.aio import get_bearer_token_provider

_CREDENTIAL = DefaultAzureCredential()
_token_provider = get_bearer_token_provider(_CREDENTIAL, "https://ai.azure.com/.default")

from agent_framework.openai import OpenAIChatCompletionClient as _BaseClient

import logging
import time
from collections.abc import AsyncIterable, Awaitable, Mapping, Sequence
from agent_framework._types import ChatResponse, ChatResponseUpdate, Content, Message, ResponseStream

logger = logging.getLogger(__name__)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

class SafeOpenAIChatCompletionClient(_BaseClient):
    """Patches the streaming bug and adds latency logging."""

    def _parse_text_from_openai(self, choice):
        message = choice.message if hasattr(choice, "message") and choice.message is not None else getattr(choice, "delta", None)
        if message is None:
            return None
        if message.content:
            return Content.from_text(text=message.content, raw_representation=choice)
        if hasattr(message, "refusal") and message.refusal:
            return Content.from_text(text=message.refusal, raw_representation=choice)
        return None

    def _inner_get_response(self, *, messages, options, stream=False, **kwargs):
        options_dict = self._prepare_options(messages, options)

        if stream:
            options_dict["stream_options"] = {"include_usage": True}

            async def _stream() -> AsyncIterable[ChatResponseUpdate]:
                t0 = time.perf_counter()
                first_chunk = True
                client = self.client
                try:
                    async for chunk in await client.chat.completions.create(stream=True, **options_dict):
                        if len(chunk.choices) == 0 and chunk.usage is None:
                            continue
                        if first_chunk:
                            logger.info("LLM time-to-first-token: %.3fs", time.perf_counter() - t0)
                            first_chunk = True  # log only once
                            first_chunk = False
                        yield self._parse_response_update_from_openai(chunk)
                except Exception as ex:
                    from agent_framework.exceptions import ChatClientException
                    raise ChatClientException(
                        f"{type(self)} service failed: {ex}", inner_exception=ex
                    ) from ex
                finally:
                    logger.info("LLM total stream duration: %.3fs", time.perf_counter() - t0)

            return self._build_response_stream(_stream(), response_format=options.get("response_format"))

        # Non-streaming
        original = super()._inner_get_response(messages=messages, options=options, stream=False, **kwargs)

        async def _timed():
            t0 = time.perf_counter()
            result = await original
            logger.info("LLM non-streaming latency: %.3fs", time.perf_counter() - t0)
            return result

        return _timed()
    

from agent_framework_foundry_hosting import ResponsesHostServer as _BaseServer
import asyncio

class WarmupResponsesHostServer(_BaseServer):
    """ResponsesHostServer that warms up the LLM connection on startup."""

    def __init__(self, agent, warmup_client=None, warmup_model=None, **kwargs):
        super().__init__(agent, **kwargs)
        self._warmup_client = warmup_client
        self._warmup_model = warmup_model

    # Warm up N connections concurrently
    async def _warmup_connections(self, client, model, n=2):
        async def _single():
            await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                store=False,
            )
        await asyncio.gather(*[_single() for _ in range(n)])        

    def run(self, host="0.0.0.0", port=None):
        from hypercorn.asyncio import serve as _hypercorn_serve

        resolved_port = int(os.environ.get("PORT", port or 8088))
        config = self._build_hypercorn_config(host, resolved_port)

        async def _run():
            # Warm up the async LLM connection on this event loop
            if False and self._warmup_client and self._warmup_model:
                logger.info("warmup: sending LLM warmup request at %s", _now())
                try:
                    await self._warmup_client.chat.completions.create(
                        model=self._warmup_model,
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=1,
                        store=False,
                    )
                    logger.info("warmup: LLM connection warm at %s", _now())
                except Exception as exc:
                    logger.warning("warmup: LLM warmup failed at %s: %s", _now(), exc)

            await self._warmup_connections(self._warmup_client, self._warmup_model, n=2)

            await _hypercorn_serve(self, config)

        asyncio.run(_run())


def main():
    # Sync warmup — pre-cache the token
    logger.info("prewarm: acquiring token at %s", _now())
    _CREDENTIAL.get_token("https://ai.azure.com/.default")
    logger.info("prewarm: token acquired at %s", _now())
    
    """
    client = FoundryChatClient(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        credential=DefaultAzureCredential(),
    )
    """

    project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"]
    openai_url = project_endpoint.rstrip("/") + "/openai/v1"

    chat_client = SafeOpenAIChatCompletionClient(
        base_url=openai_url,
        model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        api_key=_token_provider,
    )

    agent = Agent(
        client=chat_client,
        instructions="You are a friendly assistant. Keep your answers brief.",
        # History will be managed by the hosting infrastructure, thus there
        # is no need to store history by the service. Learn more at:
        # https://developers.openai.com/api/reference/resources/responses/methods/create
        default_options={"store": False},
    )

    server = WarmupResponsesHostServer(agent, warmup_client=chat_client.client, warmup_model=os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"])
    server.run()


if __name__ == "__main__":
    main()
