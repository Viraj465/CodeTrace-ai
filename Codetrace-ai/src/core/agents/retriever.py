"""
Agent Orchestrator: The Reasoning Engine (Agentic Loop).

Pure httpx implementation — zero LangChain dependency.
Each tool call is dispatched locally; the LLM API is called via direct
HTTP requests, supporting OpenAI-compatible and Anthropic-native APIs.

Tools available:
  - search_codebase : semantic hybrid search over indexed code
  - get_symbol_relations : graph traversal (callers / dependencies)
  - read_file : read full file contents
  - analyze_impact : find all downstream dependents of a symbol
"""

import copy
import os
import httpx
import json
import asyncio
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Generator

from .prompts import SYSTEM_PROMPT
# Use relative imports to ensure IDEs resolve them correctly and avoid missing-import errors.
from ..graph.builder import CodeGraph
from ...backend.vector_store import VectorStore
from .tools import (
    create_tool_schemas,
    create_anthropic_tool_schemas,
    dispatch_tool,
    inspect_index_impl,
)
# Token budget management: Counts input/output tokens, compresses history, and truncates oversized tool results.
from .token_manager import TokenBudgetManager, TurnUsage



# Provider Registry
# Maps provider name → base URL, default model, and API style.
# Config keys (provider, api_key, model_name, base_url) stay unchanged.


PROVIDER_REGISTRY: Dict[str, Dict[str, str]] = {
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-3-5-sonnet-20241022",
        "api_style": "anthropic",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.5-flash",
        "api_style": "openai",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "api_style": "openai",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini-2024-07-18",
        "api_style": "openai",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "",
        "api_style": "openai",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "google/gemini-2.5-flash",
        "api_style": "openai",
    },
    "custom": {
        "base_url": "https://openrouter.ai/api/v1",
        # deepseek/deepseek-chat via OpenRouter: best coding model at near-zero cost.
        "default_model": "deepseek/deepseek-chat",
        "api_style": "openai",
    },
    # "deepseek": {
    #     "base_url": "https://api.deepseek.com",
    #     # deepseek-chat maps to DeepSeek-V3, cheapest frontier coding model available.
    #     "default_model": "deepseek-chat",
    #     "api_style": "openai",
    # },
    # "mistral": {
    #     "base_url": "https://api.mistral.ai/v1",
    #     # codestral-latest: Mistral's dedicated code model, actively maintained.
    #     "default_model": "codestral-latest",
    #     "api_style": "openai",
    # },
    # "together": {
    #     "base_url": "https://api.together.xyz/v1",
    #     # Meta-Llama-3.3-70B-Instruct-Turbo replaced the 3.1-70B Turbo on Together.
    #     # Llama 3.1 70B Turbo was deprecated; 3.3 is the active successor.
    #     "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    #     "api_style": "openai",
    # },
    # "fireworks": {
    #     "base_url": "https://api.fireworks.ai/inference/v1",
    #     # qwen2p5-coder-32b: Fireworks' recommended coding model, fast inference.
    #     "default_model": "accounts/fireworks/models/qwen2p5-coder-32b-instruct",
    #     "api_style": "openai",
    # },
    # "sambanova": {
    #     "base_url": "https://api.sambanova.ai/v1",
    #     # Llama-3.3-70B replaces 3.1-405B on SambaNova; 405B was deprecated 2024-12.
    #     "default_model": "Meta-Llama-3.3-70B-Instruct",
    #     "api_style": "openai",
    # },
    # "siliconflow": {
    #     "base_url": "https://api.siliconflow.cn/v1",
    #     # DeepSeek-V3 on SiliconFlow: best coding model, very low cost.
    #     "default_model": "deepseek-ai/DeepSeek-V3",
    #     "api_style": "openai",
    # },
    # "hyperbolic": {
    #     "base_url": "https://api.hyperbolic.xyz/v1",
    #     # Llama-3.3-70B replaces 3.1-405B on Hyperbolic; 405B deprecated 2024-12.
    #     "default_model": "meta-llama/Llama-3.3-70B-Instruct",
    #     "api_style": "openai",
    # },
}


# Providers that require tool-schema sanitization.
# OpenAI natively supports "strict" and "additionalProperties"; every other
# OpenAI-compat provider (Gemini, Groq, Ollama, OpenRouter …) does not.

_STRICT_SCHEMA_PROVIDERS = {"openai"}


# Provider-specific temperature defaults.
# Gemini 2.5 thinking models exhibit erratic behaviour below 1.0 — keep it
# at exactly 1.0 for all Gemini calls.  Everything else uses 0.3.

_PROVIDER_TEMPERATURE: Dict[str, float] = {
    "gemini": 1.0,
}
_DEFAULT_TEMPERATURE = 0.3



# Human-friendly tool call descriptions


TOOL_THOUGHT_TEMPLATES = {
    "search_codebase":       "Searching codebase for: {query}",
    "inspect_index":         "Inspecting indexed files for: {query}",
    "get_symbol_relations":  "Tracing relationships of {symbol_id}",
    "read_file":             "Reading {file_path}",
    "analyze_impact":        "Analyzing downstream impact of {symbol_id}",
}


def _humanize_tool_call(tool_name: str, tool_input: dict) -> str:
    """Convert a raw tool call into a human-readable thought chain message."""
    template = TOOL_THOUGHT_TEMPLATES.get(tool_name)
    if template:
        try:
            return template.format(**tool_input)
        except KeyError:
            pass
    return f"Using {tool_name}..."


def _normalize_text_content(content: Any) -> str:
    """
    Normalize provider-specific content payloads into plain text.
    Different chat providers may return `str`, list-of-parts, dict payloads,
    or rich content objects.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, (list, tuple)):
        return "".join(_normalize_text_content(part) for part in content)
    if isinstance(content, dict):
        for key in ("text", "content", "output_text", "value"):
            if key in content:
                return _normalize_text_content(content.get(key))
        if "parts" in content:
            return _normalize_text_content(content.get("parts"))
        return str(content)

    text_attr = getattr(content, "text", None)
    if text_attr is not None:
        return _normalize_text_content(text_attr)

    content_attr = getattr(content, "content", None)
    if content_attr is not None and content_attr is not content:
        return _normalize_text_content(content_attr)

    return str(content)



# AgentOrchestrator: httpx-powered agentic loop


class AgentOrchestrator:
    """
    Tool-calling agent loop powered by direct HTTP calls via httpx.

    Supports all major LLM providers:
    - OpenAI-compatible: OpenAI, Groq, Gemini, DeepSeek, Mistral, Ollama,
      OpenRouter, Together, Fireworks, SambaNova, SiliconFlow, Hyperbolic
    - Native Anthropic Messages API

    Config is read from ~/.codetrace/config.json with the same keys as before:
    provider, api_key, model_name, base_url.
    """

    def __init__(self, vector_store: VectorStore, graph: CodeGraph):
        self.vector_store = vector_store
        self.graph = graph
        self.recursion_limit = int(os.getenv("CODETRACE_RECURSION_LIMIT", "60"))

        # Load config and resolve provider settings
        self.config = self._get_global_config()
        self.provider = self.config.get("provider", "").lower()
        self.api_key = self.config.get("api_key", "")

        provider_info = PROVIDER_REGISTRY.get(self.provider)
        if not provider_info:
            raise ValueError(
                f"Unsupported provider: '{self.provider}'. "
                f"Available: {', '.join(sorted(PROVIDER_REGISTRY))}"
            )

        self.api_style = provider_info["api_style"]
        self.model = self.config.get("model_name", "") or provider_info["default_model"]
        self.base_url = self.config.get("base_url", "") or provider_info["base_url"]

        # Validate API key (Ollama runs locally, so no key is needed).
        if not self.api_key and self.provider not in ("ollama",):
            raise ValueError(f"API key missing for provider: {self.provider}")

        # Handle custom provider with missing base_url by prompting the user.
        if self.provider == "custom" and not self.config.get("base_url"):
            self._prompt_custom_base_url()

        # Build HTTP client and tool schemas
        self.client = self._build_client()
        self.tool_schemas = create_tool_schemas()
        self.anthropic_tool_schemas = (
            create_anthropic_tool_schemas() if self.api_style == "anthropic" else None
        )

        # Pre-sanitize tool schemas for this provider so we don't repeat the
        # work on every single LLM call in the agentic loop.
        self._sanitized_tool_schemas = self._sanitize_tools_for_provider(self.tool_schemas)

        # Per-provider temperature (see _PROVIDER_TEMPERATURE for rationale).
        self._temperature = _PROVIDER_TEMPERATURE.get(self.provider, _DEFAULT_TEMPERATURE)

        # Token budget manager
        self.budget = TokenBudgetManager(model_name=self.model)

    
    # Config helpers
    

    def _get_global_config(self) -> Dict[str, Any]:
        """Reads the global LLM config from the user's home directory."""
        config_path = Path.home() / ".codetrace" / "config.json"
        if not config_path.exists():
            raise ValueError("LLM not configured. Run 'codetrace config' first.")
        with open(config_path, "r") as f:
            return json.load(f)

    def _prompt_custom_base_url(self) -> None:
        """
        Prompt user for a custom base_url when one is not set in config.
        Defaults to OpenRouter if the user presses Enter without input.
        Saves the choice back to ~/.codetrace/config.json for future runs.
        """
        print("\n🌐 No base_url found for custom provider.")
        print("Press Enter to use default (OpenRouter: https://openrouter.ai/api/v1)")
        user_input = input("Or enter your custom base_url: ").strip()

        self.base_url = user_input if user_input else "https://openrouter.ai/api/v1"

        self.config["base_url"] = self.base_url
        config_path = Path.home() / ".codetrace" / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def _build_client(self) -> httpx.AsyncClient:
        """
        Build a reusable httpx AsyncClient with provider-specific headers.
        Anthropic uses x-api-key + anthropic-version headers.
        All others use the standard Authorization: Bearer pattern.
        """
        headers = {"Content-Type": "application/json"}

        if self.api_style == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.provider in ("custom", "openrouter"):
            headers["HTTP-Referer"] = "https://github.com/Viraj465/CodeTrace-ai"
            headers["X-Title"] = "CodeTrace-ai"

        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(120.0, connect=15.0),
        )

    
    # Tool-schema sanitization
    

    def _sanitize_tools_for_provider(self, tools: list[dict]) -> list[dict]:
        """
        Strip fields from tool schemas that only OpenAI natively supports.

        Gemini's OpenAI-compat endpoint (and Groq, Ollama, OpenRouter) reject:
          - "strict": true  at the tool or function level
          - "additionalProperties": false  anywhere in the JSON schema
          - "$schema" declarations
          - Any unknown top-level function-definition keys

        This runs once at __init__ time; the result is cached in
        self._sanitized_tool_schemas so the hot agentic loop never pays the
        copy cost.
        """
        if self.provider in _STRICT_SCHEMA_PROVIDERS:
            # OpenAI supports all of these natively — return as-is.
            return tools

        def _clean_schema(schema: dict) -> dict:
            """Recursively remove unsupported JSON-Schema fields."""
            schema = dict(schema)
            schema.pop("additionalProperties", None)
            schema.pop("$schema", None)

            if "properties" in schema and isinstance(schema["properties"], dict):
                schema["properties"] = {
                    k: _clean_schema(v) if isinstance(v, dict) else v
                    for k, v in schema["properties"].items()
                }
            if "items" in schema and isinstance(schema["items"], dict):
                schema["items"] = _clean_schema(schema["items"])
            # Handle union types
            for union_key in ("anyOf", "oneOf", "allOf"):
                if union_key in schema and isinstance(schema[union_key], list):
                    schema[union_key] = [
                        _clean_schema(s) if isinstance(s, dict) else s
                        for s in schema[union_key]
                    ]
            return schema

        sanitized = []
        for tool in copy.deepcopy(tools):
            # Strip top-level "strict" (OpenAI structured-outputs flag)
            tool.pop("strict", None)
            if "function" in tool and isinstance(tool["function"], dict):
                fn = tool["function"]
                fn.pop("strict", None)
                if "parameters" in fn and isinstance(fn["parameters"], dict):
                    fn["parameters"] = _clean_schema(fn["parameters"])
            sanitized.append(tool)
        return sanitized

    
    # Message building
    

    def _extract_index_queries(self, query: str) -> list[str]:
        """
        Build index-inspection queries from user input so coverage checks happen
        automatically without relying on user phrasing.
        """
        lowered = query.lower()
        candidates = [""]

        path_hints = {"src", "lib", "app", "apps", "routes", "router", "components"}
        for hint in path_hints:
            if hint in lowered:
                candidates.append(hint)

        raw_tokens = re.findall(r"[A-Za-z0-9_./\\\\-]+", query)
        for token in raw_tokens:
            t = token.strip().strip("'\"")
            if not t:
                continue
            if "/" in t or "\\" in t:
                candidates.append(t)
            elif t.lower() in path_hints:
                candidates.append(t.lower())

        seen: set[str] = set()
        ordered: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        return ordered[:4]

    def _build_auto_index_context(self, query: str) -> str:
        """
        Run lightweight index coverage preflight so the agent stays DB-first
        even when the user does not explicitly request inspect_index.
        """
        reports = []
        for q in self._extract_index_queries(query):
            limit = 200 if q else 120
            result = inspect_index_impl(query=q, limit=limit)
            label = q or "<all>"
            reports.append(f"[inspect_index query={label!r}]\n{result}")
        return "\n\n".join(reports)

    def _build_messages(self, query: str, chat_history: list | None = None) -> list[dict]:
        """
        Build the conversation message list as plain dicts.
        Format: [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        messages: list[dict] = []

        auto_index_context = self._build_auto_index_context(query)
        system_content = (
            f"{SYSTEM_PROMPT}\n\n"
            "DB-only enforcement: Use only indexed DB evidence. "
            "Do not assume filesystem access. "
            "If evidence is missing, explicitly ask for re-index.\n\n"
            f"Automatic index preflight:\n{auto_index_context}"
        )
        messages.append({"role": "system", "content": system_content})

        if chat_history:
            for role, content in chat_history:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": query})
        return messages

    
    # Error helpers
    

    @staticmethod
    async def _raise_for_status_with_body(resp: httpx.Response) -> None:
        """
        Read the response body *before* raising so the caller sees the full
        provider error message rather than just "400 Bad Request".

        Without this, httpx's raise_for_status() discards the body on error,
        which is how the original code was silently eating Gemini's detailed
        explanation of what was wrong.
        """
        if resp.status_code >= 400:
            # aread() is safe here because we haven't started streaming yet.
            body = await resp.aread()
            try:
                detail = json.loads(body).get("error", {}).get("message", body.decode(errors="replace"))
            except Exception:
                detail = body.decode(errors="replace")
            raise httpx.HTTPStatusError(
                f"API error {resp.status_code}: {detail}",
                request=resp.request,
                response=resp,
            )

    
    # OpenAI-Compatible API
    

    async def _openai_completion(
        self,
        messages: list[dict],
        stream: bool = False,
        force_text: bool = False,
    ):
        """
        Send a request to any OpenAI-compatible /chat/completions endpoint.
        Works for: OpenAI, Groq, Gemini, DeepSeek, Mistral, Ollama,
                   OpenRouter, Together, Fireworks, SambaNova, SiliconFlow, Hyperbolic.

        When *force_text* is True, tools are omitted so the model MUST
        respond with text instead of making more tool calls.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": 4096,
        }
        if not force_text:
            payload["tools"] = self._sanitized_tool_schemas
            payload["tool_choice"] = "auto"

        if stream:
            payload["stream"] = True
            # Returns an async context manager — caller must use `async with`
            return self.client.stream("POST", "/chat/completions", json=payload)

        resp = await self.client.post("/chat/completions", json=payload)
        await self._raise_for_status_with_body(resp)
        return resp.json()

    @staticmethod
    def _extract_thought_signature(tc: dict) -> str:
        """Extract thought_signature from various nested structures in OpenAI compat payloads."""
        if not isinstance(tc, dict):
            return ""

        # 1. Direct or camelCase keys in tc
        for key in ("thought_signature", "thoughtSignature", "thought-signature"):
            val = tc.get(key)
            if val and isinstance(val, str):
                return val

        # 2. Nested under tc["function"]
        fn = tc.get("function")
        if isinstance(fn, dict):
            for key in ("thought_signature", "thoughtSignature", "thought-signature"):
                val = fn.get(key)
                if val and isinstance(val, str):
                    return val
            # Check extra_content inside function
            extra = fn.get("extra_content")
            if isinstance(extra, dict):
                google = extra.get("google")
                if isinstance(google, dict):
                    for key in ("thought_signature", "thoughtSignature", "thought-signature"):
                        val = google.get(key)
                        if val and isinstance(val, str):
                            return val

        # 3. Nested under tc["extra_content"]["google"]
        extra = tc.get("extra_content")
        if isinstance(extra, dict):
            google = extra.get("google")
            if isinstance(google, dict):
                for key in ("thought_signature", "thoughtSignature", "thought-signature"):
                    val = google.get(key)
                    if val and isinstance(val, str):
                        return val

        return ""

    def _parse_openai_response(self, data: dict):
        """
        Parse a non-streamed OpenAI-compatible response.
        Returns (tool_calls: list[dict], text: str).

        Guards against:
        - Missing/empty 'choices' (Ollama returns bare error dicts on bad model names)
        - Malformed/empty tool argument JSON (Groq/OpenRouter can send "" or None)
        """
        choices = data.get("choices") or []
        if not choices:
            error_msg = data.get("error", {}).get("message", "") or str(data)
            raise RuntimeError(f"LLM returned no choices: {error_msg}")

        choice = choices[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""

        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            raw_args = tc.get("function", {}).get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                args = {}
            # Preserve thought_signature so the non-streaming agent loop can
            # echo it back correctly on the follow-up request.
            sig = self._extract_thought_signature(tc)
            tool_calls.append({
                "id": tc.get("id", ""),
                "name": tc.get("function", {}).get("name", ""),
                "args": args,
                "thought_signature": sig,
            })
        return tool_calls, text

    async def _stream_openai(self, messages: list[dict], force_text: bool = False):
        """
        Stream an OpenAI-compatible completion, yielding normalized events:
          {"type": "content_delta", "text": "..."}
          {"type": "tool_call", "id": "...", "name": "...", "args": {...}}
        """
        tool_calls_acc: dict[int, dict] = {}
        active_index = 0  # Track the current index to handle missing index chunks

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": 4096,
            "stream": True,
        }
        if not force_text:
            payload["tools"] = self._sanitized_tool_schemas
            payload["tool_choice"] = "auto"

        async with self.client.stream(
            "POST", "/chat/completions",
            json=payload,
        ) as resp:
            await self._raise_for_status_with_body(resp)

            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                # Content token
                if delta.get("content"):
                    yield {"type": "content_delta", "text": delta["content"]}

                # Tool call chunks (accumulated across deltas).
                for tc_chunk in delta.get("tool_calls", []):
                    # Use the last active index if the API omits it on this chunk
                    idx = tc_chunk.get("index")
                    if idx is None:
                        idx = active_index
                    else:
                        active_index = idx

                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "", "name": "", "arguments": "",
                            "thought_signature": "",
                        }
                    
                    acc = tool_calls_acc[idx]
                    if "id" in tc_chunk and tc_chunk["id"]:
                        acc["id"] = tc_chunk["id"]
                    
                    fn = tc_chunk.get("function", {})
                    if "name" in fn and fn["name"]:
                        acc["name"] = fn["name"]
                    if "arguments" in fn:
                        acc["arguments"] += fn["arguments"]
                    
                    # Accumulate the thought_signature instead of overwriting it
                    sig = self._extract_thought_signature(tc_chunk)
                    if sig:
                        acc["thought_signature"] += sig

        # Yield fully-assembled tool calls after the stream completes.
        for idx in sorted(tool_calls_acc):
            tc = tool_calls_acc[idx]
            call_id = tc["id"] or f"call_{uuid.uuid4().hex[:8]}"
            raw_args = tc.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                args = {}
                
            yield {
                "type": "tool_call",
                "id": call_id,
                "name": tc["name"],
                "args": args,
                "thought_signature": tc.get("thought_signature", ""),
            }
    
    # Anthropic Native API
    

    def _split_system_messages(self, messages: list[dict]):
        """
        Anthropic requires system messages as a top-level 'system' field
        rather than in the messages array. This extracts them.
        Returns (system_text: str, conversation: list[dict]).
        """
        system_parts = []
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                conversation.append(msg)
        return "\n\n".join(system_parts), conversation

    async def _anthropic_completion(self, messages: list[dict], stream: bool = False):
        """
        Send a request to the native Anthropic /v1/messages endpoint.
        System messages are extracted to the top-level 'system' field.
        Tool schemas use Anthropic's 'input_schema' format.
        """
        system_text, conversation = self._split_system_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "system": system_text,
            "messages": conversation,
            "tools": self.anthropic_tool_schemas,
            "max_tokens": 4096,
            "temperature": self._temperature,
        }

        if stream:
            payload["stream"] = True
            return self.client.stream("POST", "/v1/messages", json=payload)

        resp = await self.client.post("/v1/messages", json=payload)
        await self._raise_for_status_with_body(resp)
        return resp.json()

    def _parse_anthropic_response(self, data: dict):
        """
        Parse a non-streamed Anthropic response.
        Returns (tool_calls: list[dict], text: str).
        """
        text_parts = []
        tool_calls = []

        for block in data.get("content", []):
            if block["type"] == "text":
                text_parts.append(block["text"])
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "name": block["name"],
                    "args": block["input"],
                })
        return tool_calls, "".join(text_parts)

    async def _stream_anthropic(self, messages: list[dict]):
        """
        Stream an Anthropic completion, yielding normalized events:
          {"type": "content_delta", "text": "..."}
          {"type": "tool_call", "id": "...", "name": "...", "args": {...}}

        Parses Anthropic's SSE event types: content_block_start,
        content_block_delta (text_delta / input_json_delta), message_stop.
        """
        system_text, conversation = self._split_system_messages(messages)
        tool_calls_acc: dict[int, dict] = {}

        async with self.client.stream(
            "POST", "/v1/messages",
            json={
                "model": self.model,
                "system": system_text,
                "messages": conversation,
                "tools": self.anthropic_tool_schemas,
                "max_tokens": 4096,
                "temperature": self._temperature,
                "stream": True,
            },
        ) as resp:
            await self._raise_for_status_with_body(resp)
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith("data: "):
                    continue

                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")

                # New content block (text or tool_use)
                if etype == "content_block_start":
                    block = event["content_block"]
                    idx = event["index"]
                    if block["type"] == "tool_use":
                        tool_calls_acc[idx] = {
                            "id": block["id"],
                            "name": block["name"],
                            "input_json": "",
                        }

                # Delta within a content block
                elif etype == "content_block_delta":
                    idx = event["index"]
                    delta = event["delta"]
                    if delta["type"] == "text_delta":
                        yield {"type": "content_delta", "text": delta["text"]}
                    elif delta["type"] == "input_json_delta":
                        if idx in tool_calls_acc:
                            tool_calls_acc[idx]["input_json"] += delta["partial_json"]

                # Message complete
                elif etype == "message_stop":
                    break

        for idx in sorted(tool_calls_acc):
            tc = tool_calls_acc[idx]
            yield {
                "type": "tool_call",
                "id": tc["id"],
                "name": tc["name"],
                "args": json.loads(tc["input_json"]) if tc["input_json"] else {},
            }

    
    # Agent Loop (Non-Streaming)
    

    async def _run_agent_loop(self, messages: list[dict]) -> tuple[str, TurnUsage]:
        """
        Execute the tool-calling agent loop without streaming.
        Returns (final_text, TurnUsage).
        """
        total_tool_calls = 0
        had_tool_calls = False
        force_text = False

        for _ in range(self.recursion_limit):
            messages = self.budget.enforce_budget(messages)

            if self.api_style == "anthropic":
                data = await self._anthropic_completion(messages)
                tool_calls, text = self._parse_anthropic_response(data)
            else:
                data = await self._openai_completion(messages, force_text=force_text)
                tool_calls, text = self._parse_openai_response(data)

            if not tool_calls:
                if not text and had_tool_calls and not force_text:
                    force_text = True
                    continue
                usage = self.budget.record_turn_from_response(
                    messages, data, text or "", total_tool_calls
                )
                return text or "", usage

            force_text = False
            had_tool_calls = True
            total_tool_calls += len(tool_calls)

            if self.api_style == "anthropic":
                assistant_content = []
                if text:
                    assistant_content.append({"type": "text", "text": text})
                for tc in tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    })
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for tc in tool_calls:
                    result = dispatch_tool(tc["name"], tc["args"], self.vector_store, self.graph)
                    result = self.budget.truncate_tool_result(result, tc["name"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})

            else:
                # Find turn-level thought_signature (if any) to propagate to all parallel calls
                turn_thought_sig = next(
                    (tc["thought_signature"] for tc in tool_calls if tc.get("thought_signature")),
                    ""
                )
                messages.append({
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                                # Echo thought_signature verbatim — Gemini 2.5/3 thinking
                                # models require it in the follow-up request or they
                                # return 400 INVALID_ARGUMENT.
                                **({"thought_signature": tc["thought_signature"] or turn_thought_sig}
                                   if (tc.get("thought_signature") or turn_thought_sig) else {}),
                            },
                        }
                        for tc in tool_calls
                    ],
                })
                for tc in tool_calls:
                    result = dispatch_tool(tc["name"], tc["args"], self.vector_store, self.graph)
                    result = self.budget.truncate_tool_result(result, tc["name"])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

        fallback = "⚠️ Reached maximum reasoning steps. Please try a more specific query."
        usage = self.budget.estimate_streaming_turn(messages, fallback, total_tool_calls)
        return fallback, usage

    
    # Agent Loop (Streaming)
    

    async def _stream_agent_loop(self, messages: list[dict]):
        """
        Execute the tool-calling agent loop WITH streaming.
        Yields structured event dicts:

        {"type": "thought",  "tool": "search_codebase", "message": "Searching for: auth"}
        {"type": "tool_end", "tool": "search_codebase"}
        {"type": "token",    "content": "The"}
        {"type": "usage",    "turn": TurnUsage, "session": SessionUsage}
        {"type": "done"}
        """
        total_tool_calls = 0
        messages_at_start = list(messages)
        all_output_parts: list[str] = []

        had_tool_calls = False
        force_text = False

        for _ in range(self.recursion_limit):
            content_parts: list[str] = []
            tool_calls: list[dict] = []

            messages = self.budget.enforce_budget(messages)

            streamer = (
                self._stream_anthropic(messages)
                if self.api_style == "anthropic"
                else self._stream_openai(messages, force_text=force_text)
            )
            async for event in streamer:
                if event["type"] == "content_delta":
                    yield {"type": "token", "content": event["text"]}
                    content_parts.append(event["text"])
                elif event["type"] == "tool_call":
                    tool_calls.append(event)

            if not tool_calls:
                if not content_parts and had_tool_calls and not force_text:
                    force_text = True
                    continue
                all_output_parts.extend(content_parts)
                break

            force_text = False
            had_tool_calls = True
            all_output_parts.extend(content_parts)

            text = "".join(content_parts)
            total_tool_calls += len(tool_calls)

            if self.api_style == "anthropic":
                assistant_content = []
                if text:
                    assistant_content.append({"type": "text", "text": text})
                for tc in tool_calls:
                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["args"],
                    })
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for tc in tool_calls:
                    yield {
                        "type": "thought",
                        "tool": tc["name"],
                        "message": _humanize_tool_call(tc["name"], tc["args"]),
                    }
                    result = dispatch_tool(
                        tc["name"], tc["args"],
                        self.vector_store, self.graph,
                    )
                    yield {"type": "tool_end", "tool": tc["name"]}
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})

            else:
                # Find turn-level thought_signature (if any) to propagate to all parallel calls
                turn_thought_sig = next(
                    (tc["thought_signature"] for tc in tool_calls if tc.get("thought_signature")),
                    ""
                )
                messages.append({
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                                **({"thought_signature": tc["thought_signature"] or turn_thought_sig}
                                   if (tc.get("thought_signature") or turn_thought_sig) else {}),
                            },
                        }
                        for tc in tool_calls
                    ],
                })
                for tc in tool_calls:
                    yield {
                        "type": "thought",
                        "tool": tc["name"],
                        "message": _humanize_tool_call(tc["name"], tc["args"]),
                    }
                    result = dispatch_tool(tc["name"], tc["args"], self.vector_store, self.graph)
                    result = self.budget.truncate_tool_result(result, tc["name"])
                    yield {"type": "tool_end", "tool": tc["name"]}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

        output_text = "".join(all_output_parts)
        turn_usage = self.budget.estimate_streaming_turn(
            messages_at_start, output_text, total_tool_calls
        )
        yield {"type": "usage", "turn": turn_usage, "session": self.budget.session}
        yield {"type": "done"}

    
    # Public API
    

    def ask(self, query: str, chat_history: list | None = None) -> str:
        """
        Blocking execution of the agentic pipeline.
        Returns the final text response.
        Token usage is recorded in self.budget.session for later inspection.
        """
        messages = self._build_messages(query, chat_history)

        async def _run():
            text, _usage = await self._run_agent_loop(messages)
            return text

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(_run())
        else:
            return asyncio.run(_run())

    async def astream(self, query: str, chat_history: list | None = None):
        """
        Async streaming — yields structured event dicts for the CLI.
        Same format as before: thought → tool_end → token → done.
        """
        messages = self._build_messages(query, chat_history)
        async for event in self._stream_agent_loop(messages):
            yield event

    def stream(self, query: str, chat_history: list | None = None) -> Generator[dict, None, None]:
        """
        Synchronous wrapper around astream() for use in non-async CLI code.

        Uses a Queue to bridge the async generator and the sync caller so that
        events are yielded in real time as the LLM streams them — NOT buffered
        and flushed all at once after the full response completes.
        """
        import queue
        import threading

        _SENTINEL = object()
        event_queue: queue.Queue = queue.Queue()

        async def _producer():
            try:
                async for event in self.astream(query, chat_history):
                    event_queue.put(event)
            except Exception as exc:
                event_queue.put({"type": "error", "message": str(exc)})
            finally:
                event_queue.put(_SENTINEL)

        def _run_producer():
            asyncio.run(_producer())

        thread = threading.Thread(target=_run_producer, daemon=True)
        thread.start()

        while True:
            item = event_queue.get()
            if item is _SENTINEL:
                break
            yield item

        thread.join()