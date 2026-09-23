"""
Agent Orchestrator — the reasoning engine that drives the agentic loop.

Built straight on httpx, no LangChain in sight. Tool calls run locally and the
LLM is reached over plain HTTP, so both OpenAI-compatible and Anthropic-native
APIs work.

The tools it can reach for:
  - search_codebase : semantic hybrid search over indexed code
  - get_symbol_relations : walk the graph for callers / dependencies
  - read_file : pull a file's full contents
  - analyze_impact : find everything downstream of a symbol
"""

from huggingface_hub.inference._generated.types import zero_shot_image_classification
import copy
import os
import httpx
import json
import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Generator

from .prompts import SYSTEM_PROMPT, OLLAMA_SYSTEM_PROMPT
# Relative imports so IDEs resolve them correctly and don't flag missing imports.
from ..graph.builder import CodeGraph
from ...backend.vector_store import VectorStore
from ...config_io import write_private_json
from .tools import (
    create_tool_schemas,
    create_anthropic_tool_schemas,
    dispatch_tool,
    inspect_index_impl,
)
# Handles token counting, history compression, and trimming oversized tool results.
from .token_manager import (
    TokenBudgetManager,
    TurnUsage,
    _tier_from_context_window,
    recommended_num_ctx,
    ollama_model_offloaded,
)

from .output_normalizer import OutputNormalizer




logger = logging.getLogger(__name__)


# Everything we need to talk to each provider: base URL, default model, API style.
# The config keys (provider, api_key, model_name, base_url) are unchanged.


PROVIDER_REGISTRY: Dict[str, Dict[str, str]] = {
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-5",
        "api_style": "anthropic",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        # Matches the "Gemini 2.5 thinking models" temperature rationale below
        # and the OpenRouter default (google/gemini-2.5-pro).
        "default_model": "gemini-3.5-flash",
        "api_style": "openai",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1/models",
        "default_model": "openai/gpt-oss-120b",
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
        "default_model": "z-ai/glm-5.3-flash",
        "api_style": "openai",
    },
    # A user-defined endpoint.
    "custom": {
        "base_url": "",
        "default_model": "",
        "api_style": "openai",
    },
}

# Wire formats the agent loop can speak. "openai" posts to {base_url}/chat/completions;
# "anthropic" posts to {base_url}/v1/messages.
SUPPORTED_API_STYLES = ("openai", "anthropic")

# Providers that run locally or may sit behind an unauthenticated gateway, so an
# empty api_key is legitimate rather than a misconfiguration.
_KEYLESS_PROVIDERS = {"ollama", "custom"}


# Which providers need their tool schemas cleaned up before sending. OpenAI
# understands "strict" and "additionalProperties"; the other OpenAI-compat
# providers (Gemini, Groq, Ollama, OpenRouter, …) choke on them.
_STRICT_SCHEMA_PROVIDERS = {"openai"}


# Default temperature per provider. Gemini 2.5/3 thinking models act up below 1.0
# — and on Gemini 3 thinking is always on and can't be turned off — so we pin all
# Gemini calls at exactly 1.0. Everyone else runs at 0.3.
_PROVIDER_TEMPERATURE: Dict[str, float] = {
    "gemini": 1.0,
}
_DEFAULT_TEMPERATURE = 0.3



# Human-readable exception text.


# Exceptions whose str() is famously empty, mapped to something a user can act
# on. httpx's timeout classes are the big one: a bare ReadTimeout stringifies to
# "", which surfaced in the CLI as "Architect Error:" followed by nothing.
_EXC_HINTS = {
    "ReadTimeout":    "the model took too long to send its first token",
    "ConnectTimeout": "could not reach the model endpoint in time",
    "WriteTimeout":   "timed out sending the request",
    "PoolTimeout":    "timed out waiting for a free connection",
    "ConnectError":   "could not connect to the model endpoint",
    "ReadError":      "the connection dropped while reading the response",
    "RemoteProtocolError": "the server closed the connection unexpectedly",
    "MemoryError":    "ran out of system memory",
}


def describe_exception(exc: BaseException) -> str:
    """
    Render an exception as a non-empty, user-facing sentence.

    Several exceptions we hit routinely carry no message at all, so `str(exc)`
    alone can produce an error report with nothing in it. Always fall back to
    the class name plus a hint about what it means.
    """
    name = type(exc).__name__
    detail = str(exc).strip()
    hint = _EXC_HINTS.get(name)

    if detail and hint:
        return f"{hint} ({name}: {detail})"
    if detail:
        return f"{name}: {detail}"
    if hint:
        return f"{hint} ({name})"
    return f"{name} (no further detail)"


# Turning raw tool calls into readable "here's what I'm doing" lines.


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
    Flatten whatever a provider hands back into plain text. Depending on the
    provider, `content` might be a str, a list of parts, a dict, or some richer
    content object.
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
    The tool-calling agent loop, driven by direct httpx calls.

    Works with all the major providers — the OpenAI-compatible crowd (OpenAI,
    Groq, Gemini, DeepSeek, Mistral, Ollama, OpenRouter, Together, Fireworks,
    SambaNova, SiliconFlow, Hyperbolic) plus Anthropic's native Messages API.

    Config comes from ~/.codetrace/config.json using the usual keys: provider,
    api_key, model_name, base_url.
    """

    def __init__(self, vector_store: VectorStore, graph: CodeGraph):
        self.vector_store = vector_store
        self.graph = graph
        self.recursion_limit = int(os.getenv("CODETRACE_RECURSION_LIMIT", "60"))

        # Read the config and figure out which provider we're talking to.
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
        self.base_url = (self.config.get("base_url", "") or provider_info["base_url"]).strip()

        # `custom` is defined entirely by the user's config, so its wire format is
        # configurable too — that's what lets one provider entry reach both
        # OpenAI-compatible endpoints and Anthropic-compatible ones.
        if self.provider == "custom":
            style = str(self.config.get("api_style", "") or "").strip().lower()
            if style:
                if style not in SUPPORTED_API_STYLES:
                    raise ValueError(
                        f"Unsupported api_style '{style}' for the custom provider. "
                        f"Supported: {', '.join(SUPPORTED_API_STYLES)}. "
                        "Fix it with 'codetrace config'."
                    )
                self.api_style = style

        # Local and self-hosted endpoints legitimately have no key; everyone else
        # needs one.
        if not self.api_key and self.provider not in _KEYLESS_PROVIDERS:
            raise ValueError(f"API key missing for provider: {self.provider}")

        # A custom provider with no base_url set — ask for one.
        if self.provider == "custom" and not self.base_url:
            self._prompt_custom_base_url()

        # With no vendor default to fall back on, an unset model would otherwise
        # reach the API as an empty string and fail with an opaque 400.
        if not self.model:
            raise ValueError(
                f"No model configured for provider '{self.provider}'. "
                "Set one with 'codetrace config'."
            )

        # HTTP client and tool schemas.
        # self.client is now a property to ensure event loop affinity
        self.tool_schemas = create_tool_schemas()
        self.anthropic_tool_schemas = (
            create_anthropic_tool_schemas() if self.api_style == "anthropic" else None
        )

        # Sanitize the schemas for this provider once here, rather than on every
        # LLM call inside the loop.
        self._sanitized_tool_schemas = self._sanitize_tools_for_provider(self.tool_schemas)

        # Temperature for this provider (why, see _PROVIDER_TEMPERATURE).
        self._temperature = _PROVIDER_TEMPERATURE.get(self.provider, _DEFAULT_TEMPERATURE)

        # Token budget manager.
        ollama_url = self.base_url if self.provider == "ollama" else None
        # Cloud models that litellm doesn't recognise fall back to this window
        # instead of the cramped 8K "unknown" tier. Ollama ignores it: its tier
        # is rebuilt later from a GPU-safe num_ctx, and a big default there
        # would risk OOM. Priority: saved config (`set-default-ctx`) > env var.
        default_ctx = self.config.get("default_context_window")
        if not (isinstance(default_ctx, int) and default_ctx > 0):
            env_dc = os.getenv("CODETRACE_DEFAULT_CONTEXT_WINDOW")
            if env_dc and env_dc.strip().isdigit() and int(env_dc) > 0:
                default_ctx = int(env_dc)
            else:
                default_ctx = None

        def _positive_int(value) -> int | None:
            return value if isinstance(value, int) and value > 0 else None

        configured_context_window = _positive_int(
            self.config.get("context_window")
        )
        configured_max_output = _positive_int(
            self.config.get("max_output_tokens")
        )

        # Do not let cloud settings alter Ollama's local GPU-sized context.
        if self.provider == "ollama":
            configured_context_window = None
            configured_max_output = None
        self.budget = TokenBudgetManager(
            model_name=self.model,
            ollama_base_url=ollama_url,
            default_context_window=default_ctx,
            context_window_override=configured_context_window,
            max_output_tokens_override=configured_max_output,
        )
        # OutputNormalizer is created AFTER the Ollama tier is finalized below,
        # so it always reflects the actual num_ctx in use (not the model default).
        self.output_normalizer = None

        # Ollama native-endpoint path. The OpenAI-compat /v1 endpoint silently
        # ignores options.num_ctx, so Ollama runs at its default (4096) no matter
        # how large the model's real window is — truncating our prompt server-side.
        # For Ollama only, we talk to the native /api/chat endpoint and pass
        # num_ctx explicitly so the allocated window matches the tier we sized.
        # Every other provider keeps the untouched OpenAI-compat path below.
        self._is_ollama = self.provider == "ollama"
        self._ollama_native_url = ""
        self._ollama_num_ctx = 0
        self._ollama_keep_alive = "30m"
        if self._is_ollama:
            root = self.base_url.rstrip("/")
            if root.endswith("/v1"):
                root = root[:-3]
            self._ollama_native_url = f"{root}/api/chat"

            # Size num_ctx to the detected GPU so we never ask Ollama to allocate
            # a KV cache the hardware can't hold (which would OOM or spill to CPU
            # and crawl). Priority: env var > saved config (`codetrace set-ctx`)
            # > hardware auto-detect. The chosen value is clamped to the model's
            # trained window by recommended_num_ctx.
            model_window = self.budget.tier.context_window
            chosen, detail = recommended_num_ctx(model_window)
            source = f"auto-detected ({detail})"

            cfg_ctx = self.config.get("ollama_num_ctx")
            if isinstance(cfg_ctx, int) and cfg_ctx > 0:
                chosen, source = cfg_ctx, "saved config (codetrace set-ctx)"

            env_ctx = os.getenv("CODETRACE_OLLAMA_NUM_CTX")
            if env_ctx and env_ctx.strip().isdigit() and int(env_ctx) > 0:
                chosen, source = int(env_ctx), "env CODETRACE_OLLAMA_NUM_CTX"

            self._ollama_num_ctx = chosen
            self._ollama_ctx_source = source  # for the CLI header
            # Floor for the adaptive controller — never auto-lower below this.
            self._ollama_min_ctx = 4096
            # Back-off factor: fraction of the window KEPT on each auto-lower step
            # (0.5 = halve). Configurable so users can lower more gently (e.g.
            # 0.75) or more aggressively (0.4). Clamped to a sane 0.25–0.9 band.
            backoff = self.config.get("ollama_ctx_backoff", 0.5)
            try:
                backoff = float(backoff)
            except (TypeError, ValueError):
                backoff = 0.5
            self._ollama_ctx_backoff = min(0.9, max(0.25, backoff))
            # Keep the budget/compression tier consistent with what Ollama will
            # actually allocate, so we never build a prompt bigger than num_ctx.
            # Recompute max_output from the CHOSEN window (passing None) — reusing
            # a large model's max_output against a small num_ctx would leave zero
            # headroom for generation.
            self.budget.tier = _tier_from_context_window(chosen, None)

            # keep_alive holds the model + its KV/prompt cache in memory between
            # turns, so each question isn't a cold reload. Pure latency win — no
            # extra peak memory, no effect on output. Moderate default releases
            # VRAM once the session goes idle; overridable via config.
            cfg_keep = self.config.get("ollama_keep_alive")
            if isinstance(cfg_keep, str) and cfg_keep.strip():
                self._ollama_keep_alive = cfg_keep.strip()

            logger.info(
                "Ollama num_ctx=%d via %s (model window %d); keep_alive=%s",
                chosen, source, model_window, self._ollama_keep_alive,
            )

        # Instantiate the normalizer now that tier is stable for all paths.
        # For non-Ollama providers tier is already correct; for Ollama it was
        # just recomputed from the chosen num_ctx above.
        self.output_normalizer = OutputNormalizer(tier=self.budget.tier.name)

        # Cache for the automatic index-coverage preflight. The index holds still
        # for the length of a chat session, so the same derived query-set can reuse
        # its result instead of hitting the DB again every turn.
        self._auto_index_cache: Dict[tuple, str] = {}

    def _tool_result_budget(
        self,
        messages: list[dict],
        tool_count: int,
    ) -> int | None:
        """
        `remaining_context()` already excludes the reserved final-model output.
        Reserve only a small protocol cushion, then share the rest among tool calls.
        """
        remaining = self.budget.remaining_context(messages)
        protocol_reserve = 256
        minimum_per_tool = 128

        available = remaining - protocol_reserve
        if available < minimum_per_tool * tool_count:
            return None

        return available // tool_count    
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
        Ask for the endpoint when a custom provider has no base_url in config.

        There is deliberately no default here: the whole point of the custom
        provider is that the user names the endpoint, so guessing a vendor would
        silently send their code somewhere they never chose. The answer is saved
        back to ~/.codetrace/config.json so this only happens once.
        """
        print("\n🌐 No base_url configured for the custom provider.")
        if self.api_style == "anthropic":
            print("   Anthropic-style endpoints take the bare host (requests go to /v1/messages),")
            print("   e.g. https://api.anthropic.com")
        else:
            print("   OpenAI-compatible endpoints include the version path (requests go to")
            print("   /chat/completions), e.g. https://api.deepseek.com/v1")

        user_input = ""
        while not user_input:
            user_input = input("   Enter the API base URL: ").strip()
            if not user_input:
                print("   A base URL is required. Press Ctrl+C to abort.")

        self.base_url = user_input.rstrip("/")

        self.config["base_url"] = self.base_url
        config_path = Path.home() / ".codetrace" / "config.json"
        write_private_json(config_path, self.config)

    @property
    def client(self) -> httpx.AsyncClient:
        """
        Lazily initialize the HTTP client bound to the current event loop.
        Prevents "RuntimeError: Event loop is closed" when the agent is
        used across multiple sync queries.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
            
        if getattr(self, "_client_loop", object()) is not loop or not hasattr(self, "_client_instance"):
            self._client_instance = self._build_client()
            self._client_loop = loop
        return self._client_instance

    def _build_client(self) -> httpx.AsyncClient:
        """
        Build one reusable httpx AsyncClient with the right headers for this
        provider. Anthropic wants x-api-key + anthropic-version; everyone else
        uses the usual Authorization: Bearer.
        """
        headers = {"Content-Type": "application/json"}

        if self.api_style == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # OpenRouter's attribution headers. Keyed off the actual endpoint rather
        # than the provider name, so a custom endpoint only gets them when it
        # really is OpenRouter — other vendors reject unknown headers or log them
        # as noise.
        if "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = "https://github.com/Viraj465/CodeTrace-ai"
            headers["X-Title"] = "CodeTrace-ai"

        # Read timeout: the gap between *chunks*, not the length of the whole
        # response, so it only has to cover time-to-first-token. A hosted API
        # answers in seconds, but a local Ollama loading a model into a small
        # GPU — or running one partly offloaded to CPU — can sit silent for
        # several minutes before the first token, and a 120s cap turned that
        # into a bare `httpx.ReadTimeout` (which stringifies to "", hence the
        # blank "Architect Error:"). Give local models the room they need.
        # (self._is_ollama is set after this runs, so read self.provider.)
        read_timeout = 900.0 if self.provider == "ollama" else 180.0

        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(read_timeout, connect=15.0),
        )

    
    # Tool-schema sanitization
    

    def _sanitize_tools_for_provider(self, tools: list[dict]) -> list[dict]:
        """
        Drop the schema fields that only OpenAI actually understands.

        Gemini's OpenAI-compat endpoint — and Groq, Ollama, OpenRouter — all
        reject a handful of things:
          - "strict": true  on a tool or function
          - "additionalProperties": false  anywhere in the JSON schema
          - "$schema" declarations
          - any unknown top-level keys on the function definition

        We do this once in __init__ and cache the result in
        self._sanitized_tool_schemas, so the hot loop never pays for the copy.
        """
        if self.provider in _STRICT_SCHEMA_PROVIDERS:
            # OpenAI takes all of it as-is.
            return tools

        def _clean_schema(schema: dict) -> dict:
            """Walk the schema and strip the unsupported fields as we go."""
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
            # Recurse into union types too.
            for union_key in ("anyOf", "oneOf", "allOf"):
                if union_key in schema and isinstance(schema[union_key], list):
                    schema[union_key] = [
                        _clean_schema(s) if isinstance(s, dict) else s
                        for s in schema[union_key]
                    ]
            return schema

        sanitized = []
        for tool in copy.deepcopy(tools):
            # Drop the top-level "strict" (OpenAI's structured-outputs flag).
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
        Derive a set of index-inspection queries from the user's message, so the
        coverage check fires on its own instead of depending on how they phrased it.
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


    # Preflight cap. Deliberately smaller than what inspect_index allows on demand:
    # this block rides along on EVERY request, so it pays for orientation only —
    # the agent calls the tool itself when it needs the full listing.
    _PREFLIGHT_MAX_PATHS = 80

    def _build_auto_index_context(self, query: str) -> str:
        """
        Do a quick index-coverage check up front, so the agent stays DB-first
        even when the user never asks for inspect_index by name.

        The derived queries overlap heavily — a 'src' filter returns a subset of
        the unfiltered listing — so their results are merged into one
        deduplicated block. Emitting them separately repeated both the metadata
        header and most of the paths up to four times, on every single request.
        """
        queries = self._extract_index_queries(query)
        cache_key = tuple(queries)
        cached = self._auto_index_cache.get(cache_key)
        if cached is not None:
            return cached

        header_lines: list[str] = []
        paths: list[str] = []
        seen: set[str] = set()

        for i, q in enumerate(queries):
            result = inspect_index_impl(query=q, limit=self._PREFLIGHT_MAX_PATHS)
            for line in result.splitlines():
                if line.startswith("- "):
                    path = line[2:].strip()
                    if path and path not in seen:
                        seen.add(path)
                        paths.append(path)
                # Keep the metadata header from the first result only.
                elif i == 0 and line.startswith(
                    ("Project root:", "Supported files indexed:", "Tracked text snapshots:")
                ):
                    header_lines.append(line)

        if not paths:
            # Empty or missing index — keep the raw message so the agent still
            # learns it has to ask for a re-index.
            context = inspect_index_impl(query="", limit=self._PREFLIGHT_MAX_PATHS)
        else:
            shown = paths[:self._PREFLIGHT_MAX_PATHS]
            lines = ["Index preflight — call inspect_index for a fuller or filtered listing:"]
            lines.extend(header_lines)
            lines.extend(f"- {p}" for p in shown)
            if len(paths) > len(shown):
                lines.append(
                    f"... and {len(paths) - len(shown)} more file(s) — "
                    "use inspect_index with a path filter to see them"
                )
            context = "\n".join(lines)

        self._auto_index_cache[cache_key] = context
        return context

    
    def _build_messages(self, query: str, chat_history: list | None = None) -> list[dict]:
        """
        Assemble the conversation as a list of plain dicts, each shaped like
        {"role": "system"|"user"|"assistant", "content": "..."}.
        """
        messages: list[dict] = []

        # The system prompt is emitted as TWO system messages, stable-first, and
        # the split is load-bearing for prompt caching -- do not merge them back
        # into one string here.
        #
        #   [0] stable   -- byte-identical on every request of every session
        #   [1] volatile -- the preflight, which is derived from the query
        #
        # Providers that bill cached input (Anthropic) can then cache through the
        # end of [0]; see _split_system_messages. Ordering matters: caching is a
        # prefix match, so putting the volatile block first would mean nothing
        # before it could ever be reused.
        #
        # Ollama gets its own, much shorter prompt: it states plainly that the
        # code stays on-device (true only on the local path) and it is sized for
        # the small num_ctx local models actually run at.
        base_prompt = OLLAMA_SYSTEM_PROMPT if self._is_ollama else SYSTEM_PROMPT
        stable_system = (
            f"{base_prompt}\n\n"
            "DB-only enforcement: Use only indexed DB evidence. "
            "Do not assume filesystem access. "
            "If evidence is missing, explicitly ask for re-index."
        )
        auto_index_context = self._build_auto_index_context(query)

        messages.append({"role": "system", "content": stable_system})
        messages.append({
            "role": "system",
            "content": f"Automatic index preflight:\n{auto_index_context}",
        })

        if chat_history:
            for role, content in chat_history:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": query})
        return messages

    def _preflight_budget(self, messages: list[dict]) -> tuple[list[dict], str | None]:
        """
        Trim the volatile system message (the auto-index preflight) so the
        initial prompt fits inside the soft budget even before the first LLM
        call.  Returns (messages, warning_text | None).

        Why this matters: compress_history() can only drop *old conversation
        turns*, not fixed system content.  On the first query there is no
        history at all, so if the system prompt + preflight already exceeds the
        budget, compression is a no-op and the context window is effectively
        full before the model sees a single tool result.  Trimming the preflight
        (messages[1]) prevents that without touching the stable prompt or the
        user message.
        """
        from .token_manager import count_messages_tokens

        tokens = count_messages_tokens(messages, self.budget._count_model)
        headroom = self.budget.tier.soft_budget - tokens
        if headroom >= 0:
            return messages, None  # already fits — nothing to do

        # Identify the volatile block (index 1, role=system, starts with the
        # preflight marker).  If the layout ever changes, bail out safely.
        if (
            len(messages) > 1
            and messages[1].get("role") == "system"
            and isinstance(messages[1].get("content"), str)
            and messages[1]["content"].startswith("Automatic index preflight")
        ):
            preflight_content = messages[1]["content"]
            # How many chars to cut: ~4 chars/token, with a small safety margin.
            cut_chars = (-headroom + 64) * 4
            keep = max(200, len(preflight_content) - cut_chars)
            if keep < len(preflight_content):
                trimmed = list(messages)  # shallow copy so the original is clean
                trimmed[1] = dict(messages[1])
                trimmed[1]["content"] = (
                    preflight_content[:keep]
                    + "\n... [preflight truncated to fit context budget]"
                )
                tokens_after = count_messages_tokens(trimmed, self.budget._count_model)
                logger.warning(
                    "Initial prompt (%d tokens) exceeded soft budget (%d); "
                    "trimmed auto-index preflight to %d tokens.",
                    tokens, self.budget.tier.soft_budget, tokens_after,
                )
                remaining_after = self.budget.tier.context_window - tokens_after
                if remaining_after < self.budget.tier.max_output_tokens:
                    warn = (
                        f"⚠  Context window is tight ({tokens_after:,} / "
                        f"{self.budget.tier.context_window:,} tokens used before "
                        f"the first tool call). Responses may be cut short. "
                        f"Run 'codetrace set-default-ctx <N>' to raise the window."
                    )
                    return trimmed, warn
                return trimmed, None

        # Couldn't trim anything meaningful — warn but proceed.
        warn = (
            f"⚠  Initial prompt ({tokens:,} tokens) already fills "
            f"{self.budget.tier.context_window:,}-token window. "
            f"Responses may be cut short or empty. "
            f"Run 'codetrace set-default-ctx <N>' to raise the window."
        )
        return messages, warn

    
    # Error helpers
    

    @staticmethod
    async def _raise_for_status_with_body(resp: httpx.Response) -> None:
        """
        Read the body before raising, so the caller gets the provider's actual
        error message instead of a bare "400 Bad Request".

        httpx's own raise_for_status() throws away the body on error — which is
        exactly how the old code kept swallowing Gemini's detailed explanation of
        what went wrong.
        """
        if resp.status_code >= 400:
            # Safe to aread() here since we haven't started streaming yet.
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
        Hit any OpenAI-compatible /chat/completions endpoint — OpenAI, Groq,
        Gemini, DeepSeek, Mistral, Ollama, OpenRouter, Together, Fireworks,
        SambaNova, SiliconFlow, Hyperbolic.

        With force_text=True we leave the tools out of the payload, forcing the
        model to answer in text instead of firing off more tool calls.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            # Rejoin the split system messages (see _build_messages) — the split
            # exists for Anthropic cache breakpoints, not for the wire.
            "messages": self._merge_system_messages(messages),
            "temperature": self._temperature,
            "max_tokens": self.budget.tier.max_output_tokens,
        }
        if not force_text:
            payload["tools"] = self._sanitized_tool_schemas
            payload["tool_choice"] = "auto"

        if stream:
            payload["stream"] = True
            # This hands back an async context manager, so the caller has to
            # `async with` it.
            return self.client.stream("POST", "/chat/completions", json=payload)

        resp = await self.client.post("/chat/completions", json=payload)
        await self._raise_for_status_with_body(resp)
        return resp.json()

    @staticmethod
    def _extract_thought_signature(tc: dict) -> str:
        """Dig thought_signature out of wherever OpenAI-compat payloads hide it."""
        if not isinstance(tc, dict):
            return ""

        # Right on tc, snake_case / camelCase / kebab-case.
        for key in ("thought_signature", "thoughtSignature", "thought-signature"):
            val = tc.get(key)
            if val and isinstance(val, str):
                return val

        # One level down, under tc["function"].
        fn = tc.get("function")
        if isinstance(fn, dict):
            for key in ("thought_signature", "thoughtSignature", "thought-signature"):
                val = fn.get(key)
                if val and isinstance(val, str):
                    return val
            # ...or tucked inside function.extra_content.google.
            extra = fn.get("extra_content")
            if isinstance(extra, dict):
                google = extra.get("google")
                if isinstance(google, dict):
                    for key in ("thought_signature", "thoughtSignature", "thought-signature"):
                        val = google.get(key)
                        if val and isinstance(val, str):
                            return val

        # Last place to look: tc["extra_content"]["google"].
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
        Parse a non-streamed OpenAI-compatible response into (tool_calls, text).

        Defends against a couple of real-world quirks: a missing or empty
        'choices' (Ollama hands back a bare error dict when the model name is
        wrong) and junk tool-argument JSON (Groq/OpenRouter sometimes send "" or
        None).
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
            # Hang onto thought_signature so the non-streaming loop can echo it
            # back on the follow-up request.
            sig = self._extract_thought_signature(tc)
            tool_calls.append({
                "id": tc.get("id", ""),
                "name": tc.get("function", {}).get("name", ""),
                "args": args,
                "thought_signature": sig,
            })

        # Some Gemini responses put the thought signature on the message itself
        # instead of on a specific tool call. Pin it to the first tool call so it
        # still round-trips (Gemini 3 insists on it).
        if tool_calls and not any(tc["thought_signature"] for tc in tool_calls):
            msg_sig = self._extract_thought_signature(msg)
            if msg_sig:
                tool_calls[0]["thought_signature"] = msg_sig

        return tool_calls, text

    def _build_openai_tool_calls(self, tool_calls: list[dict]) -> list[dict]:
        """
        Rebuild the assistant message's ``tool_calls`` array for the follow-up
        request, echoing back each call's thought signature.

        Thinking models (Gemini 2.5 / 3) hand back a per-call ``thought_signature``
        that has to be returned, or the next request comes back as
        ``400 INVALID_ARGUMENT``. Gemini 3 always thinks — you can't turn it off —
        so skip this round-trip and the tool loop can't even get past the first
        call.

        We put the signature under ``extra_content.google.thought_signature`` on
        the call, since that's the exact spot Gemini's OpenAI-compat endpoint
        reads it from (a flat ``function.thought_signature`` is ignored). It's
        kept per-call — never copied between parallel calls — and only added when
        one actually exists, so non-thinking providers (OpenAI, Groq, Ollama, …)
        never see it.
        """
        entries: list[dict] = []
        for tc in tool_calls:
            entry: dict[str, Any] = {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"]),
                },
            }
            sig = tc.get("thought_signature")
            if sig:
                entry["extra_content"] = {"google": {"thought_signature": sig}}
            entries.append(entry)
        return entries

    async def _stream_openai(self, messages: list[dict], force_text: bool = False):
        """
        Stream an OpenAI-compatible completion, yielding normalized events:
          {"type": "content_delta", "text": "..."}
          {"type": "tool_call", "id": "...", "name": "...", "args": {...}}
        """
        tool_calls_acc: dict[int, dict] = {}
        active_index = 0  # remembered so we can cope with chunks that omit the index
        stream_level_sig = ""  # Gemini sometimes puts the thought signature on the delta

        payload: dict[str, Any] = {
            "model": self.model,
            # Rejoin the split system messages (see _build_messages) — the split
            # exists for Anthropic cache breakpoints, not for the wire.
            "messages": self._merge_system_messages(messages),
            "temperature": self._temperature,
            "max_tokens": self.budget.tier.max_output_tokens,
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

                if chunk.get("usage"):
                    yield {"type": "usage",
                            "usage": chunk["usage"]}
                # Reasoning tokens, under whichever key the provider uses —
                # DeepSeek and OpenRouter send `reasoning_content`, others
                # `reasoning`. Same problem as Ollama's `thinking`: unread, the
                # UI sits silent for the entire reasoning phase.
                reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(reasoning_delta, str) and reasoning_delta:
                    yield {"type": "reasoning_delta", "text": reasoning_delta}

                # A content token.
                if delta.get("content"):
                    yield {"type": "content_delta", "text": delta["content"]}

                # Grab a delta-level thought signature if there is one (Gemini
                # sometimes puts it on the delta instead of on a tool call).
                delta_sig = self._extract_thought_signature(delta)
                if delta_sig:
                    stream_level_sig = delta_sig

                # Tool-call chunks arrive piecemeal and get stitched together.
                for tc_chunk in delta.get("tool_calls", []):
                    # Fall back to the last index if this chunk doesn't carry one.
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
                    
                    # Append to the signature rather than clobbering it.
                    sig = self._extract_thought_signature(tc_chunk)
                    if sig:
                        acc["thought_signature"] += sig

        # If no individual call picked up a signature but one showed up at the
        # delta level, attach it to the first call so it still round-trips.
        if stream_level_sig and tool_calls_acc and not any(
            tc.get("thought_signature") for tc in tool_calls_acc.values()
        ):
            first_idx = sorted(tool_calls_acc)[0]
            tool_calls_acc[first_idx]["thought_signature"] = stream_level_sig

        # Once the stream ends, emit the fully stitched-together tool calls.
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
    
    # Ollama Native API (/api/chat)
    #
    # Ollama's OpenAI-compat endpoint ignores options.num_ctx, so we use its
    # native /api/chat here purely to pass num_ctx and stop server-side prompt
    # truncation. History stays OpenAI-shaped everywhere else in the loop; we
    # translate to native shape only at the moment of sending, and translate the
    # native response back into the same normalized form the OpenAI path emits —
    # so nothing downstream (or any other provider) is affected.

    def _to_ollama_messages(self, messages: list[dict]) -> list[dict]:
        """
        Convert the loop's OpenAI-shaped history into Ollama's native shape:
          - assistant tool calls: function.arguments as an object, not a JSON
            string; drop the OpenAI id/type wrappers.
          - tool results: {role: "tool", content: ...} — native matches them
            positionally, so tool_call_id is dropped.
        Everything else (system/user/plain assistant) passes through.
        """
        out: list[dict] = []
        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                native_calls = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    raw = fn.get("arguments")
                    if isinstance(raw, str):
                        try:
                            args = json.loads(raw or "{}")
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    elif isinstance(raw, dict):
                        args = raw
                    else:
                        args = {}
                    native_calls.append(
                        {"function": {"name": fn.get("name", ""), "arguments": args}}
                    )
                out.append({
                    "role": "assistant",
                    "content": _normalize_text_content(msg.get("content")),
                    "tool_calls": native_calls,
                })
            elif role == "tool":
                out.append({
                    "role": "tool",
                    "content": _normalize_text_content(msg.get("content")),
                })
            else:
                content = msg.get("content")
                out.append({
                    "role": role,
                    "content": content if isinstance(content, str)
                    else _normalize_text_content(content),
                })
        return out

    def _ollama_payload(self, messages: list[dict], stream: bool, force_text: bool) -> dict:
        """Build the native /api/chat request body, with num_ctx pinned."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_ollama_messages(
                self._merge_system_messages(messages)
            ),
            "stream": stream,
            # Keep the model (and its warm KV/prompt cache) resident between turns.
            "keep_alive": self._ollama_keep_alive,
            "options": {
                "temperature": self._temperature,
                "num_ctx": self._ollama_num_ctx,
                "num_predict": self.budget.tier.max_output_tokens,
            },
        }
        # Native /api/chat takes the same OpenAI function-schema tools; it has no
        # tool_choice field, so we just omit tools to force a text answer.
        if not force_text:
            payload["tools"] = self._sanitized_tool_schemas
        return payload

    @staticmethod
    def _ollama_tool_call(tc: dict) -> dict:
        """Normalize one native tool call into the loop's canonical shape."""
        fn = tc.get("function", {})
        raw = fn.get("arguments")
        if isinstance(raw, str):
            try:
                args = json.loads(raw or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
        elif isinstance(raw, dict):
            args = raw
        else:
            args = {}
        return {
            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
            "name": fn.get("name", ""),
            "args": args,
            "thought_signature": "",
        }

    async def _ollama_completion(self, messages: list[dict], force_text: bool = False) -> dict:
        """Non-streaming native /api/chat call. Returns the raw native json with a
        synthesized OpenAI-style 'usage' block so record_iteration reads it."""
        resp = await self.client.post(
            self._ollama_native_url,
            json=self._ollama_payload(messages, stream=False, force_text=force_text),
        )
        await self._raise_for_status_with_body(resp)
        data = resp.json()
        if "message" not in data:
            err = data.get("error") or str(data)
            raise RuntimeError(f"Ollama returned no message: {err}")
        # Native reports prompt_eval_count / eval_count at the top level; map them
        # onto OpenAI field names so the shared usage recorder works unchanged.
        data.setdefault("usage", {
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
        })
        return data

    def _parse_ollama_response(self, data: dict):
        """Parse a non-streamed native response into (tool_calls, text)."""
        msg = data.get("message") or {}
        text = msg.get("content") or ""
        tool_calls = [self._ollama_tool_call(tc) for tc in msg.get("tool_calls") or []]
        return tool_calls, text

    async def _stream_ollama(self, messages: list[dict], force_text: bool = False):
        """
        Stream native /api/chat, yielding the same normalized events as
        _stream_openai. Native streams newline-delimited JSON (not SSE 'data:'
        lines), and hands tool calls back fully formed, so no piecemeal stitching
        is needed.
        """
        async with self.client.stream(
            "POST", self._ollama_native_url,
            json=self._ollama_payload(messages, stream=True, force_text=force_text),
        ) as resp:
            await self._raise_for_status_with_body(resp)
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = chunk.get("message") or {}
                # Thinking models (qwen3, deepseek-r1, gemma with thinking, …)
                # stream their reasoning into a separate `thinking` field. Without
                # this branch the CLI shows nothing at all for the whole reasoning
                # phase — on a partly CPU-offloaded model that reads as a hang.
                if msg.get("thinking"):
                    yield {"type": "reasoning_delta", "text": msg["thinking"]}
                if msg.get("content"):
                    yield {"type": "content_delta", "text": msg["content"]}
                for tc in msg.get("tool_calls") or []:
                    yield {"type": "tool_call", **self._ollama_tool_call(tc)}

                if chunk.get("done"):
                    if chunk.get("prompt_eval_count") is not None or \
                            chunk.get("eval_count") is not None:
                        yield {"type": "usage", "usage": {
                            "prompt_tokens": chunk.get("prompt_eval_count"),
                            "completion_tokens": chunk.get("eval_count"),
                        }}
                    break

    # Ollama adaptive num_ctx control
    #
    # Two independent safety behaviors, both gated so they never resize a running
    # generation (Ollama can't grow/shrink a live KV cache anyway):
    #   - Soft: after a finished turn, if the model spilled to CPU, step num_ctx
    #     down for the NEXT turn (comfort/perf, automatic, informational).
    #   - Hard: if a request errors with an out-of-memory failure, the turn is
    #     stopped, num_ctx is lowered, and the CLI asks whether to continue.

    # Substrings that mark a memory-allocation failure across NVIDIA/AMD/Metal.
    _OOM_MARKERS = (
        "out of memory", "oom", "cudamalloc", "cuda error", "hipmalloc",
        "failed to allocate", "unable to allocate", "cannot allocate",
        "insufficient memory", "not enough memory", "ggml_assert",
        "vk_error_out_of_device_memory", "mtlbuffer",
    )

    def _is_oom_error(self, exc: Exception) -> bool:
        """True if an exception looks like a GPU/host memory-allocation failure."""
        msg = str(exc).lower()
        return any(marker in msg for marker in self._OOM_MARKERS)

    def _lower_ollama_ctx(self) -> tuple[int, int] | None:
        """
        Halve num_ctx (floored at _ollama_min_ctx) and rebuild the budget tier to
        match, so the next request asks Ollama for a smaller KV cache. Returns
        (old, new), or None if already at the floor and can't go lower.
        """
        old = self._ollama_num_ctx
        new = max(self._ollama_min_ctx, int(old * self._ollama_ctx_backoff))
        if new >= old:
            return None
        self._ollama_num_ctx = new
        self._ollama_ctx_source = "auto-lowered (memory pressure)"
        self.budget.tier = _tier_from_context_window(new, None)
        logger.warning("Ollama num_ctx lowered %d -> %d (memory pressure)", old, new)
        return old, new

    def _autotune_ollama_after_turn(self) -> str | None:
        """
        Turn-boundary check (never mid-turn). If the model has spilled to CPU and
        we're still above the floor, step num_ctx down for the next turn and
        return a user-facing note. Otherwise None.
        """
        if not self._is_ollama or self._ollama_num_ctx <= self._ollama_min_ctx:
            return None
        if ollama_model_offloaded(self.model, self.base_url):
            res = self._lower_ollama_ctx()
            if res:
                old, new = res
                return (
                    f"Detected GPU memory spill to CPU — lowered the context "
                    f"window {old:,} → {new:,} tokens for the next turn to keep "
                    f"responses fast."
                )
        return None

    # Anthropic Native API


    def _split_system_messages(self, messages: list[dict]):
        """
        Anthropic wants the system prompt as a top-level 'system' field, not as a
        message in the array — so pull the system messages out here.
        Returns (system_blocks, conversation).

        The blocks come back in order, with a cache_control breakpoint on every
        block except the last. _build_messages emits [stable, volatile], so that
        puts the breakpoint exactly at the end of the stable prompt: Anthropic
        renders tools → system → messages, so one marker there caches the tool
        schemas AND the prompt, while the per-query preflight after it stays
        uncached (it changes, so caching it would only pay write premiums).

        If there's only one system message the whole thing is treated as volatile
        and nothing is marked — better to skip caching than to write a cache
        entry on every request and never read one.
        """
        system_blocks: list[dict] = []
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system_blocks.append({"type": "text", "text": msg["content"]})
            else:
                conversation.append(msg)

        for block in system_blocks[:-1]:
            block["cache_control"] = {"type": "ephemeral"}

        return system_blocks, conversation

    @staticmethod
    def _merge_system_messages(messages: list[dict]) -> list[dict]:
        """
        Collapse consecutive system messages back into one.

        _build_messages deliberately splits the system prompt in two so the
        Anthropic path can put a cache breakpoint between them. Every other
        provider is a plain OpenAI-compatible endpoint, and while most accept
        repeated system messages, some in PROVIDER_REGISTRY don't — so rejoin
        them here and keep the wire format byte-for-byte what it was before the
        split. The separator matches the old concatenation, so the rendered
        prompt is unchanged.
        """
        merged: list[dict] = []
        for msg in messages:
            if (
                msg.get("role") == "system"
                and merged
                and merged[-1].get("role") == "system"
                and isinstance(msg.get("content"), str)
                and isinstance(merged[-1].get("content"), str)
            ):
                merged[-1] = {
                    "role": "system",
                    "content": f"{merged[-1]['content']}\n\n{msg['content']}",
                }
            else:
                merged.append(msg)
        return merged

    async def _anthropic_completion(self, messages: list[dict], stream: bool = False):
        """
        Hit Anthropic's native /v1/messages endpoint. System messages get lifted
        into the top-level 'system' field and the tools use Anthropic's
        'input_schema' shape.
        """
        system_blocks, conversation = self._split_system_messages(messages)

        payload: dict[str, Any] = {
            "model": self.model,
            "system": system_blocks,
            "messages": conversation,
            "tools": self.anthropic_tool_schemas,
            "tool_choice":"auto",
            "max_tokens": self.budget.tier.max_output_tokens,
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
        Parse a non-streamed Anthropic response into (tool_calls, text).
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
        system_blocks, conversation = self._split_system_messages(messages)
        tool_calls_acc: dict[int, dict] = {}

        async with self.client.stream(
            "POST", "/v1/messages",
            json={
                "model": self.model,
                "system": system_blocks,
                "messages": conversation,
                "tools": self.anthropic_tool_schemas,
                "max_tokens": self.budget.tier.max_output_tokens,
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

                # Start of a new content block (text or tool_use).
                if etype == "content_block_start":
                    block = event["content_block"]
                    idx = event["index"]
                    if block["type"] == "tool_use":
                        tool_calls_acc[idx] = {
                            "id": block["id"],
                            "name": block["name"],
                            "input_json": "",
                        }
                elif etype == "message_start":
                    yield {"type": "usage",
                            "usage": event["message"]["usage"]}

                elif etype == "message_delta":
                    yield {"type": "usage",
                            "usage": event.get("usage", {})}

                # A delta inside the current content block.
                elif etype == "content_block_delta":
                    idx = event["index"]
                    delta = event["delta"]
                    if delta["type"] == "text_delta":
                        yield {"type": "content_delta", "text": delta["text"]}
                    elif delta["type"] == "input_json_delta":
                        if idx in tool_calls_acc:
                            tool_calls_acc[idx]["input_json"] += delta["partial_json"]

                # End of the message.
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
        Run the tool-calling loop without streaming. Returns (final_text, TurnUsage).
        """
        total_tool_calls = 0
        had_tool_calls = False
        force_text = False
        # A single TurnUsage collects every iteration of this query — each API
        # call tacks on one (input, output) pair, so total_input works out to 3N+2X+Y.
        turn_usage = self.budget.session.new_turn()

        for _ in range(self.recursion_limit):
            messages = self.budget.enforce_budget(messages)
            iter_messages = list(messages)  # the exact list billed this iter

            try:
                if self.api_style == "anthropic":
                    data = await self._anthropic_completion(messages)
                    tool_calls, text = self._parse_anthropic_response(data)
                elif self._is_ollama:
                    data = await self._ollama_completion(messages, force_text=force_text)
                    tool_calls, text = self._parse_ollama_response(data)
                else:
                    data = await self._openai_completion(messages, force_text=force_text)
                    tool_calls, text = self._parse_openai_response(data)
            except Exception as exc:
                # Non-interactive path (ask()/MCP): on an Ollama OOM, lower
                # num_ctx and retry the same turn; give up once at the floor.
                if self._is_ollama and self._is_oom_error(exc) and self._lower_ollama_ctx():
                    continue
                raise

            # Book this iteration's billed tokens, tool-call iterations included
            # (those used to slip through the cracks). turn_usage is already in
            # the session.
            self.budget.record_iteration(
                turn_usage, iter_messages, data, text or "", self.api_style
            )
            if not tool_calls:
                if not text and had_tool_calls and not force_text:
                    force_text = True
                    continue
                self.budget.finalize_turn(turn_usage, total_tool_calls)
                # Turn finished — retune for the next one if the GPU spilled.
                self._autotune_ollama_after_turn()
                return self.output_normalizer.normalize(text or ""), turn_usage

            force_text = False
            had_tool_calls = True
            total_tool_calls += len(tool_calls)

            remaining = self.budget.remaining_context(messages)
            tool_result_budget = self._tool_result_budget(messages, len(tool_calls))
            if tool_result_budget is None:
                logger.warning(
                    "Context nearly exhausted (%d tokens remaining); "
                    "skipping tool dispatch for this turn to avoid hard limit.",
                    remaining,
                )
                self.budget.finalize_turn(turn_usage, total_tool_calls)
                return text or "", turn_usage

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
                    result = self.budget.truncate_tool_result(
                        result,
                        tc["name"],
                        remaining=tool_result_budget,
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})

            else:
                messages.append({
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": self._build_openai_tool_calls(tool_calls),
                })
                for tc in tool_calls:
                    result = dispatch_tool(tc["name"], tc["args"], self.vector_store, self.graph)
                    result = self.budget.truncate_tool_result(
                        result, tc["name"], remaining=tool_result_budget
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

            # Re-check after appending: if we burned through our safety margin
            # stop the loop — the next iteration would just hit the hard limit.
            if self.budget.remaining_context(messages) < 0:
                logger.warning(
                    "Context window exhausted after tool dispatch; "
                    "ending turn early."
                )
                self.budget.finalize_turn(turn_usage, total_tool_calls)
                return text or "", turn_usage

        fallback = "⚠️ Reached maximum reasoning steps. Please try a more specific query."
        self.budget.finalize_turn(turn_usage, total_tool_calls)
        return fallback, turn_usage

    
    # Agent Loop (Streaming)
    

    async def _stream_agent_loop(self, messages: list[dict]):
        """
        Run the tool-calling loop with streaming. Yields structured event dicts:

        {"type": "thought",   "tool": "search_codebase", "message": "Searching for: auth"}
        {"type": "tool_end",  "tool": "search_codebase"}
        {"type": "reasoning", "content": "Let me check"}   # thinking models only
        {"type": "token",     "content": "The"}
        {"type": "usage",     "turn": TurnUsage, "session": SessionUsage}
        {"type": "done"}

        "reasoning" carries a thinking model's internal reasoning. It is display-only
        — never part of the answer, never written to chat history.
        """
        total_tool_calls = 0
        all_output_parts: list[str] = []

        had_tool_calls = False
        force_text = False

        turn_usage = self.budget.session.new_turn()
        
        for _ in range(self.recursion_limit):
            content_parts: list[str] = []
            tool_calls: list[dict] = []

            messages = self.budget.enforce_budget(messages)
            iter_messages = list(messages)
            iter_usage_data: dict | None = None
            iter_output_text_parts: list[str] = []

            if self.api_style == "anthropic":
                streamer = self._stream_anthropic(messages)
            elif self._is_ollama:
                streamer = self._stream_ollama(messages, force_text=force_text)
            else:
                streamer = self._stream_openai(messages, force_text=force_text)
            try:
                async for event in streamer:
                    if event["type"] == "content_delta":
                        yield {"type": "token", "content": event["text"]}
                        content_parts.append(event["text"])
                        iter_output_text_parts.append(event["text"])
                    elif event["type"] == "reasoning_delta":
                        # Surfaced to the UI so the user can see the model is
                        # working, but deliberately kept out of content_parts:
                        # reasoning is not the answer and must never be sent back
                        # as assistant content or saved to chat history. It does
                        # count toward billed output tokens, so it goes into
                        # iter_output_text_parts.
                        yield {"type": "reasoning", "content": event["text"]}
                        iter_output_text_parts.append(event["text"])
                    elif event["type"] == "usage":
                        iter_usage_data = event["usage"]
                    elif event["type"] == "tool_call":
                        tool_calls.append(event)
            except Exception as exc:
                # Hard memory failure mid-turn: stop the response now, lower
                # num_ctx, and let the CLI ask whether to continue. Only Ollama's
                # local allocation fails this way; re-raise anything else.
                if not (self._is_ollama and self._is_oom_error(exc)):
                    raise
                self.budget.finalize_turn(turn_usage, total_tool_calls)
                lowered = self._lower_ollama_ctx()
                if lowered:
                    old, new = lowered
                    yield {
                        "type": "memory_limit",
                        "old_ctx": old,
                        "new_ctx": new,
                        "message": (
                            f"GPU ran out of memory at a {old:,}-token context "
                            f"window. Lowered to {new:,}."
                        ),
                    }
                else:
                    yield {
                        "type": "error",
                        "message": (
                            f"GPU out of memory even at the minimum context "
                            f"window ({self._ollama_num_ctx:,} tokens). Free GPU "
                            f"memory or switch to a smaller model."
                        ),
                    }
                return

            # Book this iteration's billed tokens. If the stream gave us a usage
            # chunk we trust it (provider-accurate); if not, record_iteration
            # estimates from iter_messages + the output text. The turn already
            # exists from new_turn() above, so we hand it in — nothing new is created.
            self.budget.record_iteration(
                turn_usage,
                iter_messages,
                iter_usage_data,
                "".join(iter_output_text_parts),
                self.api_style,
            )

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

            # ----- context-aware tool dispatch (streaming) -----
            remaining = self.budget.remaining_context(messages)
            tool_result_budget = self._tool_result_budget(messages, len(tool_calls))
            if tool_result_budget is None:
                logger.warning(
                    "Context nearly exhausted (%d tokens remaining); "
                    "skipping tool dispatch for this turn to avoid hard limit.",
                    remaining,
                )
                yield {
                    "type": "notice",
                    "message": (
                        "Context window nearly full - ending turn early to "
                        "protect the response quality."
                    ),
                }
                break

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
                    result = self.budget.truncate_tool_result(
                        result, tc["name"], remaining=tool_result_budget
                    )
                    yield {"type": "tool_end", "tool": tc["name"]}
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})

            else:
                messages.append({
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": self._build_openai_tool_calls(tool_calls),
                })
                for tc in tool_calls:
                    yield {
                        "type": "thought",
                        "tool": tc["name"],
                        "message": _humanize_tool_call(tc["name"], tc["args"]),
                    }
                    result = dispatch_tool(tc["name"], tc["args"], self.vector_store, self.graph)
                    result = self.budget.truncate_tool_result(
                        result, tc["name"], remaining=tool_result_budget
                    )
                    yield {"type": "tool_end", "tool": tc["name"]}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

            # Re-check after appending: if we burned through our safety margin
            # stop the loop — the next iteration would just hit the hard limit.
            if self.budget.remaining_context(messages) < 0:
                logger.warning(
                    "Context window exhausted after tool dispatch; "
                    "ending stream turn early."
                )
                yield {
                    "type": "notice",
                    "message": (
                        "Context window exhausted — ending turn early. "
                        "Start a new query to continue."
                    ),
                }
                break

        self.budget.finalize_turn(turn_usage, total_tool_calls)
        yield {"type": "usage", "turn": turn_usage, "session": self.budget.session}

        # Turn is fully finished — safe to retune. If the model spilled to CPU,
        # step num_ctx down for the NEXT turn (never mid-response).
        note = self._autotune_ollama_after_turn()
        if note:
            yield {"type": "notice", "message": note}

        final_text= "".join(all_output_parts)
        normalized = self.output_normalizer.normalize(final_text)
        yield {"type":"normalized", "content": normalized}
        yield {"type": "done"}


    # Public API
    

    def ask(self, query: str, chat_history: list | None = None) -> str:
        """
        Run the whole pipeline and block until it's done, returning the final
        text. Token usage lands in self.budget.session if you want to look later.
        """
        messages = self._build_messages(query, chat_history)
        messages, budget_warn = self._preflight_budget(messages)
        if budget_warn:
            # Non-streaming path has no event stream, so just log it.
            logger.warning("Preflight budget warning: %s", budget_warn)

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
        Async streaming variant — yields the same structured event dicts the CLI
        consumes: thought → tool_end → token → done.
        """
        messages = self._build_messages(query, chat_history)
        messages, budget_warn = self._preflight_budget(messages)
        if budget_warn:
            yield {"type": "notice", "message": budget_warn}
        async for event in self._stream_agent_loop(messages):
            yield event

    def stream(self, query: str, chat_history: list | None = None) -> Generator[dict, None, None]:
        """
        Sync wrapper over astream() so non-async CLI code can use it.

        A Queue bridges the async generator and the sync caller, so events come
        out in real time as the LLM streams them, instead of piling up and
        flushing all at once when the response finishes.
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
                event_queue.put({"type": "error", "message": describe_exception(exc)})
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