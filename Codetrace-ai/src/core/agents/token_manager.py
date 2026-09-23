"""
TokenBudgetManager: token counting, budgeting, and history compression.

What it does:
  1. Counts tokens via litellm, which routes to each provider's own tokenizer.
  2. Keeps history under a context budget, compressing it when things get tight.
  3. Caps oversized tool results so a single one can't eat the whole window.
  4. Tracks per-turn input/output tokens and hands them to the UI.
  5. Warns before you run into the model's hard limit.

A couple of choices worth knowing:
  - When a provider doesn't report token counts (Ollama, custom endpoints), we
    fall back to an estimate so the counter never goes blank.
  - Compression keeps the last `keep_turns` exchanges word-for-word and only
    summarizes the older ones — recent context stays exact.
"""

from __future__ import annotations

import time
import shutil
import platform
import subprocess
import httpx
import litellm  # used to look up model context windows dynamically
# litellm prints a 'Provider List: …' banner to stdout for every model it
# doesn't recognise. That clutters the CLI and would corrupt an MCP stdio
# stream, so keep it quiet — lookups already fall back gracefully.
litellm.suppress_debug_info = True
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# A "tier" bundles a model's context window with a per-turn soft budget — the
# target size for the history we hand the LLM. Go over that and we start
# compressing. The tier is worked out from the model name; nobody sets it by hand.

@dataclass
class ModelTier:
    name: str
    context_window: int      # hard limit in tokens
    soft_budget: int         # target size for the messages list (input tokens)
    tool_result_cap: int     # max tokens per single tool result
    max_output_tokens: int   # max output tokens the model is allowed to generate


# Cache of resolved tiers, keyed by (model name, ollama base url) so the same
# model name pointed at different endpoints doesn't collide (all sizes in tokens).
_tier_cache: dict[
    tuple[str, str | None, int | None, int | None, int | None],
    tuple[ModelTier, float],
] = {}
_CACHE_TTL = 86_400  # 24h


def get_ollama_context(model_name: str, base_url: str = "http://localhost:11434") -> int | None:
    # The config stores the OpenAI-compatible URL (…/v1), but /api/show is a
    # *native* Ollama endpoint that does NOT live under /v1. Strip the shim
    # suffix (and any trailing slash) so we hit the native API, not a 404.
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        resp = httpx.post(f"{root}/api/show", json={"name": model_name}, timeout=3)
        resp.raise_for_status()
        data = resp.json()
        info = data.get("model_info", {})
        for key, val in info.items():
            if key.endswith("context_length"):
                return int(val)
    except Exception:
        return None
    return None


def get_static_model_info(model_name: str) -> dict | None:
    try:
        info = litellm.get_model_info(model_name)
        return {
            "context_window": info.get("max_input_tokens") or info.get("max_tokens"),
            "max_output_tokens": info.get("max_output_tokens"),
        }
    except Exception:
        return None


def _tier_from_context_window(context_window: int, max_output_tokens: int | None) -> ModelTier:
    if context_window >= 100_000:
        name, soft_ratio, cap = "tier3", 0.60, 8_000
    elif context_window >= 32_000:
        name, soft_ratio, cap = "tier2", 0.65, 4_000
    else:
        name, soft_ratio, cap = "tier1", 0.55, 2_000
    return ModelTier(
        name=name,
        context_window=context_window,
        soft_budget=int(context_window * soft_ratio),
        tool_result_cap=min(cap, context_window // 4),
        max_output_tokens=max_output_tokens or min(8192, context_window // 8),
    )


def resolve_tier(
    model_name: str,
    ollama_base_url: str | None = None,
    default_context_window: int | None = None,
    context_window_override: int | None = None,
    max_output_tokens_override: int | None = None,
) -> ModelTier:
    if ollama_base_url:
        context_window_override = None
        max_output_tokens_override = None
    cache_key = (
        model_name,
        ollama_base_url,
        default_context_window,
        context_window_override,
        max_output_tokens_override,
    )

    cached = _tier_cache.get(cache_key)
    if cached and (time.time() - cached[1]) < _CACHE_TTL:
        return cached[0]

    context_window = context_window_override
    max_output = max_output_tokens_override

    # Explicit config is authoritative. Only use LiteLLM lookup when no
    # context window was supplied.
    if context_window is None and ollama_base_url:
        context_window = get_ollama_context(model_name, ollama_base_url)

    if context_window is None:
        static = get_static_model_info(model_name)
        if static:
            context_window = static["context_window"]
            max_output = max_output_tokens_override or static["max_output_tokens"]

    if context_window is None:
        if ollama_base_url is None:
            context_window = default_context_window or 32_768
            logger.info(
                "Unknown cloud model '%s' — using configured default window %d",
                model_name,
                context_window,
            )
        else:
            # Even on the Ollama path, honour a user-supplied default so that
            # cloud/custom models served through an Ollama-compatible endpoint
            # (e.g. z-ai, LM Studio, custom OpenAI-compat servers) don't get
            # clamped to the local-GPU-safe 8K floor.  The 8K fallback is kept
            # only when no default was configured, because an unconstrained
            # large window would OOM a real local GPU.
            if default_context_window:
                context_window = default_context_window
                logger.warning(
                    "Unknown Ollama-endpoint model '%s' — using configured "
                    "default window %d (set via 'codetrace set-default-ctx')",
                    model_name,
                    context_window,
                )
            else:
                logger.warning(
                    "Unknown model '%s' — using conservative 8K fallback. "
                    "Run 'codetrace set-default-ctx <N>' to raise this.",
                    model_name,
                )
                context_window = 8_192

    tier = _tier_from_context_window(context_window, max_output)
    _tier_cache[cache_key] = (tier, time.time())
    return tier


# GPU detection and hardware-safe num_ctx (Ollama)
#
# The KV cache Ollama allocates grows linearly with num_ctx, so asking for a
# window the GPU can't hold either fails outright or spills to system RAM and
# crawls. We detect device memory and pick a conservative num_ctx that leaves
# ample room for model weights + activations, then clamp to the model's own
# trained window. Detection is best-effort: anything we can't read confidently
# falls back to a small, universally-safe default rather than guessing high.

def detect_gpu_memory_mb() -> tuple[str, int] | None:
    """
    Best-effort probe for (backend, total device memory in MB). Returns None
    when no supported accelerator is confidently detected — callers must treat
    that as "assume low-end" and stay conservative, never optimistic.
    """
    # NVIDIA — nvidia-smi ships with the driver on Windows and Linux.
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=4,
            )
            if out.returncode == 0:
                vals = [int(x) for x in out.stdout.replace(",", " ").split() if x.strip().isdigit()]
                if vals:
                    return ("nvidia", max(vals))  # largest GPU if several
        except Exception:
            pass

    # Apple Silicon — Metal draws from unified memory, so total RAM is the pool.
    if platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64"):
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=4
            )
            s = out.stdout.strip()
            if out.returncode == 0 and s.isdigit():
                return ("metal", int(s) // (1024 * 1024))
        except Exception:
            pass

    # AMD / Intel / unknown: no reliable, low-risk probe here. Returning None
    # keeps us on the conservative default (over-reading VRAM risks the exact
    # OOM we're avoiding); users on those cards can raise it via `set-ctx`.
    return None


def ollama_model_offloaded(model_name: str, base_url: str) -> bool | None:
    """
    Detect whether the loaded model spilled out of VRAM onto the CPU — the
    tell-tale sign that num_ctx (KV cache) is too big for the hardware.

    Returns True if partially on CPU, False if fully resident in VRAM, None if
    it can't be determined. Reads Ollama's native /api/ps, which reports each
    loaded model's total `size` and the portion in `size_vram`.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        resp = httpx.get(f"{root}/api/ps", timeout=3)
        resp.raise_for_status()
        for m in resp.json().get("models", []):
            name = m.get("name") or m.get("model") or ""
            if name == model_name or model_name in name or name in model_name:
                size = m.get("size") or 0
                size_vram = m.get("size_vram") or 0
                if size > 0:
                    # Small tolerance so rounding noise doesn't read as a spill.
                    return size_vram < int(size * 0.98)
        return None
    except Exception:
        return None


def _ctx_ceiling_for_vram(vram_mb: int) -> int:
    """Map device memory to a safe num_ctx ceiling. Each step is chosen so the
    KV cache stays a fraction of memory, leaving room for weights + overhead."""
    # The steps are deliberately one notch below what the arithmetic allows.
    # The KV cache has to share the card with the model weights *and* the
    # runtime's activations, so sizing it to the theoretical maximum means the
    # model spills to CPU on the first real turn — which is exactly what a
    # 6 GB card did at the old 16k ceiling, then spent every turn halving its
    # way back down. Starting low and staying there is faster than converging.
    if vram_mb < 8_000:
        return 8_192
    if vram_mb < 12_000:
        return 16_384
    if vram_mb < 20_000:
        return 32_768
    if vram_mb < 32_000:
        return 65_536
    return 131_072


def recommended_num_ctx(model_window: int) -> tuple[int, str]:
    """
    Pick a hardware-safe num_ctx, clamped to the model's own trained window.
    Returns (num_ctx, human-readable source) for logging/UX.
    """
    detected = detect_gpu_memory_mb()
    if detected is None:
        ceiling = 8_192
        detail = "no supported GPU detected — conservative default"
    else:
        backend, vram_mb = detected
        ceiling = _ctx_ceiling_for_vram(vram_mb)
        detail = f"{backend} ~{vram_mb:,} MB"

    # Never exceed what the model was trained for, and never drop absurdly low.
    num_ctx = max(2_048, min(model_window, ceiling))
    return num_ctx, detail


# Token counting

def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """
    Count tokens in a string. litellm picks the right tokenizer for the model.
    """
    if not text:
        return 0
    try:
        return litellm.token_counter(model=model_name, text=text)
    except Exception:
        # If that fails, rough it out at ~1.3 tokens per word.
        return max(1, int(len(text.split()) * 1.3))


def _msg_has_tool_calls(msg: dict) -> bool:
    """
    True if this assistant message issues tool calls — either OpenAI's
    `tool_calls` field or Anthropic's `tool_use` content blocks. Compression must
    never split such a message from the tool-result messages that follow it;
    orphan them and the provider throws a 400 over the unmatched tool_call_id.
    """
    if msg.get("role") != "assistant":
        return False
    if msg.get("tool_calls"):
        return True
    content = msg.get("content")
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)
    return False


def count_messages_tokens(messages: list[dict], model_name: str = "gpt-4o") -> int:
    """Estimate total tokens for a messages list using litellm."""
    try:
        return litellm.token_counter(model=model_name, messages=messages)
    except Exception:
        total = 0
        for msg in messages:
            total += 4
            content = msg.get("content", "")
            if isinstance(content, str):
                total += count_tokens(content, model_name)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += count_tokens(block.get("text", "") or block.get("content", ""), model_name)
                    elif isinstance(block, str):
                        total += count_tokens(block, model_name)
        return total


# Per-turn usage tracking

@dataclass
class TurnUsage:
    """
    Token counts for a single agent turn (one user query).

    iter_*_tokens has one entry per API call in the agentic loop, so a query that
    hits the API three times gives lists of length 3 like [N, N+X, N+X+Y].
    total_input/total_output are the cumulative billed counts — the sum across
    those iterations, e.g. 3N+2X+Y for input.

    total_input is the FULL prompt size, cached or not. total_cache_read is the
    portion of it that was served from a provider's prompt cache (billed at a
    fraction of the normal rate), and total_cache_write the portion written to
    cache. Both are subsets of total_input, never additions to it — so
    total_input + total_output stays the true size of the exchange whether or not
    caching is engaged, and turning caching on shows up as a rising "cached"
    figure rather than a falling "in" figure.
    """
    iter_input_tokens: list[int] = field(default_factory=list)
    iter_output_tokens: list[int] = field(default_factory=list)
    # Cache accounting, for providers that bill cached input separately.
    # iter_input_tokens is the FULL prompt size, so cache_read is a subset of it,
    # not an addition — see record_iteration for how the two provider shapes are
    # normalized onto that convention.
    iter_cache_read_tokens: list[int] = field(default_factory=list)
    iter_cache_write_tokens: list[int] = field(default_factory=list)
    tool_calls: int = 0

    def add_iteration(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        """Append one iteration's billed input/output to this turn."""
        self.iter_input_tokens.append(int(input_tokens or 0))
        self.iter_output_tokens.append(int(output_tokens or 0))
        self.iter_cache_read_tokens.append(int(cache_read_tokens or 0))
        self.iter_cache_write_tokens.append(int(cache_write_tokens or 0))

    @property
    def total_input(self) -> int:
        return sum(self.iter_input_tokens)

    @property
    def total_output(self) -> int:
        return sum(self.iter_output_tokens)

    @property
    def last_input(self) -> int:
        return self.iter_input_tokens[-1] if self.iter_input_tokens else 0

    @property
    def last_output(self) -> int:
        return self.iter_output_tokens[-1] if self.iter_output_tokens else 0

    @property
    def total_cache_read(self) -> int:
        """Prompt tokens served from cache (a subset of total_input)."""
        return sum(self.iter_cache_read_tokens)

    @property
    def total_cache_write(self) -> int:
        """Prompt tokens written to cache (also a subset of total_input)."""
        return sum(self.iter_cache_write_tokens)

    @property
    def total_tokens(self) -> int:
        return self.total_input + self.total_output

    def format(self) -> str:
        """Return a compact display string for the CLI token counter."""
        cached = self.total_cache_read
        return (
            f"[dim]Tokens: [cyan]{self.total_input:,}[/cyan] in"
            + (f" ([blue]{cached:,}[/blue] cached)" if cached else "")
            + f" · [green]{self.total_output:,}[/green] out · "
            f"[yellow]{self.total_tokens:,}[/yellow] total"
            + (f" · [magenta]{self.tool_calls}[/magenta] tool call(s)" if self.tool_calls else "")
            + "[/dim]"
        )


@dataclass
class SessionUsage:
    """Aggregate token usage for the current chat session."""
    turns: List[TurnUsage] = field(default_factory=list)

    def record(self, turn: TurnUsage) -> None:
        self.turns.append(turn)

    def new_turn(self) -> TurnUsage:
        """Make a fresh TurnUsage for a new user query and record it right away.

        The agent loop then owns this turn for its whole lifetime and calls
        turn.add_iteration(...) once per API call. Since it's already appended
        here, finalize_turn() must not record it a second time.
        """
        turn = TurnUsage()
        self.turns.append(turn)
        return turn

    @property
    def total_input(self) -> int:
        return sum(t.total_input for t in self.turns)

    @property
    def total_output(self) -> int:
        return sum(t.total_output for t in self.turns)

    @property
    def total_cache_read(self) -> int:
        return sum(t.total_cache_read for t in self.turns)

    @property
    def total_tokens(self) -> int:
        return self.total_input + self.total_output

    def format_session(self) -> str:
        cached = self.total_cache_read
        return (
            f"[dim]Session total: [cyan]{self.total_input:,}[/cyan] in"
            + (f" ([blue]{cached:,}[/blue] cached)" if cached else "")
            + f" · [green]{self.total_output:,}[/green] out · "
            f"[yellow]{self.total_tokens:,}[/yellow] total[/dim]"
        )


# TokenBudgetManager: Context budget enforcement

class TokenBudgetManager:
    """
    Keeps the agentic loop inside its token budget.

    It tracks per-turn input/output tokens (from the API response, or an estimate
    when there isn't one), trims tool results that are too big before they hit the
    message list, compresses history as it nears the context window, and reports
    both per-turn and session totals to the CLI.

    Usage:
        budget = TokenBudgetManager(model_name="gpt-4o-mini")
        # Before sending to LLM:
        messages = budget.enforce_budget(messages)
        # After LLM responds:
        usage = budget.record_turn(messages, response_data, output_text)
        # In the CLI:
        console.print(usage.format())
    """

    def __init__(
        self,
        model_name: str,
        ollama_base_url: str | None = None,
        default_context_window: int | None = None,
        context_window_override: int | None = None,
        max_output_tokens_override: int | None = None,
    ):
        self.model_name = model_name
        self._count_model = model_name
        if ollama_base_url and not model_name.startswith(("ollama/", "ollama_chat/")):
            self._count_model = f"ollama/{model_name}"
        self.tier = resolve_tier(
            model_name=model_name,
            ollama_base_url=ollama_base_url,
            default_context_window=default_context_window,
            context_window_override=context_window_override,
            max_output_tokens_override=max_output_tokens_override,
        )
        self.session = SessionUsage()
        logger.debug(
            "TokenBudgetManager: model=%s tier=%s window=%d soft_budget=%d",
            model_name, self.tier.name, self.tier.context_window, self.tier.soft_budget
        )

    # Tool result truncation

    def remaining_context(self, messages: list[dict]) -> int:
        """
        How many tokens are left before the model's hard limit, with the safety
        margin (max_output_tokens) already subtracted.

        Returns the headroom in tokens.  Negative means we are already past the
        safe limit and the caller should stop appending content.

        This is the single source of truth the agent loop queries before
        dispatching tools so it can decide whether to skip or early-abort rather
        than rely on the post-hoc truncation in ``truncate_tool_result`` alone.
        """
        tokens = count_messages_tokens(messages, self._count_model)
        return self.tier.context_window - tokens - self.tier.max_output_tokens

    def truncate_tool_result(
        self, result: str, tool_name: str = "", remaining: int | None = None
    ) -> str:
        """
        Cap tool output at ``tier.tool_result_cap`` tokens, leaving a visible
        note when it happens so the agent knows something was cut.

        If *remaining* is given it overrides the effective cap — the result is
        truncated at ``min(tool_result_cap, remaining)`` so it never eats more
        room than the context window actually has left.  Callers should pass
        ``self.remaining_context(messages)`` for this parameter.

        Without this, one big result — say, reading a 5000-line file — could
        swallow the whole context window and leave nothing for reasoning.
        """
        cap = self.tier.tool_result_cap
        if remaining is not None and remaining < cap:
            cap = max(1, remaining)

        tokens = count_tokens(result, self._count_model)
        if tokens <= cap:
            return result

        # Cut by character count — rough, but it does the job. Figure ~4 chars
        # per token for code.
        char_limit = cap * 4
        truncated = result[:char_limit]
        removed = tokens - cap
        notice = (
            f"\n\n[TRUNCATED: {removed:,} tokens removed to fit context budget. "
            f"Tool: {tool_name or 'unknown'}. "
            f"Ask to 'read_file' specific sections if you need more.]"
        )
        logger.debug("Truncated tool result '%s': %d → %d tokens", tool_name, tokens, cap)
        return truncated + notice

    # History compression

    def compress_history(
        self,
        messages: list[dict],
        keep_system: bool = True,
        keep_turns: int = 6,
    ) -> list[dict]:
        """
        Shrink the message history until it fits inside soft_budget.

        The approach: always keep the system message (compressed on its own if
        need be), keep the last `keep_turns` user/assistant exchanges word-for-word,
        and collapse everything older into a short summary line. If even that's
        still over the hard limit, drop `keep_turns` one at a time until it fits,
        so we don't hit a cutoff error.

        Recent context stays exact, older context gets summarized — which is the
        trade-off that actually works for long coding sessions.
        """
        current_tokens = count_messages_tokens(messages, self._count_model)
        if current_tokens <= self.tier.soft_budget:
            return messages  # already within budget, nothing to do

        # Internal compression logic - no user-facing logs needed

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system  = [m for m in messages if m.get("role") != "system"]

        # Group the messages into whole turns.
        #
        # A turn only closes on a *final* assistant message — one with no tool
        # calls. An assistant message that does make tool calls stays glued to the
        # tool-result messages after it, so a full
        # user → assistant(tool_calls) → tool… → assistant(final) run is treated as
        # one indivisible unit. That's what stops compression from summarizing away
        # the assistant(tool_calls) message while leaving its tool results orphaned
        # (which would trip a provider 400 over the unmatched tool_call_id).
        turn_pairs: list[list[dict]] = []
        current_pair: list[dict] = []
        for msg in non_system:
            current_pair.append(msg)
            if msg.get("role") == "assistant" and not _msg_has_tool_calls(msg):
                turn_pairs.append(current_pair)
                current_pair = []
        if current_pair:
            # A leftover run with no closing assistant message — usually the
            # in-flight query whose tool results are the last messages. It's the
            # most recent thing there is, so it never gets summarized.
            turn_pairs.append(current_pair)

        # Leave headroom for the model's own output.
        safe_hard_limit = self.tier.context_window - self.tier.max_output_tokens

        compressed = messages
        new_tokens = current_tokens

        while keep_turns >= 1:
            if len(turn_pairs) <= keep_turns:
                old_turns = []
                kept_turns = turn_pairs
            else:
                old_turns = turn_pairs[:-keep_turns]
                kept_turns = turn_pairs[-keep_turns:]

            if not old_turns:
                compressed = messages
            else:
                # Roll the older turns up into a few short lines.
                summary_lines = []
                for pair in old_turns:
                    for msg in pair:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        text = content if isinstance(content, str) else "[tool interaction]"
                        # One line each, first 120 chars.
                        short = text[:120].replace("\n", " ")
                        if len(text) > 120:
                            short += "…"
                        summary_lines.append(f"[{role}]: {short}")

                summary_msg = {
                    "role": "user",
                    "content": (
                        "[Earlier conversation summary — detail omitted to save context]\n"
                        + "\n".join(summary_lines)
                    ),
                }
                summary_reply = {
                    "role": "assistant",
                    "content": "[Acknowledged — continuing from summary above]",
                }

                compressed = system_msgs + [summary_msg, summary_reply]
                for pair in kept_turns:
                    compressed.extend(pair)

            new_tokens = count_messages_tokens(compressed, self._count_model)

            # Under the soft budget? Done.
            if new_tokens <= self.tier.soft_budget:
                break

            # Still over the hard limit — have to keep fewer turns and retry.
            if new_tokens > safe_hard_limit:
                if not old_turns and keep_turns <= 1:
                    # Nothing left to summarize: it's all one turn (a long tool
                    # run, typically). Dropping turns can't help — trim contents.
                    break
                keep_turns -= 1
                continue

            # Between the two: over soft budget but under the hard limit. Accept it
            # rather than throw away more recent context than we need to.
            break

        # Turn-dropping alone can leave us over budget — most often on a single
        # long turn, where there are no older turns to summarize at all. Left
        # there, the prompt keeps growing until the provider silently truncates
        # it (Ollama drops the *front*, taking the system prompt with it) and the
        # model answers with a stub or nothing. So shrink the biggest message
        # bodies until it actually fits.
        if new_tokens > safe_hard_limit:
            compressed = self._trim_largest_messages(compressed, safe_hard_limit)
            new_tokens = count_messages_tokens(compressed, self._count_model)

        # logger.info("History compressed: %d → %d tokens", current_tokens, new_tokens)
        return compressed

    def _trim_largest_messages(self, messages: list[dict], limit: int) -> list[dict]:
        """
        Last-resort shrink: repeatedly truncate the largest non-system string
        message until the whole list fits in `limit` tokens.

        System messages are left alone (they carry the tool contract and the
        agent's instructions — losing them is worse than losing any transcript),
        and so are non-string contents, which are structured tool_use/tool_result
        blocks whose shape providers validate.
        """
        trimmed = [dict(m) for m in messages]
        # Enough passes to halve the biggest offenders several times over.
        for _ in range(40):
            total = count_messages_tokens(trimmed, self._count_model)
            if total <= limit:
                break

            candidates = [
                (len(m.get("content") or ""), i)
                for i, m in enumerate(trimmed)
                if m.get("role") != "system" and isinstance(m.get("content"), str)
            ]
            if not candidates:
                break
            size, idx = max(candidates)
            if size < 400:
                break  # nothing big enough left to be worth cutting

            # Cut the message roughly in proportion to the overshoot, but never
            # by less than a third — otherwise we loop without making progress.
            overshoot_chars = (total - limit) * 4
            keep = max(200, min(size - overshoot_chars, int(size * 0.66)))
            trimmed[idx]["content"] = (
                trimmed[idx]["content"][:keep]
                + "\n\n[TRUNCATED to fit the context window.]"
            )

        final = count_messages_tokens(trimmed, self._count_model)
        if final > limit:
            logger.warning(
                "Could not trim history under the hard limit: %d > %d tokens.",
                final, limit,
            )
        return trimmed

    # Budget enforcement (called before every LLM request)

    def enforce_budget(self, messages: list[dict]) -> list[dict]:
        """
        Pre-flight before every request: compress the history, then check how
        much room is left. Returns the (possibly compressed) list, ready to send.
        """
        compressed = self.compress_history(messages)
        tokens = count_messages_tokens(compressed, self._count_model)

        remaining = self.tier.context_window - tokens
        if remaining < 2_048:
            # Getting tight — warn but send it anyway and let the LLM decide.
            logger.warning(
                "Context nearly full: %d/%d tokens used, %d remaining.",
                tokens, self.tier.context_window, remaining
            )
        return compressed

    # Usage recording

    # The cumulative-tracking API.
    # The agent loop keeps one TurnUsage per user query (from
    # self.session.new_turn()) and calls record_iteration(...) once per API call.
    # That gives iter_*_tokens = [N, N+X, N+X+Y] and total_input = 3N+2X+Y —
    # the real billed cumulative count.

    def record_iteration(
        self,
        turn: TurnUsage,
        messages_sent_this_iter: list[dict],
        response_data: dict | None,
        output_text_this_iter: str,
        api_style: str = "",
    ) -> None:
        """
        Append ONE iteration's billed input/output to an existing TurnUsage.

        `response_data` may be:
          - a full non-streaming response dict (has "usage"),
          - a streaming usage chunk dict ({"prompt_tokens":...} or
            {"input_tokens":...}),
          - None for providers that emit no usage at all (Ollama streaming) —
            in which case we fall back to tiktoken estimation.

        It does NOT call session.record() — the turn was already recorded by
        session.new_turn() back at the start of the loop.
        """
        usage_data: dict = {}
        if isinstance(response_data, dict):
            usage_data = response_data.get("usage") or response_data

        # OpenAI's field names first, then Anthropic's. Coalesce on None (not
        # falsy) so a genuine 0 from the provider survives instead of being
        # treated as "unreported" and re-estimated below.
        input_tokens = usage_data.get("prompt_tokens")
        if input_tokens is None:
            input_tokens = usage_data.get("input_tokens")
        output_tokens = usage_data.get("completion_tokens")
        if output_tokens is None:
            output_tokens = usage_data.get("output_tokens")

        # Cached-prompt accounting. The two provider families disagree on whether
        # cached tokens are already inside the input count, so normalize both onto
        # "input_tokens is the whole prompt, cache_read is a subset of it":
        #
        #   Anthropic  input_tokens EXCLUDES cached — the full prompt is
        #              input + cache_creation + cache_read, so add them on.
        #              Left alone, the counter *drops* once caching engages and
        #              reads as a bug rather than a saving.
        #   OpenAI     prompt_tokens ALREADY INCLUDES cached_tokens (reported
        #              under prompt_tokens_details), so adding would double-count.
        cache_read = usage_data.get("cache_read_input_tokens")
        cache_write = usage_data.get("cache_creation_input_tokens")
        if cache_read is not None or cache_write is not None:
            cache_read = int(cache_read or 0)
            cache_write = int(cache_write or 0)
            if input_tokens is not None:
                input_tokens = int(input_tokens) + cache_read + cache_write
        else:
            details = usage_data.get("prompt_tokens_details") or {}
            cache_read = int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0
            cache_write = 0

        # Nothing reported? Estimate from the messages and the output text.
        # Use `is None` so a legitimately-reported 0 isn't re-estimated.
        if input_tokens is None:
            input_tokens = count_messages_tokens(messages_sent_this_iter, self._count_model)
        if output_tokens is None:
            output_tokens = count_tokens(output_text_this_iter, self._count_model)

        turn.add_iteration(input_tokens, output_tokens, cache_read, cache_write)

    # Legacy single-shot APIs, kept for backward compatibility.
    # They still work, but each just makes a one-iteration TurnUsage. The agent
    # loops themselves have moved to the new_turn()/record_iteration() flow above.

    def record_turn_from_response(
        self,
        messages_sent: list[dict],
        response_data: dict,
        output_text: str,
        tool_call_count: int = 0,
    ) -> TurnUsage:
        """
        Legacy: extract/estimate token counts from a single API response and
        record them as a one-iteration turn.
        """
        turn = self.session.new_turn()
        turn.tool_calls = tool_call_count
        self.record_iteration(turn, messages_sent, response_data, output_text)
        return turn

    def estimate_streaming_turn(
        self,
        messages_sent: list[dict],
        output_text: str,
        tool_call_count: int = 0,
    ) -> TurnUsage:
        """
        Legacy: estimate token usage for a single streaming turn where the
        response body isn't available. Uses tiktoken counts.
        """
        turn = self.session.new_turn()
        turn.tool_calls = tool_call_count
        self.record_iteration(turn, messages_sent, None, output_text)
        return turn

    def finalize_turn(self, turn: TurnUsage, tool_call_count: int = 0) -> TurnUsage:
        """
        Stamp the final tool-call count onto a turn that was created via
        session.new_turn() and accumulated across iterations.  Does NOT
        re-record (the turn is already in self.session.turns).
        """
        turn.tool_calls = tool_call_count
        return turn
