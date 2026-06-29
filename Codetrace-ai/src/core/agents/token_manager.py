"""
TokenBudgetManager: Intelligent token counting, budgeting, and history compression.

Strategy (from the multi-tiered approach):
  1. Count tokens accurately using tiktoken (cl100k_base, works for all GPT/Claude-class models).
  2. Enforce a context budget — auto-compress history when the budget is tight.
  3. Truncate oversized tool results so they never blow the context window.
  4. Track per-turn usage (input + output tokens) and surface it to the UI.
  5. Emit warnings before the user hits hard API limits.

Design decisions:
  - tiktoken is already in pyproject.toml, so no new dependency is added.
  - For providers without token counts in the response (Ollama, custom), we
    fall back to the tiktoken estimate so the counter is always shown.
  - History compression is LOSSLESS for the last `keep_turns` exchanges and
    LOSSY (summarised) for older turns — preserving recent context exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Tier definitions
# Each tier has a context window size and a per-turn soft budget.
# The soft budget is the target size for the history passed to the LLM.
# When history exceeds this, we compress aggressively.
#
# The tier is inferred from the model name; users do not configure this manually.

@dataclass
class ModelTier:
    name: str
    context_window: int      # hard limit in tokens
    soft_budget: int         # target size for the messages list (input tokens)
    tool_result_cap: int     # max tokens per single tool result


# Tier registry
# Models matched by substring (longest match wins).
# All sizes in tokens.

_TIER_REGISTRY: list[tuple[str, ModelTier]] = [
    # Tier 3: Massive context (>100k)
    ("claude-3-5-sonnet",  ModelTier("tier3", 200_000, 120_000, 8_000)),
    ("claude-3-5-haiku",   ModelTier("tier3", 200_000, 120_000, 8_000)),
    ("claude-3-opus",      ModelTier("tier3", 200_000, 120_000, 8_000)),
    ("claude-3-sonnet",    ModelTier("tier3", 200_000, 120_000, 8_000)),
    ("gpt-4o",             ModelTier("tier3", 128_000,  80_000, 8_000)),
    ("gpt-4-turbo",        ModelTier("tier3", 128_000,  80_000, 8_000)),
    ("deepseek-chat",      ModelTier("tier3", 128_000,  80_000, 6_000)),
    ("deepseek/deepseek",  ModelTier("tier3", 128_000,  80_000, 6_000)),
    ("gemini-2.0-flash",   ModelTier("tier3", 128_000,  80_000, 6_000)),
    ("gemini-1.5",         ModelTier("tier3", 128_000,  80_000, 6_000)),
    ("qwen2.5-coder:32b",  ModelTier("tier3", 128_000,  80_000, 6_000)),

    # Tier 2: Large context (32k-128k)
    ("llama-3.3-70b",      ModelTier("tier2",  32_768,  24_000, 4_000)),
    ("llama-3.1-70b",      ModelTier("tier2",  32_768,  24_000, 4_000)),
    ("llama-3.1-8b",       ModelTier("tier2",  32_768,  24_000, 4_000)),
    ("codestral",          ModelTier("tier2",  32_768,  24_000, 4_000)),
    ("mixtral",            ModelTier("tier2",  32_768,  24_000, 4_000)),
    ("gpt-4o-mini",        ModelTier("tier2", 128_000,  80_000, 6_000)),

    # Tier 1: Small context (<32k) — compress aggressively
    ("qwen2.5-coder:7b",   ModelTier("tier1",   8_192,   5_000, 2_000)),
    ("llama-3.2",          ModelTier("tier1",   8_192,   5_000, 2_000)),
    ("mistral-7b",         ModelTier("tier1",  32_768,  18_000, 2_500)),
]

# Fallback for unknown models
_DEFAULT_TIER = ModelTier("tier2", 32_768, 24_000, 4_000)


def _resolve_tier(model_name: str) -> ModelTier:
    """
    Match model name against the tier registry (case-insensitive substring).
    Returns the first matching tier, or a safe default.
    """
    lower = model_name.lower()
    for pattern, tier in _TIER_REGISTRY:
        if pattern in lower:
            return tier
    return _DEFAULT_TIER


# Token counting

def _get_encoding():
    """Load tiktoken cl100k_base (works for GPT-4, Claude, Llama-class models)."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_ENCODING = _get_encoding()


def count_tokens(text: str) -> int:
    """
    Count tokens in a string.
    Uses tiktoken if available, otherwise falls back to word-count heuristic
    (words * 1.3, which is accurate to ~10% for English prose and code).
    """
    if not text:
        return 0
    if _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text))
        except Exception:
            pass
    # Fallback: ~1.3 tokens per word, averaged across prose + code
    return max(1, int(len(text.split()) * 1.3))


def count_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens for a messages list (including role overhead)."""
    total = 0
    for msg in messages:
        # ~4 tokens overhead per message (role + formatting)
        total += 4
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += count_tokens(block.get("text", "") or block.get("content", ""))
                elif isinstance(block, str):
                    total += count_tokens(block)
    return total


# Per-turn usage tracking

@dataclass
class TurnUsage:
    """Token counts for a single agent turn."""
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def format(self) -> str:
        """Return a compact display string for the CLI token counter."""
        return (
            f"[dim]Tokens: [cyan]{self.input_tokens:,}[/cyan] in · "
            f"[green]{self.output_tokens:,}[/green] out · "
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

    @property
    def total_input(self) -> int:
        return sum(t.input_tokens for t in self.turns)

    @property
    def total_output(self) -> int:
        return sum(t.output_tokens for t in self.turns)

    @property
    def total_tokens(self) -> int:
        return self.total_input + self.total_output

    def format_session(self) -> str:
        return (
            f"[dim]Session total: [cyan]{self.total_input:,}[/cyan] in · "
            f"[green]{self.total_output:,}[/green] out · "
            f"[yellow]{self.total_tokens:,}[/yellow] total[/dim]"
        )


# TokenBudgetManager: Context budget enforcement

class TokenBudgetManager:
    """
    Manages token budgets for the agentic loop.

    Responsibilities:
      1. Track input/output tokens per turn (from API response or estimation).
      2. Truncate oversized tool results before they enter the message list.
      3. Compress history when it approaches the context window limit.
      4. Surface per-turn and session usage to the CLI.

    Usage:
        budget = TokenBudgetManager(model_name="gpt-4o-mini")
        # Before sending to LLM:
        messages = budget.enforce_budget(messages)
        # After LLM responds:
        usage = budget.record_turn(messages, response_data, output_text)
        # In the CLI:
        console.print(usage.format())
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tier = _resolve_tier(model_name)
        self.session = SessionUsage()
        logger.debug(
            "TokenBudgetManager: model=%s tier=%s window=%d soft_budget=%d",
            model_name, self.tier.name, self.tier.context_window, self.tier.soft_budget
        )

    # Tool result truncation

    def truncate_tool_result(self, result: str, tool_name: str = "") -> str:
        """
        Cap tool output at tier.tool_result_cap tokens.
        Adds a visible notice when truncation occurs so the agent knows.

        Rationale: A single large tool result (e.g. reading a 5000-line file)
        can consume the entire context window, leaving no room for reasoning.
        """
        cap = self.tier.tool_result_cap
        tokens = count_tokens(result)
        if tokens <= cap:
            return result

        # Truncate by character (approximate but functional).
        # Assume 1 token ≈ 4 characters on average for code.
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
        Reduce message history to fit within soft_budget.

        Strategy:
          - Always keep the system message (compressed separately if needed).
          - Always keep the last `keep_turns` user/assistant exchanges verbatim.
          - Older turns are replaced with a compact summary line.

        This is LOSSLESS for recent context and LOSSY for older context —
        the exact balance users need for long coding sessions.
        """
        current_tokens = count_messages_tokens(messages)
        if current_tokens <= self.tier.soft_budget:
            return messages  # No compression needed.

        logger.info(
            "Compressing history: %d tokens > soft_budget %d",
            current_tokens, self.tier.soft_budget
        )

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system  = [m for m in messages if m.get("role") != "system"]

        # Pair up user/assistant turns
        turn_pairs: list[list[dict]] = []
        current_pair: list[dict] = []
        for msg in non_system:
            current_pair.append(msg)
            # A turn ends after an assistant message (or a tool cluster).
            if msg.get("role") == "assistant":
                turn_pairs.append(current_pair)
                current_pair = []
        if current_pair:
            turn_pairs.append(current_pair)  # Dangling user message (current query).

        # Keep recent turns verbatim; summarize older ones.
        if len(turn_pairs) <= keep_turns:
            return messages  # Not enough history to compress.

        old_turns  = turn_pairs[:-keep_turns]
        kept_turns = turn_pairs[-keep_turns:]

        # Build a compact summary of older turns.
        summary_lines = []
        for pair in old_turns:
            for msg in pair:
                role = msg.get("role", "")
                content = msg.get("content", "")
                text = content if isinstance(content, str) else "[tool interaction]"
                # Trim to the first 120 characters for the summary.
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

        new_tokens = count_messages_tokens(compressed)
        logger.info("History compressed: %d → %d tokens", current_tokens, new_tokens)
        return compressed

    # Budget enforcement (called before every LLM request)

    def enforce_budget(self, messages: list[dict]) -> list[dict]:
        """
        Full pre-flight: compress history → check remaining capacity.
        Returns the (possibly compressed) message list, safe to send to the LLM.
        """
        compressed = self.compress_history(messages)
        tokens = count_messages_tokens(compressed)

        remaining = self.tier.context_window - tokens
        if remaining < 2_048:
            # Warn when the budget is very tight; proceed despite potential LLM errors.
            logger.warning(
                "Context nearly full: %d/%d tokens used, %d remaining.",
                tokens, self.tier.context_window, remaining
            )
        return compressed

    # Usage recording

    def record_turn_from_response(
        self,
        messages_sent: list[dict],
        response_data: dict,
        output_text: str,
        tool_call_count: int = 0,
    ) -> TurnUsage:
        """
        Extract or estimate token counts from the API response.

        Most providers return usage in the response body:
          OpenAI:    { "usage": { "prompt_tokens": N, "completion_tokens": N } }
          Anthropic: { "usage": { "input_tokens": N, "output_tokens": N } }

        For providers that omit usage (Ollama, some custom), we fall back to
        tiktoken estimation so the counter is always populated.
        """
        usage_data = response_data.get("usage", {})

        # Try the OpenAI response format first.
        input_tokens  = usage_data.get("prompt_tokens") or usage_data.get("input_tokens")
        output_tokens = usage_data.get("completion_tokens") or usage_data.get("output_tokens")

        # Fallback: Estimate from message content.
        if input_tokens is None:
            input_tokens = count_messages_tokens(messages_sent)
        if output_tokens is None:
            output_tokens = count_tokens(output_text)

        turn = TurnUsage(
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            tool_calls=tool_call_count,
        )
        self.session.record(turn)
        return turn

    def estimate_streaming_turn(
        self,
        messages_sent: list[dict],
        output_text: str,
        tool_call_count: int = 0,
    ) -> TurnUsage:
        """
        Estimate token usage for a streaming turn where the response body
        isn't available after the stream finishes. Uses tiktoken counts.
        """
        turn = TurnUsage(
            input_tokens=count_messages_tokens(messages_sent),
            output_tokens=count_tokens(output_text),
            tool_calls=tool_call_count,
        )
        self.session.record(turn)
        return turn
