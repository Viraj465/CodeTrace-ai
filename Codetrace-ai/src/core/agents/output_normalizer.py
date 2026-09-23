"""
Output Normalizer: Provider- and tier-agnostic response consistency layer.

Different LLM providers (Anthropic, OpenAI, Groq, Gemini, Ollama, …) format
their answers differently — heading levels drift, list bullets vary (*, +, -),
code-fence language tags use aliases (py vs python), and spacing around
headings is inconsistent. This module normalizes all of that into a single,
predictable Markdown shape so the CLI renders uniformly regardless of backend.

It also provides light-weight quality checks used for telemetry / debugging:
  - Citation extraction & presence checks (file:line evidence).
  - Fact-vs-inference separation heuristics.
  - Code-block extraction & validation.

Nothing here calls the network; it is pure text processing. The public surface
matches the contract exercised by tests/test_output_normalizer.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union


# Data models


@dataclass
class Citation:
    """A `file:line` (or `file`) reference pulled from model output."""
    file_path: str
    line_number: Optional[int] = None


@dataclass
class CodeBlock:
    """A fenced code block with its (normalized) language and body."""
    language: str
    content: str


class QueryType(Enum):
    """Coarse query categories that drive citation-density expectations."""
    GENERAL = "general"
    EXPLANATION = "explanation"
    BLAST_RADIUS = "blast_radius"
    SEMANTIC_SEARCH = "semantic_search"
    ARCHITECTURE = "architecture"


def _coerce_query_type(query_type: Union[str, QueryType, None]) -> QueryType:
    """Accept either a QueryType or a string label and return a QueryType."""
    if isinstance(query_type, QueryType):
        return query_type
    if not query_type:
        return QueryType.GENERAL
    try:
        return QueryType(str(query_type).strip().lower())
    except ValueError:
        return QueryType.GENERAL


# Shared regexes


# A fenced code block, indentation-tolerant, with an optional language tag.
_FENCE_RE = re.compile(
    r"^[ \t]*```([^\n`]*)\r?\n(.*?)^[ \t]*```",
    re.DOTALL | re.MULTILINE,
)
# Any fenced block (used to blank out code before scanning prose).
_ANY_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# Inline `...` spans.
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
# A path-like token: has an extension, optionally suffixed with :line.
_PATH_RE = re.compile(r"^(?P<path>[\w./\\-]+\.[A-Za-z0-9_]+)(?::(?P<line>\d+))?$")
# A single-level heading (exactly one leading #).
_H1_RE = re.compile(r"^#(?!#)[ \t]*", re.MULTILINE)
# A list bullet using *, +, or -.
_BULLET_RE = re.compile(r"^(\s*)[*+\-][ \t]+", re.MULTILINE)
# Any heading, level 1-6.
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s")


# Citation validation


class CitationValidator:
    """Extract and format `file:line` citations from model output."""

    @staticmethod
    def extract_citations(text: str) -> List[Citation]:
        """
        Return every inline citation of the form `path/to/file.ext` or
        `path/to/file.ext:line`. Code fences are ignored so code bodies are
        never mistaken for citations.
        """
        if not text:
            return []
        clean = _ANY_FENCE_RE.sub("", text)
        citations: List[Citation] = []
        for match in _INLINE_CODE_RE.finditer(clean):
            token = match.group(1).strip()
            path_match = _PATH_RE.match(token)
            if not path_match:
                continue
            line = path_match.group("line")
            citations.append(
                Citation(
                    file_path=path_match.group("path"),
                    line_number=int(line) if line else None,
                )
            )
        return citations

    @staticmethod
    def format_citation(file_path: str, line_number: Optional[int] = None) -> str:
        """Render a citation as `file:line` (or `file` when no line)."""
        if line_number is not None:
            return f"`{file_path}:{line_number}`"
        return f"`{file_path}`"


# Code-block handling


class CodeBlockFormatter:
    """Extract, validate, and format fenced code blocks."""

    # Common language aliases → canonical fence tag.
    LANGUAGE_ALIASES = {
        "py": "python",
        "python3": "python",
        "js": "javascript",
        "node": "javascript",
        "jsx": "javascript",
        "ts": "typescript",
        "typescriptreact": "tsx",
        "rb": "ruby",
        "yml": "yaml",
        "sh": "bash",
        "shell": "bash",
        "zsh": "bash",
        "kt": "kotlin",
        "cs": "csharp",
        "c++": "cpp",
        "cxx": "cpp",
        "md": "markdown",
        "rs": "rust",
        "golang": "go",
    }

    @staticmethod
    def normalize_language(language: str) -> str:
        """Map a language alias to its canonical form (py → python)."""
        if not language:
            return ""
        lowered = language.strip().lower()
        return CodeBlockFormatter.LANGUAGE_ALIASES.get(lowered, lowered)

    @staticmethod
    def extract_code_blocks(text: str) -> List[CodeBlock]:
        """Return all fenced code blocks with normalized language tags."""
        if not text:
            return []
        blocks: List[CodeBlock] = []
        for match in _FENCE_RE.finditer(text):
            language = CodeBlockFormatter.normalize_language(match.group(1).strip())
            content = match.group(2).rstrip("\n")
            blocks.append(CodeBlock(language=language, content=content))
        return blocks

    @staticmethod
    def validate_code_block(block: CodeBlock) -> Tuple[bool, List[str]]:
        """
        Validate a code block. Returns (is_valid, issues).
        A block is invalid if it has no content; a missing language is a
        non-fatal issue that is still reported.
        """
        issues: List[str] = []
        if not block.content or not block.content.strip():
            issues.append("Code block is empty")
        if not block.language:
            issues.append("Code block has no language specified")
        # Only emptiness makes a block invalid.
        is_valid = not any(i == "Code block is empty" for i in issues)
        return is_valid, issues

    @staticmethod
    def format_code_block(language: str, content: str) -> str:
        """Render a fenced code block with a normalized language tag."""
        lang = CodeBlockFormatter.normalize_language(language)
        return f"```{lang}\n{content}\n```"


# Markdown normalization


class MarkdownNormalizer:
    """Normalize headings, list bullets, and spacing to a consistent shape."""

    @staticmethod
    def normalize_headings(text: str) -> str:
        """Promote single-# (h1) headings to ## so answers never start at h1."""
        return _H1_RE.sub("## ", text)

    @staticmethod
    def normalize_lists(text: str) -> str:
        """Rewrite *, +, and - bullets to a consistent '- ' bullet."""
        return _BULLET_RE.sub(r"\1- ", text)

    @staticmethod
    def add_missing_spacing(text: str) -> str:
        """
        Ensure a blank line sits before and after every heading line.
        Lines inside ``` fences are code, not headings — a Python or shell
        '# comment' must come through untouched.
        """
        lines = text.split("\n")
        out: List[str] = []
        in_code = False
        for i, line in enumerate(lines):
            if line.lstrip().startswith("```"):
                in_code = not in_code
                out.append(line)
                continue
            is_heading = not in_code and bool(_HEADING_LINE_RE.match(line))
            if is_heading and out and out[-1].strip() != "":
                out.append("")
            out.append(line)
            if is_heading and i + 1 < len(lines) and lines[i + 1].strip() != "":
                out.append("")
        return "\n".join(out)


# Quality checks


class QualityChecker:
    """Heuristic checks for citation density and fact/inference hygiene."""

    # Minimum citation count expected per query type.
    _MIN_CITATIONS = {
        QueryType.GENERAL: 0,
        QueryType.EXPLANATION: 1,
        QueryType.SEMANTIC_SEARCH: 1,
        QueryType.BLAST_RADIUS: 2,
        QueryType.ARCHITECTURE: 2,
    }

    # Phrases that explicitly mark a statement as inference, not fact.
    _INFERENCE_MARKERS = (
        "based on", "appears to", "looks like", "seems to", "seems like",
        "likely", "probably", "suggests", "presumably", "i think",
        "might be", "could be", "may be",
    )

    # Verbs that signal a definitive structural claim.
    _CLAIM_RE = re.compile(
        r"\b(is|are|handles?|implements?|calls?|defined|contains?|returns?|uses?|imports?)\b"
    )

    @staticmethod
    def check_citation_presence(
        text: str, query_type: Union[str, QueryType]
    ) -> Tuple[bool, float]:
        """
        Return (has_enough, score) where score is citations_found / required.
        When a query type requires no citations, score is 1.0.
        """
        qt = _coerce_query_type(query_type)
        found = len(CitationValidator.extract_citations(text))
        required = QualityChecker._MIN_CITATIONS.get(qt, 0)
        if required <= 0:
            return True, 1.0
        score = found / required
        return found >= required, score

    @staticmethod
    def check_fact_vs_inference_separation(text: str) -> Tuple[bool, List[str]]:
        """
        Flag definitive structural claims that carry neither a citation nor an
        inference marker. Returns (is_ok, issues).
        """
        issues: List[str] = []
        clean = _ANY_FENCE_RE.sub("", text)
        for raw_sentence in re.split(r"(?<=[.!?])\s+|\n", clean):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            lowered = sentence.lower()
            has_marker = any(m in lowered for m in QualityChecker._INFERENCE_MARKERS)
            has_citation = bool(CitationValidator.extract_citations(sentence))
            is_claim = bool(QualityChecker._CLAIM_RE.search(lowered))
            if is_claim and not has_marker and not has_citation:
                issues.append(f"Uncited definitive claim: {sentence[:60]}")
        return len(issues) == 0, issues


# Main normalizer


class OutputNormalizer:
    """
    Normalize raw LLM output into a consistent Markdown shape and surface a
    quality report. Code fences are preserved verbatim; only prose and fence
    language tags are rewritten.
    """

    def __init__(self, tier: str = "tier2"):
        self.tier = tier

    def normalize(self, raw: str, query_type: Union[str, QueryType] = "general") -> str:
        """
        Return `raw` with consistent headings, bullets, fence languages, and
        heading spacing. Empty or whitespace-only input returns "".
        """
        if not raw or not raw.strip():
            return ""

        out: List[str] = []
        in_code = False
        for line in raw.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("```"):
                if not in_code:
                    indent = line[: len(line) - len(stripped)]
                    lang = CodeBlockFormatter.normalize_language(stripped[3:].strip())
                    out.append(f"{indent}```{lang}")
                    in_code = True
                else:
                    out.append(line)
                    in_code = False
                continue

            if in_code:
                out.append(line)
                continue

            # Prose line: normalize heading level and list bullet.
            line = _H1_RE.sub("## ", line)
            line = _BULLET_RE.sub(r"\1- ", line)
            out.append(line)

        text = MarkdownNormalizer.add_missing_spacing("\n".join(out))
        return text.strip()

    def get_quality_report(
        self, text: str, query_type: Union[str, QueryType] = "general"
    ) -> dict:
        """Return a quality/telemetry report for a (usually normalized) answer."""
        qt = _coerce_query_type(query_type)
        citations = CitationValidator.extract_citations(text)
        code_blocks = CodeBlockFormatter.extract_code_blocks(text)
        has_structure = bool(re.search(r"^#{2,6}\s", text, re.MULTILINE))
        has_enough, citation_score = QualityChecker.check_citation_presence(text, qt)
        fact_ok, fact_issues = QualityChecker.check_fact_vs_inference_separation(text)

        quality_score = (
            0.5 * min(citation_score, 1.0)
            + 0.25 * (1.0 if has_structure else 0.0)
            + 0.25 * (1.0 if fact_ok else 0.0)
        )

        return {
            "quality_score": round(quality_score, 3),
            "citations_count": len(citations),
            "code_blocks_count": len(code_blocks),
            "has_structure": has_structure,
            "has_enough_citations": has_enough,
            "fact_inference_ok": fact_ok,
            "fact_inference_issues": fact_issues,
            "tier": self.tier,
        }
