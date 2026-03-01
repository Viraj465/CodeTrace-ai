"""
Agent Orchestrator: The Reasoning Engine (Agentic Loop).

Instead of a single-shot RAG pipeline, the LLM now has access to a
toolkit and autonomously decides which tools to call and when.

Tools available:
  - search_codebase : semantic hybrid search over indexed code
  - get_symbol_relations : graph traversal (callers / dependencies)
  - read_file : read full file contents
  - analyze_impact : find all downstream dependents of a symbol
"""

import os
import json
import asyncio
import re
from pathlib import Path
from typing import Dict, Any, Generator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from src.core.graph.builder import CodeGraph
from src.backend.vector_store import VectorStore
from src.core.agents.tools import create_tools, inspect_index_impl

# System Prompt

SYSTEM_PROMPT = (
    "You are Codetrace-ai, an expert Autonomous System Architect. "
    "You have access to a set of tools that let you investigate AND modify a codebase:\n\n"
    "1. **search_codebase** — Semantic search for relevant code symbols.\n"
    "2. **inspect_index** — Check indexed file coverage and list indexed files.\n"
    "3. **get_symbol_relations** — Find what calls a symbol and what it depends on.\n"
    "4. **read_file** — Read indexed file content from DB snapshots.\n"
    "5. **analyze_impact** — Find all downstream dependents of a symbol.\n"
    "6. **write_file** — Propose a file change (user must approve before it takes effect).\n"
    "7. **git_diff** — Show git diff to review recent changes or compare branches.\n\n"
    "WORKFLOW:\n"
    "- For high-level architecture or entrypoint/routing questions, ALWAYS call `inspect_index` first.\n"
    "- ALWAYS use `search_codebase` after `inspect_index` to find relevant code symbols.\n"
    "- If a user asks about structure/architecture, also use `get_symbol_relations`.\n"
    "- If you need more context (imports, constants), use `read_file`.\n"
    "- If the user asks about impact or breakage, use `analyze_impact`.\n"
    "- If the user asks you to fix, refactor, or generate code, use `write_file`.\n"
    "- If the user asks about recent changes or PR review, use `git_diff`.\n"
    "- You may call multiple tools in sequence to build a complete picture.\n\n"
    "EDIT BATCH POLICY:\n"
    "- When proposing edits, prepare one coherent batch, summarize it, and stop.\n"
    "- If more fixes remain, explicitly list them as next-batch items instead of continuing tool loops.\n\n"
    "EVIDENCE POLICY:\n"
    "- Use only DB-backed tool evidence; do not assume filesystem access.\n"
    "- If indexed evidence is insufficient or missing, explicitly say so and ask for re-index.\n"
    "- Do NOT invent framework names, entrypoints, or routing files.\n\n"
    "PROACTIVE ISSUE DETECTION:\n"
    "While investigating code for the user's question, if you notice:\n"
    "  • Deprecated APIs or outdated patterns\n"
    "  • Potential bugs or unhandled edge cases\n"
    "  • Missing error handling or security issues\n"
    "  • Performance anti-patterns\n"
    "Then MENTION them clearly in your response and explain what's wrong.\n"
    "If the fix is straightforward, use `write_file` to propose the fix.\n"
    "The user will see a diff preview and can approve or decline.\n"
    "NEVER assume the user wants a change — always explain WHY first.\n\n"
    "After gathering enough context, provide a clear, well-structured answer "
    "using Markdown formatting. Cite specific file paths and symbol names."
)

# Human-friendly descriptions for tool calls
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
    # Fallback: just show the tool name
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


class AgentOrchestrator:
    def __init__(self, vector_store: VectorStore, graph: CodeGraph):
        self.vector_store = vector_store
        self.graph = graph
        self.recursion_limit = int(os.getenv("CODETRACE_RECURSION_LIMIT", "60"))
        self.llm = self._initialize_llm()
        self.tools = create_tools(vector_store, graph)
        self.agent_executor = self._create_agent_executor()

    
    # Config & LLM init
    

    def _get_global_config(self) -> Dict[str, Any]:
        """Reads the global LLM config from the user's home directory."""
        config_path = Path.home() / ".codetrace" / "config.json"
        if not config_path.exists():
            raise ValueError("LLM not configured. Run 'codetrace config' first.")
        with open(config_path, "r") as f:
            return json.load(f)

    def _initialize_llm(self):
        """Dynamically loads the LangChain model based on user preference."""
        config = self._get_global_config()
        provider = config.get("provider", "").lower()
        api_key = config.get("api_key", "")
        model_name = config.get("model_name", "")

        if not api_key and provider not in ("ollama",):
            raise ValueError(f"API key missing for provider: {provider}")

        if api_key:
            os.environ[f"{provider.upper()}_API_KEY"] = api_key

        if provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(api_key=api_key, model_name=model_name or "llama3-70b-8192")

        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(api_key=api_key, model_name=model_name or "gpt-4o")

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(api_key=api_key, model_name=model_name or "claude-3-5-sonnet-20240620")

        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(google_api_key=api_key, model=model_name or "gemini-1.5-pro")

        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            base_url = config.get("base_url", "http://localhost:11434")
            return ChatOllama(model=model_name or "llama3.2", base_url=base_url)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    
    # Agent creation
    

    def _create_agent_executor(self):
        """Create a tool-calling agent with the configured LLM and tools."""
        return create_react_agent(
            self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT
        )

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

        # keep order, drop duplicates
        seen = set()
        ordered = []
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

    def _build_messages(self, query: str, chat_history: list | None = None) -> list:
        messages = []
        if chat_history:
            messages.extend([
                HumanMessage(content=c) if r == "user" else AIMessage(content=c)
                for r, c in chat_history
            ])

        auto_index_context = self._build_auto_index_context(query)
        messages.append(SystemMessage(content=(
            "DB-only enforcement: Use only indexed DB evidence. "
            "Do not assume filesystem access. "
            "If evidence is missing, explicitly ask for re-index.\n\n"
            f"Automatic index preflight:\n{auto_index_context}"
        )))
        messages.append(HumanMessage(content=query))
        return messages

    
    # Public API
    

    def ask(self, query: str, chat_history: list | None = None) -> str:
        """Blocking execution of the agentic pipeline."""
        messages = self._build_messages(query, chat_history)

        result = self.agent_executor.invoke(
            {"messages": messages},
            config={"recursion_limit": self.recursion_limit},
        )
        return _normalize_text_content(result["messages"][-1].content)

    async def astream(self, query: str, chat_history: list | None = None):
        """
        Async streaming via astream_events v2.
        Yields structured event dicts for the CLI to display:

        {"type": "thought",  "message": "Searching codebase for: auth logic"}
        {"type": "tool_end", "tool": "search_codebase"}
        {"type": "token",    "content": "The"}
        {"type": "done"}
        """
        messages = self._build_messages(query, chat_history)
        
        async for event in self.agent_executor.astream_events(
            {"messages": messages},
            version="v2",
            config={"recursion_limit": self.recursion_limit},
        ):
            kind = event.get("event", "")

            # ── Tool invocation start ──
            if kind == "on_tool_start":
                tool_name = event.get("name", "")
                tool_input = event.get("data", {}).get("input", {})
                if isinstance(tool_input, str):
                    tool_input = {"query": tool_input}
                yield {
                    "type": "thought",
                    "tool": tool_name,
                    "message": _humanize_tool_call(tool_name, tool_input),
                }

            # ── Tool finished ──
            elif kind == "on_tool_end":
                yield {
                    "type": "tool_end",
                    "tool": event.get("name", ""),
                }

            # ── LLM token streamed (final answer) ──
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk:
                    # Ignore tool call chunks
                    if getattr(chunk, "tool_call_chunks", []):
                        continue

                    content_text = _normalize_text_content(getattr(chunk, "content", None))
                    if not content_text:
                        continue

                    # Only emit tokens from the final answer, not inner reasoning
                    parent_ids = event.get("parent_ids", [])
                    if parent_ids:  # has a parent = it's from the agent's LLM
                        yield {
                            "type": "token",
                            "content": content_text,
                        }

        yield {"type": "done"}

    def stream(self, query: str, chat_history: list | None = None) -> Generator[dict, None, None]:
        """
        Synchronous wrapper around astream() for use in non-async CLI code.
        Creates its own event loop if needed.
        """
        async def _collect():
            events = []
            async for event in self.astream(query, chat_history):
                events.append(event)
            return events

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an async context — use nest_asyncio or fallback
            import nest_asyncio
            nest_asyncio.apply()
            events = loop.run_until_complete(_collect())
        else:
            events = asyncio.run(_collect())

        yield from events
