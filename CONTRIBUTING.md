**Contributing to Codetrace-ai**

*First off, thank you for considering contributing to Codetrace-ai! Whether you're fixing a bug, adding a new language, or improving the AI agent's reasoning, your help makes this "Hybrid Brain" smarter for everyone.*

🚀 Getting Started

1. Prerequisites
- Python 3.10+
- Ollama (recommended for local LLM testing)
- Git

2. Setup your Environment
  Fork the repository and clone it locally:
```
  git clone [https://github.com/YOUR_USERNAME/codetrace-ai.git](https://github.com/YOUR_USERNAME/codetrace-ai.git)
  cd codetrace-ai
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -e ".[dev]"
```

The -e ".[dev]" command installs the package in editable mode along with development dependencies like pytest, black, and isort.

🛠 Project Architecture

Codetrace-ai is built on a Dual-Backend "Hybrid Brain" system. Understanding this is key to contributing:

1. Semantic Brain (Vector Store): Uses ChromaDB to store code embeddings. This allows the agent to find code based on meaning/intent rather than just keywords. (See src/backend/vector_store.py)
2. Structural Brain (Graph Store): Uses SQLite and NetworkX to map out relationships (who calls what, class ownership). This powers the Impact Analysis and Call Graph features. (See src/core/graph/builder.py)

When you index a file, the GraphOrchestrator coordinates both "brains" to ensure the AI has a complete picture.

🌍 Adding New Language Support

Adding support for a new programming language is a great way to start.
1. Map the Extension: Add the file extension to the EXTENSIONS_MAP in src/file_extension.py.
2. Define AST Queries: Create a new Tree-sitter query file in src/core/parser/queries/{language}.scm.
 - Use @symbol.name and @symbol.definition for classes/functions.
 - Use @call.name for function calls.

Update Parser: Ensure the language is added to LANGUAGE_MODULES in src/core/parser/parser.py


🐞 Reporting Bugs & Known Issues

If you find a bug, please open an issue with:
1. A clear description of the bug.
2. Steps to reproduce.
3. Your OS and Python version.

Current focus areas:
1. AST Accuracy: Fixing typos in .scm query files (e.g., ensuring capture names like @call don't have trailing punctuation).
2. Large Repo Scaling: Optimizing SQLite queries for repos with >1000 symbols to avoid variable limits.

🧪 Testing
We use pytest for all logic verification. Please add tests for any new features.

```
pytest tests/
```
🎨 Code Style

To keep the codebase maintainable:
 1. Formatting: We use black for code formatting and isort for import sorting.
 2. Type Hints: Use Python type hints (def func(name: str) -> bool:) for all new code.

Agent Policy: Ensure any changes to the AI agent follow the EVIDENCE POLICY (only use DB-backed evidence, don't guess).

📬 Pull Request Process

Create a feature branch: git checkout -b feat/your-feature.
Commit your changes with clear messages.
Ensure all tests pass.
Open a PR against the main branch.

Wait for the maintainers to review (we try to respond within 48 hours!).

📜 License

**By contributing to Codetrace-ai, you agree that your contributions will be licensed under the MIT License.**

Happy Tracing!
