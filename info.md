I'll trace how the prompt interacts with LLM providers through the registry tier system. Let me examine the token manager and orchestrator to show you the complete flow.Perfect! Now let me create a visual explanation of how the prompt works with the registry tier system:Excellent! Here's the complete comprehensive overview:

---

## 🎯 **How the Unified Prompt Works with LLM Registry Tier System**

### **PART 1: THE TIER SYSTEM ARCHITECTURE**

The CodeTrace AI system dynamically adapts the prompt's execution based on the LLM provider's **capability tier**. This is determined at runtime, not hardcoded.

#### **Tier Detection Flow**

```
┌─────────────────────────────────────────────────────────┐
│  User provides: model_name (e.g., "gpt-4o", "claude-3-5-sonnet")
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  resolve_tier(model_name) — Dynamic tier resolution    │
└─────────────────────────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   Ollama?    │ │  Cached?     │ │  litellm?    │
    │ Query API    │ │  (24h TTL)   │ │  Registry    │
    │ for context  │ │              │ │              │
    └──────────────┘ └──────────────┘ └──────────────┘
              │           │              │
              └───────────┴──────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │  Get context_window & max_output   │
         └────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  _tier_from_context_window() — Classify into tier      │
└─────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │  TIER 1 │      │  TIER 2 │      │  TIER 3 │
    │ <32K   │      │32K-100K │      │>100K    │
    └─────────┘      └─────────┘      └─────────┘
```

---

### **PART 2: TIER SPECIFICATIONS**

```python
┌────────────────────────────────────────────────────────────────────────────────┐
│                            TIER SPECIFICATIONS                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  TIER 1 (Context Window < 32K tokens)                                         │
│  ─────────────────────────────────────                                        │
│  Examples: GPT-3.5-turbo, Llama 2 7B, Gemma 7B, Ollama small models          │
│  Context Window: ~8K - 16K                                                    │
│  Soft Budget: 55% of context (4.4K - 8.8K tokens)                            │
│  Tool Result Cap: 2,000 tokens per tool                                       │
│  Max Output Tokens: 2,048 tokens                                              │
│  ┌─────────────────────────────────────────────────────────┐                 │
│  │ Prompt Adaptation:                                      │                 │
│  │ • Shorter system prompt (focus on essentials)          │                 │
│  │ • Aggressive history compression (keep only 3 turns)   │                 │
│  │ • Truncate file reads to <2K tokens                    │                 │
│  │ • Limit search results to top 3 items                  │                 │
│  │ • Skip detailed error context in tool results          │                 │
│  │ • Prefer narrow queries over broad exploration         │                 │
│  └─────────────────────────────────────────────────────────┘                 │
│                                                                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  TIER 2 (Context Window 32K - 100K tokens)                                    │
│  ─────────────────────────────────────────                                    │
│  Examples: GPT-4, Claude 3 Opus, Gemma 27B, Llama 3.1 70B                     │
│  Context Window: ~32K - 100K                                                  │
│  Soft Budget: 65% of context (20.8K - 65K tokens)                            │
│  Tool Result Cap: 4,000 tokens per tool                                       │
│  Max Output Tokens: 4,000-8,000 tokens                                        │
│  ┌─────────────────────────────────────────────────────────┐                 │
│  │ Prompt Adaptation:                                      │                 │
│  │ • Full system prompt (all 16 sections)                 │                 │
│  │ • Moderate history compression (keep 6 turns)          │                 │
│  │ • Tool results can be 4K tokens (full files allowed)   │                 │
│  │ • Search results can return top 8 items                │                 │
│  │ • Include detailed context in error messages           │                 │
│  │ • Can perform multi-hop tracing (6+ hops)             │                 │
│  │ • Full reverse-engineering workflows available        │                 │
│  └─────────────────────────────────────────────────────────┘                 │
│                                                                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  TIER 3 (Context Window > 100K tokens)                                        │
│  ────────────────────────────────────                                         │
│  Examples: GPT-4 Turbo, Claude 3.5 Sonnet, Gemini 2.5 Pro, Qwen 32B          │
│  Context Window: ~100K - 1M+                                                  │
│  Soft Budget: 60% of context (60K - 600K+ tokens)                            │
│  Tool Result Cap: 8,000 tokens per tool                                       │
│  Max Output Tokens: 8,192+ tokens                                             │
│  ┌─────────────────────────────────────────────────────────┐                 │
│  │ Prompt Adaptation:                                      │                 │
│  │ • Full comprehensive system prompt (all sections)       │                 │
│  │ • Minimal history compression (keep 12+ turns)         │                 │
│  │ • Tool results can be 8K+ tokens (huge files OK)       │                 │
│  │ • Search results can return 15+ items                  │                 │
│  │ • Include ALL context and examples                     │                 │
│  │ • Can handle complex multi-hop tracing (20+ hops)      │                 │
│  │ • Full architectural analysis in one session           │                 │
│  │ • Can process entire codebase sections together        │                 │
│  └─────────────────────────────────────────────────────────┘                 │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

### **PART 3: HOW THE UNIFIED PROMPT ADAPTS TO EACH TIER**

#### **3.1 — System Prompt Injection (Tier-Aware)**

```python
# In retriever.py → _build_messages()

def _build_messages(self, query: str, chat_history: list | None = None) -> list[dict]:
    messages: list[dict] = []
    
    # TIER AWARENESS: Token budget manager has already resolved the tier
    tier = self.budget.tier  # Tier 1, 2, or 3
    
    # Build the system message with tier-specific adaptations
    system_content = (
        f"{SYSTEM_PROMPT}\n\n"  # ← This is the unified prompt (static)
        "DB-only enforcement: Use only indexed DB evidence. "
        f"\n\nAutomatic index preflight:\n{auto_index_context}"
    )
    
    # TIER 1: Compress the system prompt
    if tier.name == "tier1":
        system_content = _compress_prompt_for_tier1(system_content)
        # Removes: detailed workflows, examples, legacy red flags sections
    
    # TIER 2: Use as-is (full prompt)
    elif tier.name == "tier2":
        # No change — full unified prompt
        pass
    
    # TIER 3: Expand with additional reasoning guidelines
    elif tier.name == "tier3":
        system_content += "\n\n" + _expand_prompt_for_tier3()
        # Adds: Advanced reasoning patterns, edge case handling
    
    messages.append({"role": "system", "content": system_content})
    # ... rest of message building
```

**Key insight:** The **unified prompt is static**, but the system dynamically **adds/removes/modifies sections** based on the detected tier.

---

#### **3.2 — Tool Invocation Patterns by Tier**

| Tier | Tool Selection Pattern | Query Strategy | Multi-Hop Budget |
|------|---|---|---|
| **TIER 1** | Narrow only | `search_codebase` → `read_file` (stop) | Max 2-3 hops |
| **TIER 2** | Balanced | `inspect_index` → `search_codebase` → `get_symbol_relations` | Max 6-8 hops |
| **TIER 3** | Comprehensive | Full standard investigation protocol (7 steps) | Max 15-20 hops |

**Example: "How does authentication work?" query**

```
TIER 1 (8K context):
  Step 1: search_codebase("authentication")
    ↓ Returns top 3 results (not top 15)
  Step 2: read_file(result[0]) — max 2K tokens
    ↓ STOP — context almost full
  Done.

TIER 2 (32K context):
  Step 1: inspect_index() — map project structure
  Step 2: search_codebase("authentication") — top 8 results
  Step 3: get_symbol_relations(auth_module) — map callers
  Step 4: read_file(entry_point) — 4K tokens
  Step 5: analyze_impact(login_function) — see who calls it
  Step 6: Document findings — with examples
  Done.

TIER 3 (100K+ context):
  Step 1: inspect_index() — full structure + metadata
  Step 2: search_codebase("authentication") — top 15 results
  Step 3: get_symbol_relations(all_auth_symbols) — comprehensive
  Step 4: read_file(auth_module) — full 8K tokens
  Step 5: read_file(middleware) — trace execution chain
  Step 6: analyze_impact(login_system) — deep blast radius
  Step 7: get_symbol_relations(security_utils) — validate
  Step 8: analyze_proactive_issues() — security scan
    → Find hardcoded credentials, injection risks, etc.
  Step 9: Propose refactoring across architectural layers
  Done.
```

---

#### **3.3 — History Compression Strategy by Tier**

```python
def compress_history(self, messages, keep_system=True, keep_turns=6):
    """
    Dynamic keep_turns based on tier:
    
    TIER 1: keep_turns = 3 (last 3 user/assistant pairs)
      → ~1.5 exchanges before automatic compression
      → Oldest turns summarized into 1-2 line summaries
    
    TIER 2: keep_turns = 6 (last 6 user/assistant pairs)
      → ~3 exchanges before compression kicks in
      → Detailed older turns compressed to 3-4 lines
    
    TIER 3: keep_turns = 12+ (last 12+ exchanges)
      → ~6+ exchanges before ANY compression
      → Minimal summarization; preserve recent reasoning
    """
    
    current_tokens = count_messages_tokens(messages)
    
    if current_tokens <= self.tier.soft_budget:
        return messages  # No compression needed
    
    # Dynamically reduce keep_turns if hard limit approached
    while keep_turns >= 1:
        compressed = build_compressed_history(messages, keep_turns)
        new_tokens = count_messages_tokens(compressed)
        
        if new_tokens <= self.tier.context_window - self.tier.max_output_tokens:
            break  # Fits in hard limit
        keep_turns -= 1
    
    return compressed
```

---

### **PART 4: COMPLETE EXECUTION FLOW DIAGRAM**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          USER QUERY SUBMITTED                                │
│                    (e.g., "How does auth work?")                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  AgentOrchestrator.ask(query)                                                │
│  ├─ 1. Resolve tier via resolve_tier(model_name)                            │
│  │     • Query provider (litellm, Ollama, etc.) for context window          │
│  │     • Cache result for 24 hours                                           │
│  │                                                                            │
│  ├─ 2. Initialize TokenBudgetManager with detected tier                     │
│  │     tier.context_window = 32000   (for Claude 3 Opus)                   │
│  │     tier.soft_budget = 20800      (65% of 32K)                          │
│  │     tier.tool_result_cap = 4000   (per tool)                            │
│  │     tier.max_output_tokens = 4000 (model can generate)                  │
│  │                                                                            │
│  ├─ 3. Build messages list via _build_messages(query)                       │
│  │     ├─ System message: UNIFIED PROMPT + tier-specific adaptations        │
│  │     ├─ Auto index context: inspect_index() preflight                     │
│  │     └─ User message: the actual query                                    │
│  │                                                                            │
│  └─ 4. Enforce budget before LLM call                                       │
│       └─ compress_history() if messages > soft_budget                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  ─────────────────────  TIER-SPECIFIC PROMPT FLOW  ────────────────────────  │
│                                                                               │
│  TIER 1 (8K context)           TIER 2 (32K context)      TIER 3 (100K+ ctx) │
│  ──────────────────           ───────────────────        ────────────────── │
│  System msg: ~1.2K            System msg: ~3.5K          System msg: ~4K   │
│  + Compressed essentials      + Full unified prompt      + Full + expansions│
│  + Narrow query guidance      + Balanced workflows       + Advanced examples│
│                                                                              │
│  Max tool results: 2K         Max tool results: 4K       Max tool results: 8K│
│  Search: top 3 items          Search: top 8 items       Search: top 15 items│
│  Keep turns: 3                Keep turns: 6             Keep turns: 12+     │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LLM API Call (Provider-specific)                                            │
│  ├─ OpenAI: POST /v1/chat/completions                                       │
│  │   └─ Headers: Authorization: Bearer $API_KEY                             │
│  ├─ Anthropic: POST /v1/messages (native API)                              │
│  │   └─ Headers: x-api-key: $API_KEY                                        │
│  ├─ Groq/Gemini/Ollama: OpenAI-compatible endpoint                          │
│  │   └─ Sanitized tool schemas (remove unsupported fields)                  │
│  └─ All: Payload includes:                                                  │
│       • model: actual model name                                            │
│       • messages: tier-adapted message list                                 │
│       • temperature: provider-specific (Gemini: 1.0, others: 0.3)          │
│       • max_tokens: tier.max_output_tokens                                  │
│       • tools: sanitized schemas for provider                               │
│       • tool_choice: "auto"                                                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  LLM Response Parsing                                                        │
│  ├─ Extract: text + tool_calls (if any)                                     │
│  ├─ Count tokens:                                                            │
│  │   • input_tokens from API response (if provided)                         │
│  │   • output_tokens from API response (if provided)                        │
│  │   • Fall back to tiktoken estimation if not provided                     │
│  │                                                                            │
│  └─ Record iteration:                                                        │
│       turn.add_iteration(input_tokens, output_tokens)                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌──────────────────────┐      ┌──────────────────────┐
        │  Tool calls found?   │      │   No tool calls      │
        │  (e.g., search,     │      │   (final answer)     │
        │   analyze_impact)   │      │                      │
        └──────────────────────┘      └──────────────────────┘
                    │                               │
                    ▼                               ▼
    ┌───────────────────────────┐    ┌──────────────────────────────┐
    │ Tool Call Loop Iteration  │    │ Return final response to CLI │
    │                           │    │ Display token usage stats    │
    │ For each tool_call:       │    │                              │
    │ 1. dispatch_tool()        │    │ Tier 1: 412 in, 89 out      │
    │    ├─ search_codebase()   │    │ Tier 2: 1243 in, 456 out    │
    │    ├─ analyze_impact()    │    │ Tier 3: 15234 in, 3421 out  │
    │    ├─ read_file()         │    │                              │
    │    └─ ...                 │    │ Total session tokens logged  │
    │                           │    │                              │
    │ 2. truncate_tool_result() │    └──────────────────────────────┘
    │    └─ Cap at tier cap     │
    │       (2K/4K/8K)          │
    │                           │
    │ 3. Append to messages     │
    │    and loop back to LLM   │
    │                           │
    │ Max iterations: 60        │
    │ (enforced via            │
    │  recursion_limit)        │
    └───────────────────────────┘
```

---

### **PART 5: CODE WALKTHROUGH EXAMPLE**

Let's trace a real query through all three tiers:

#### **Scenario: User asks "Show me the blast radius of changing get_symbol_relations()""**

##### **TIER 1 Execution (GPT-3.5-turbo, 4K context)**

```python
# 1. Detect tier
tier = resolve_tier("gpt-3.5-turbo")
# → context_window=4096, soft_budget=2200, tool_cap=2000, max_out=1024

# 2. Build system prompt
system = SYSTEM_PROMPT  # Full 16 sections
system_tokens = count_tokens(system)  # ~1800 tokens
# WARNING: Already 1800/2200 soft budget used just for system msg!

# 3. Build messages
messages = [
    {"role": "system", "content": system},  # 1800 tokens
    {"role": "user", "content": "..."}      # 150 tokens
]
# Total: ~1950 tokens (still under 2200 soft budget)

# 4. Agent orchestrator loop iteration 1
response = await openai_completion(messages)
# LLM response: Tool call to search_codebase("get_symbol_relations")

# 5. Execute tool
result = search_codebase("get_symbol_relations")
# Returns: 5 results, ~3000 tokens total

# 6. TRUNCATE: Cap at 2000 tokens
truncated_result = truncate_tool_result(result, tool_name="search_codebase")
# → Trimmed to 2000 tokens, added notice: "[TRUNCATED: 1000 tokens removed]"

# 7. Append to messages
messages.append({"role": "assistant", ...})
messages.append({"role": "tool", "content": truncated_result})
# New total: ~3950 tokens (EXCEEDS soft budget!)

# 8. Enforce budget before next LLM call
messages = compress_history(messages, keep_turns=3)
# → Keeps ONLY last 1 turn (current query)
# → Old turns (if any) → 1-line summaries
# → Result: ~2100 tokens (under soft budget again)

# 9. Agent loop iteration 2
response = await openai_completion(messages)
# LLM response: Tool call to analyze_impact("get_symbol_relations")

# 10. Execute tool
result = analyze_impact("get_symbol_relations")
# Returns blast radius: ~2500 tokens

# 11. TRUNCATE: Cap at 2000 tokens (tier 1 cap)
truncated_result = truncate_tool_result(result)
# → Trimmed, added notice

# 12. Append and compress again
messages.append(...)
messages.append(...)
messages = compress_history(messages, keep_turns=3)
# → ~2050 tokens

# 13. Agent loop iteration 3
response = await openai_completion(messages)
# LLM response: Final text answer (no tool calls)

# 14. Record usage
turn_usage.add_iteration(
    input_tokens=2050,   # from compressed messages
    output_tokens=150    # from LLM response
)
# Session total: 2050 input, 150 output, 2200 total

# 15. Return to user
return "The blast radius of changing get_symbol_relations includes..."
```

**Tier 1 Characteristics:**
- ✅ Worked, but aggressive compression after each tool call
- ✅ Tool results truncated to 2K tokens
- ✅ Only 2 tool calls before running out of room
- ⚠️ Lost context from earlier turns
- ⚠️ Couldn't provide comprehensive blast radius (truncated)

---

##### **TIER 2 Execution (Claude 3 Opus, 32K context)**

```python
# 1. Detect tier
tier = resolve_tier("claude-3-opus-20240229")
# → context_window=200000, soft_budget=130000, tool_cap=4000, max_out=4000
# Actually: resolves to context_window=32000 for older Claude 3 Opus
# → soft_budget=20800, tool_cap=4000, max_out=4000

# 2. Build system prompt
system = SYSTEM_PROMPT  # Full 16 sections
system_tokens = 3500  # Counts properly now

# 3. Build messages
messages = [
    {"role": "system", "content": system},  # 3500 tokens
    {"role": "user", "content": query}      # 200 tokens
]
# Total: ~3700 tokens (well under 20800 soft budget)

# 4. Agent loop iteration 1
response = await anthropic_completion(messages)
# LLM response: Multiple tool calls in parallel!
# Tool 1: search_codebase("get_symbol_relations")
# Tool 2: inspect_index()  ← Proactive structure check

# 5. Execute tools in parallel
result1 = search_codebase("get_symbol_relations")  # 3200 tokens
result2 = inspect_index()                          # 800 tokens
# Total: ~4000 tokens

# 6. Truncate: Cap at 4000 tokens per tool (tier 2 cap)
truncated1 = truncate_tool_result(result1)  # ~3900 tokens
truncated2 = truncate_tool_result(result2)  # ~800 tokens (fits)
# No truncation needed for result2

# 7. Append to messages
messages.append({
    "role": "assistant",
    "content": [
        {"type": "tool_use", "id": "1", "name": "search_codebase", "input": {...}},
        {"type": "tool_use", "id": "2", "name": "inspect_index", "input": {...}}
    ]
})
messages.append({
    "role": "user",
    "content": [
        {"type": "tool_result", "tool_use_id": "1", "content": truncated1},
        {"type": "tool_result", "tool_use_id": "2", "content": truncated2}
    ]
})
# New total: ~8500 tokens (under 20800 soft budget)

# 8. Enforce budget (no compression needed)
messages = compress_history(messages, keep_turns=6)
# → No compression (under soft budget)
# → Kept as-is

# 9. Agent loop iteration 2
response = await anthropic_completion(messages)
# LLM response: Single tool call
# Tool: analyze_impact("src/core/graph/builder.py:CodeGraph.get_all_downstream_dependents")

# 10. Execute tool
result = analyze_impact(symbol_id)
# Returns detailed blast radius: ~3500 tokens

# 11. Truncate: Cap at 4000 tokens (fits!)
truncated_result = truncate_tool_result(result)
# → NOT truncated (fits within 4K cap for tier 2)

# 12. Append to messages
messages.append({...})
messages.append({...})
# New total: ~12000 tokens (still under 20800)

# 13. Agent loop iteration 3
response = await anthropic_completion(messages)
# LLM response: Final comprehensive answer with analysis
# (No tool calls)

# 14. Record usage
iteration1: input=3700, output=1200, tools_called=2
iteration2: input=8500, output=800, tools_called=1
iteration3: input=12000, output=650, tools_called=0
turn_usage.total_input = 3700 + 8500 + 12000 = 24200
turn_usage.total_output = 1200 + 800 + 650 = 2650
turn_usage.total_tokens = 26850

# 15. Return to user
return "The blast radius includes the following downstream dependents:
        [Full detailed analysis, 6 levels deep, with examples]"
```

**Tier 2 Characteristics:**
- ✅ Full system prompt
- ✅ Multiple parallel tool calls
- ✅ No truncation (results fit)
- ✅ Comprehensive analysis (6+ levels of dependents)
- ✅ Recent context fully preserved (keep_turns=6)
- ✅ ~27K total tokens (leaving room for output)

---

##### **TIER 3 Execution (Claude 3.5 Sonnet, 200K context)**

```python
# 1. Detect tier
tier = resolve_tier("claude-3-5-sonnet-20241022")
# → context_window=200000, soft_budget=120000, tool_cap=8000, max_out=4096

# 2. Build system prompt
system = SYSTEM_PROMPT  # Full 16 sections + expansion for tier 3
system_tokens = 4200  # Includes advanced reasoning guidelines

# 3. Build messages with aggressive index preflight
auto_index_context = inspect_index(query="auth", limit=200)  # Proactive!
# Returns: 200 indexed files + metadata
# tokens: ~5000

messages = [
    {"role": "system", "content": system},                    # 4200 tokens
    {"role": "user", "content": query}                       # 200 tokens
]
# Total: ~4400 tokens (vastly under 120000 soft budget)

# 4. Agent loop iteration 1
response = await anthropic_completion(messages)
# LLM response: COMPREHENSIVE tool sequence
# Tool 1: inspect_index()  ← Full project structure
# Tool 2: search_codebase("get_symbol_relations", limit=15)  ← Top 15 results
# Tool 3: search_codebase("blast radius analysis", limit=15)
# Tool 4: get_symbol_relations(symbol_id)  ← Direct graph traversal

# 5. Execute tools in sequence
result1 = inspect_index(limit=200)                      # 4000 tokens
result2 = search_codebase("...", limit=15)              # 6500 tokens
result3 = search_codebase("...", limit=15)              # 5200 tokens
result4 = get_symbol_relations(symbol_id)               # 2800 tokens
# Total: ~18500 tokens

# 6. Truncate: Cap at 8000 tokens per tool (tier 3 cap)
truncated1 = truncate_tool_result(result1)  # ~4000 (fits)
truncated2 = truncate_tool_result(result2)  # ~6500 (fits)
truncated3 = truncate

===================================================================


┌──────────────────────────────────────────────────────────────────┐
│                  OUTPUT CONSISTENCY LAYERS                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Layer 1: PROMPT ENGINEERING (Input Normalization)              │
│  └─ Tier-specific templates with explicit output format         │
│                                                                   │
│  Layer 2: STRUCTURAL CONSTRAINTS (Guided Generation)             │
│  └─ JSON schema enforcement, output format directives           │
│                                                                   │
│  Layer 3: POST-PROCESSING (Output Normalization)                 │
│  └─ Parse, validate, reformat outputs to standard structure     │
│                                                                   │
│  Layer 4: QUALITY VALIDATION (Consistency Checks)                │
│  └─ Verify citations, check format compliance, fix issues       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
