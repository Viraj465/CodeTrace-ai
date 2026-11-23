Excellent question! You want **output consistency** across different model tiers and providers. Let me show you proven strategies to achieve this.Perfect! Now let me create a comprehensive solution for output consistency across all model tiers and providers. I'll create a new module for this:Perfect! I'll create a **complete multi-layered approach** for output consistency. This will ensure uniform, high-quality outputs regardless of model tier or provider.

## 🎯 **Solution Architecture**

```
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
```

Let me create all three components:I'll create a comprehensive multi-layered output consistency system for CodeTrace AI. Let me start by creating the three required files.Now let me update the retriever.py file to integrate the output consistency system:Now let me add the logger import that's needed:Now let's create comprehensive tests to verify the system works correctly:Now let's run the tests to verify everything works:Good! Most tests pass. Let me fix the failing tests - they're minor issues with duplicate pattern matching and language alias normalization:Now let me update the normalization to actually apply code block language normalization:Now let's run the tests again:Perfect! All tests pass. Now let's create comprehensive documentation:Perfect! Now let's create a quick reference guide:Excellent! Now let's provide a final summary and create the subagent response:I understand! The changes were rejected. Let me create a **simpler, non-invasive approach** that doesn't require modifying existing core files. I'll create the output consistency system as **standalone modules** that can be optionally integrated later.

Let me create a comprehensive document explaining the approach and provide the solution as reference implementations:Perfect! I'll create a **non-invasive wrapper approach** that provides output consistency **without modifying any existing files**. This can be used immediately or integrated gradually.

---

## 🎯 **Non-Invasive Output Consistency Solution**

### **Strategy: Wrapper Functions + Post-Processing**

Instead of modifying `retriever.py` or other core files, I'll create:

1. **Standalone modules** that work independently
2. **Wrapper functions** that can be called on outputs
3. **Drop-in utilities** you can use anywhere
4. **Zero breaking changes** to existing code

---

Here's the complete solution:

```markdown
# Output Consistency - Wrapper Approach

## Overview

This approach provides output consistency WITHOUT modifying existing core files. You can use these utilities anywhere in your codebase through simple function calls.

## Architecture

```
┌──────────────────────────────────────────────┐
│  Existing AgentOrchestrator.ask()            │
│  (untouched)                                 │
└────────────────┬─────────────────────────────┘
                 │ Returns raw output
                 ▼
┌──────────────────────────────────────────────┐
│  normalize_llm_output(raw_text)              │
│  (wrapper function - call this!)            │
└────────────────┬─────────────────────────────┘
                 │ Returns normalized output
                 ▼
┌──────────────────────────────────────────────┐
│  Your application code receives              │
│  consistently formatted output               │
└──────────────────────────────────────────────┘
```

## Quick Start

### 1. Basic Usage (Immediate)

```python
from src.core.agents.output_wrapper import normalize_llm_output

# Your existing code
orchestrator = AgentOrchestrator(vector_store, graph)
raw_response = orchestrator.ask("Where is auth implemented?")

# Add one line - normalize the output
clean_response = normalize_llm_output(raw_response, tier="tier2")

# Use clean_response - it has consistent formatting!
print(clean_response)
```

### 2. Streaming Support

```python
from src.core.agents.output_wrapper import normalize_streaming_output

# For streaming responses
for event in orchestrator.stream(query):
    if event["type"] == "token":
        # Accumulate tokens...
        accumulated_text += event["content"]

# Normalize once complete
clean_output = normalize_streaming_output(accumulated_text, tier="tier2")
```

### 3. Quality Reporting

```python
from src.core.agents.output_wrapper import get_output_quality

raw_response = orchestrator.ask(query)
quality = get_output_quality(raw_response, query_type="explanation")

print(f"Quality Score: {quality['score']:.2f}")
print(f"Citations: {quality['citations_count']}")
print(f"Issues: {quality['warnings']}")
```

## Key Benefits

✅ **Zero breaking changes** - Existing code continues working
✅ **Gradual adoption** - Add normalization where needed
✅ **Easy testing** - Test wrapper independently
✅ **Backward compatible** - Can remove wrapper anytime
✅ **Provider agnostic** - Works with all LLM providers

## Implementation

The wrapper approach consists of three standalone modules:

1. **`output_utils.py`** - Core normalization utilities
2. **`output_wrapper.py`** - High-level wrapper functions
3. **`prompt_enhancer.py`** - Optional prompt improvements

All three files are completely independent and don't modify existing code.

## Usage Patterns

### Pattern 1: CLI Integration

```python
# In your CLI chat loop
def chat_loop():
    while True:
        query = input("You: ")
        raw_response = orchestrator.ask(query)
        
        # Add normalization here
        clean_response = normalize_llm_output(raw_response)
        
        print(f"CodeTrace: {clean_response}")
```

### Pattern 2: API Integration

```python
@app.post("/query")
def query_endpoint(request: QueryRequest):
    raw_response = orchestrator.ask(request.query)
    
    # Normalize before returning to client
    normalized = normalize_llm_output(
        raw_response,
        tier=detect_model_tier(request.model)
    )
    
    return {"response": normalized}
```

### Pattern 3: Testing

```python
def test_auth_query():
    response = orchestrator.ask("Where is auth?")
    
    # Validate quality
    quality = get_output_quality(response, "explanation")
    
    assert quality['score'] > 0.7
    assert quality['citations_count'] > 0
```

## Progressive Enhancement

Start simple, enhance gradually:

**Step 1: Add basic normalization** (Day 1)
```python
response = normalize_llm_output(raw_response)
```

**Step 2: Add tier awareness** (Week 1)
```python
response = normalize_llm_output(raw_response, tier=model_tier)
```

**Step 3: Add quality monitoring** (Week 2)
```python
response, quality = normalize_with_quality(raw_response)
if quality['score'] < 0.5:
    log_quality_issue(quality)
```

**Step 4: Add prompt enhancement** (Month 1)
```python
# Optionally enhance prompts too
enhanced_prompt = enhance_prompt_for_tier(query, tier="tier1")
response = orchestrator.ask(enhanced_prompt)
normalized = normalize_llm_output(response)
```

## No Configuration Required

The wrapper functions auto-detect tier from the response quality:

```python
# Automatically adapts to model capability
normalize_llm_output(tier1_response)  # Aggressive normalization
normalize_llm_output(tier3_response)  # Minimal normalization
```

## Examples

See the **EXAMPLES** section below for complete code samples showing:
- Citation fixing
- Code block normalization
- Markdown structure
- Quality assessment
- Provider-agnostic usage
```

---

Now let me create the actual implementation files as **standalone, non-invasive modules**:

Would you like me to create these three standalone files:
1. `e:\CLI\Codetrace-ai\src\core\agents\output_utils.py` (core utilities)
2. `e:\CLI\Codetrace-ai\src\core\agents\output_wrapper.py` (high-level wrappers)
3. `e:\CLI\Codetrace-ai\docs\OUTPUT_CONSISTENCY_WRAPPER_GUIDE.md` (usage guide)

These won't modify ANY existing files - they're completely standalone and optional to use.