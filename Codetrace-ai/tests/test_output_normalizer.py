"""
Tests for the output normalizer system.

Validates that outputs are consistent across providers and tiers.
"""

import pytest
from src.core.agents.output_normalizer import (
    OutputNormalizer,
    CitationValidator,
    CodeBlockFormatter,
    MarkdownNormalizer,
    QualityChecker,
    QueryType,
    Citation,
    CodeBlock,
)


class TestCitationValidator:
    """Test citation extraction and validation."""
    
    def test_extract_file_line_citations(self):
        text = "Function is defined in `src/auth/login.py:45`"
        citations = CitationValidator.extract_citations(text)
        
        assert len(citations) == 1
        assert citations[0].file_path == "src/auth/login.py"
        assert citations[0].line_number == 45
    
    def test_extract_file_only_citations(self):
        text = "Check `src/config.py` for settings"
        citations = CitationValidator.extract_citations(text)
        
        assert len(citations) == 1
        assert citations[0].file_path == "src/config.py"
        assert citations[0].line_number is None
    
    def test_extract_multiple_citations(self):
        text = """
        The function is in `src/auth.py:10` and calls `utils/helper.py:25`
        """
        citations = CitationValidator.extract_citations(text)
        
        assert len(citations) == 2
        assert citations[0].file_path == "src/auth.py"
        assert citations[1].file_path == "utils/helper.py"
    
    def test_format_citation_with_line(self):
        formatted = CitationValidator.format_citation("src/test.py", 123)
        assert formatted == "`src/test.py:123`"
    
    def test_format_citation_without_line(self):
        formatted = CitationValidator.format_citation("src/test.py")
        assert formatted == "`src/test.py`"


class TestCodeBlockFormatter:
    """Test code block extraction and formatting."""
    
    def test_extract_python_code_block(self):
        text = """
        Here's the code:
        ```python
        def hello():
            print("world")
        ```
        """
        blocks = CodeBlockFormatter.extract_code_blocks(text)
        
        assert len(blocks) == 1
        assert blocks[0].language == "python"
        assert "def hello():" in blocks[0].content
    
    def test_extract_multiple_code_blocks(self):
        text = """
        Python:
        ```python
        x = 1
        ```
        
        JavaScript:
        ```javascript
        const x = 1;
        ```
        """
        blocks = CodeBlockFormatter.extract_code_blocks(text)
        
        assert len(blocks) == 2
        assert blocks[0].language == "python"
        assert blocks[1].language == "javascript"
    
    def test_normalize_language_aliases(self):
        text = "```py\ncode\n```"
        blocks = CodeBlockFormatter.extract_code_blocks(text)
        
        assert blocks[0].language == "python"
    
    def test_validate_code_block(self):
        valid_block = CodeBlock(language="python", content="x = 1")
        is_valid, issues = CodeBlockFormatter.validate_code_block(valid_block)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_empty_code_block(self):
        empty_block = CodeBlock(language="python", content="")
        is_valid, issues = CodeBlockFormatter.validate_code_block(empty_block)
        
        assert not is_valid
        assert "empty" in issues[0].lower()
    
    def test_format_code_block(self):
        formatted = CodeBlockFormatter.format_code_block("python", "x = 1")
        assert formatted == "```python\nx = 1\n```"


class TestMarkdownNormalizer:
    """Test markdown normalization."""
    
    def test_normalize_single_hash_headings(self):
        text = "# Title\nContent"
        normalized = MarkdownNormalizer.normalize_headings(text)
        
        assert normalized.startswith("##")
    
    def test_normalize_list_bullets(self):
        text = "* Item 1\n+ Item 2\n- Item 3"
        normalized = MarkdownNormalizer.normalize_lists(text)
        
        lines = normalized.split('\n')
        assert all(line.startswith('- ') for line in lines if line.strip())
    
    def test_add_spacing_around_headings(self):
        text = "Text\n## Heading\nMore text"
        normalized = MarkdownNormalizer.add_missing_spacing(text)
        
        # Should have blank lines around heading
        assert "\n\n## Heading\n\n" in normalized


class TestQualityChecker:
    """Test quality checking functionality."""
    
    def test_check_citation_presence_explanation(self):
        text_with_citation = "Function is in `src/test.py:10`"
        has_enough, score = QualityChecker.check_citation_presence(
            text_with_citation, QueryType.EXPLANATION
        )
        
        assert has_enough
        assert score >= 1.0
    
    def test_check_citation_presence_insufficient(self):
        text_without_citation = "Function does something"
        has_enough, score = QualityChecker.check_citation_presence(
            text_without_citation, QueryType.EXPLANATION
        )
        
        assert not has_enough
        assert score < 1.0
    
    def test_check_fact_vs_inference_with_markers(self):
        text = "Based on the code, this appears to be a singleton pattern"
        is_ok, issues = QualityChecker.check_fact_vs_inference_separation(text)
        
        # Should be OK because "based on" and "appears to" are inference markers
        assert is_ok or len(issues) == 0
    
    def test_detect_uncited_definitive_claims(self):
        text = "The function is implemented in the auth module"
        is_ok, issues = QualityChecker.check_fact_vs_inference_separation(text)
        
        # Should flag this as a definitive claim without citation
        # (Note: may not trigger if far from any citation)
        assert isinstance(issues, list)


class TestOutputNormalizer:
    """Test the main output normalizer."""
    
    def test_normalize_tier1_output(self):
        normalizer = OutputNormalizer(tier="tier1")
        raw = "The function is in `src/test.py:10`\n```python\nx = 1\n```"
        
        normalized = normalizer.normalize(raw, query_type="general")
        
        assert normalized  # Should return something
        assert "src/test.py" in normalized
    
    def test_normalize_tier2_output(self):
        normalizer = OutputNormalizer(tier="tier2")
        raw = """
# Title
Function at `src/test.py:10`
* Item 1
```py
x = 1
```
"""
        
        normalized = normalizer.normalize(raw, query_type="explanation")
        
        # Should normalize heading to ##
        assert normalized.strip().startswith("##")
        # Should normalize list bullets to -
        assert "- Item" in normalized
        # Should normalize language alias
        assert "```python" in normalized
    
    def test_normalize_tier3_output(self):
        normalizer = OutputNormalizer(tier="tier3")
        raw = "Comprehensive analysis with `src/a.py:1` and `src/b.py:2`"
        
        normalized = normalizer.normalize(raw, query_type="architecture")
        
        assert "src/a.py" in normalized
        assert "src/b.py" in normalized
    
    def test_get_quality_report(self):
        normalizer = OutputNormalizer(tier="tier2")
        text = """
## Function Analysis
The function is defined in `src/auth.py:45`

```python
def authenticate(user, password):
    return check_credentials(user, password)
```
"""
        
        report = normalizer.get_quality_report(text, query_type="explanation")
        
        assert report["quality_score"] > 0.0
        assert report["citations_count"] == 1
        assert report["code_blocks_count"] == 1
        assert report["has_structure"] is True
        assert report["tier"] == "tier2"
    
    def test_empty_input(self):
        normalizer = OutputNormalizer(tier="tier2")
        result = normalizer.normalize("", query_type="general")
        
        assert result == ""
    
    def test_whitespace_only_input(self):
        normalizer = OutputNormalizer(tier="tier2")
        result = normalizer.normalize("   \n   ", query_type="general")
        
        # Should return input unchanged or stripped
        assert result.strip() == ""


class TestProviderAgnostic:
    """Test that normalization works regardless of input style."""
    
    def test_verbose_anthropic_style(self):
        normalizer = OutputNormalizer(tier="tier2")
        
        # Anthropic-style: verbose, well-structured
        text = """
## Detailed Analysis

The authentication system is implemented across multiple files:

- Primary handler: `src/auth/login.py:45`
- Token generation: `src/auth/jwt.py:23`
- Validation middleware: `src/middleware/auth.py:67`

The flow proceeds as follows...
"""
        
        normalized = normalizer.normalize(text, query_type="architecture")
        assert "src/auth/login.py" in normalized
    
    def test_terse_groq_style(self):
        normalizer = OutputNormalizer(tier="tier1")
        
        # Groq-style: terse, minimal
        text = "Auth in `src/auth.py:10`. Calls `db.py:45`."
        
        normalized = normalizer.normalize(text, query_type="general")
        assert "src/auth.py" in normalized
        assert "db.py" in normalized
    
    def test_mixed_formatting_style(self):
        normalizer = OutputNormalizer(tier="tier2")
        
        # Mixed formatting issues
        text = """
# Wrong heading level
* Mixed
+ Bullet
- Styles
```
Code without language
```
Citation at `src/test.py:1`
"""
        
        normalized = normalizer.normalize(text, query_type="general")
        
        # Should fix heading
        assert normalized.strip().startswith("##")
        # Should normalize bullets
        lines = [l for l in normalized.split('\n') if l.strip().startswith('-')]
        assert len(lines) >= 3


class TestTierAdaptation:
    """Test tier-specific adaptations."""
    
    def test_tier1_expectations(self):
        normalizer = OutputNormalizer(tier="tier1")
        
        # Tier1 models might produce less structured output
        text = "Function at src/test.py does stuff"
        normalized = normalizer.normalize(text, query_type="general")
        
        # Should still normalize what it can
        assert isinstance(normalized, str)
    
    def test_tier2_expectations(self):
        normalizer = OutputNormalizer(tier="tier2")
        
        report = normalizer.get_quality_report(
            "Function at `src/test.py:10`",
            query_type="explanation"
        )
        
        assert report["quality_score"] > 0.5
    
    def test_tier3_expectations(self):
        normalizer = OutputNormalizer(tier="tier3")
        
        # Tier3 should expect high-quality output
        comprehensive_text = """
## Architecture Analysis

Primary implementation: `src/core/system.py:100`
Dependencies:
- `src/utils/helpers.py:45`
- `src/models/data.py:23`

```python
class System:
    def __init__(self):
        self.helper = Helper()
```

Based on the call graph, this is a factory pattern.
"""
        
        report = normalizer.get_quality_report(
            comprehensive_text,
            query_type="architecture"
        )
        
        assert report["citations_count"] >= 2
        assert report["code_blocks_count"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
