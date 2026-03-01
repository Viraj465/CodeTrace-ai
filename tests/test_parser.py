import pytest
from src.core.parser.parser import CodeParser

def test_extract_symbols_python():
    parser = CodeParser(".py")
    code = "def my_func(): pass"
    
    symbols = parser.extract_symbols(code)
    
    assert len(symbols) == 1
    assert symbols[0]["name"] == "my_func"
    assert symbols[0]["type"] == "function"

def test_empty_code():
    parser = CodeParser(".py")
    assert parser.extract_symbols("") == []