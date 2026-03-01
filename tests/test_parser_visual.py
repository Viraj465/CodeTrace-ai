# from src.core.parser.parser import CodeParser
from src.core.parser.parser import CodeParser
from rich.console import Console
from pathlib import Path
from rich.table import Table

console = Console()

def test_visual():
    parser = CodeParser(".py")
    sample_code = """
class DatabaseManager:
    def connect(self):
        pass

def calculate_metrics(data):
    return len(data)
"""
    symbols = parser.extract_symbols(sample_code)
    
    table = Table(title="Extracted Symbols")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Line", style="green")

    for s in symbols:
        table.add_row(s["type"], s["name"], str(s["start_line"]))

    console.print(table)

def test_cpp_parsing():
    # Ensure you use the correct extension for the map
    parser = CodeParser(".cpp") 
    
    # Load the complex file
    file_path = Path(r"E:/CLI/Codetrace-ai/tests/legacy.cpp")
    if not file_path.exists():
        console.print("[red]Error: Create tests/samples/complex_legacy.cpp first![/red]")
        return
        
    code = file_path.read_text()
    symbols = parser.extract_symbols(code)
    
    table = Table(title=f"Extracted Symbols from {file_path.name}")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Line", style="green")

    for s in symbols:
        table.add_row(s["type"], s["name"], str(s["start_line"]))

    console.print(table)

if __name__ == "__main__":
    test_cpp_parsing()
    test_visual()