"""
These helpers provide language-agnostic traversal over Tree-sitter ASTs
to resolve semantic ownership, such as:

Class -> owns -> Method
Function -> calls -> Function

This works across different languages by deriving structure node types from queries/*.scm.
"""

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional



def node_text(node, source_code: bytes) -> str:
    """
    Extract the text of a node from the source code.
    """
    return source_code[node.start_byte:node.end_byte].decode("utf-8")


def get_identifier_from_children(node, source_code: bytes) -> Optional[str]:
    """
    Many languages store names as child identifiers.
    This function extracts the first meaningful identifier found.
    """
    identifier_types = {
        "identifier",
        "type_identifier",
        "field_identifier",
        "property_identifier",
        "name",
    }

    for child in node.children:
        if child.type in identifier_types:
            return node_text(child, source_code)

    return None

def find_parent_of_type(node, target_types: Iterable[str]):
    """
    Traverse upward until a parent matching one of the target types is found.
    """
    parent = node.parent

    while parent is not None:
        if parent.type in target_types:
            return parent
        parent = parent.parent

    return None


def extract_name_from_definition(node, source_bytes: bytes) -> Optional[str]:
    """
    Extract the identifier name from a function or class node.
    """
    for child in node.children:
        if "identifier" in child.type:
            return node_text(child, source_bytes)
    return None


def climb_to_root(node):
    """
    Traverse the AST to reach the root node. Useful for debugging.
    """
    while node.parent is not None:
        node = node.parent
    return node


CLASS_NODE_TYPES = {
    "class_declaration",
    "type_declaration",
    "struct_specifier",
}

FUNCTION_NODE_TYPES = {
    "function_declaration",
    "function_definition",
    "method_definition",
    "method_declaration",
}

QUERY_DIR = Path(__file__).parent / "queries"

# Values that make a JS/TS `variable_declarator` a function definition.
_FUNCTION_VALUE_TYPES = {
    "arrow_function",
    "function_expression",
    "function",
    "generator_function",
}


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _extract_node_type_for_capture(query_text: str, capture_index: int) -> Optional[str]:
    """
    For a capture like:
        ... ) @function.definition
    This walks backward to find the captured S-expression's root node type.
    """
    i = capture_index - 1
    while i >= 0 and query_text[i].isspace():
        i -= 1

    if i < 0:
        return None

    if query_text[i] != ")":
        while i >= 0 and _is_word_char(query_text[i]):
            i -= 1
        start = i + 1
        end = start
        while end < len(query_text) and _is_word_char(query_text[end]):
            end += 1
        return query_text[start:end] if end > start else None

    depth = 1
    i -= 1
    while i >= 0:
        ch = query_text[i]
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
            if depth == 0:
                j = i + 1
                while j < len(query_text) and query_text[j].isspace():
                    j += 1
                k = j
                while k < len(query_text) and _is_word_char(query_text[k]):
                    k += 1
                return query_text[j:k] if k > j else None
        i -= 1

    return None


def _capture_kind(capture_name: str) -> Optional[str]:
    if capture_name.endswith(".name"):
        return None
    if capture_name.startswith("class."):
        return "class"
    if capture_name.startswith("function."):
        return "function"
    if capture_name.startswith("symbol."):
        return "symbol"
    return None


def _infer_symbol_kind_from_node_type(node_type: str) -> str:
    class_hints = ("class", "struct", "interface", "enum", "type")
    lowered = node_type.lower()
    if any(hint in lowered for hint in class_hints):
        return "class"
    return "function"


def _extract_types_from_query_text(query_text: str) -> tuple[set[str], set[str]]:
    class_types: set[str] = set()
    function_types: set[str] = set()

    idx = 0
    while idx < len(query_text):
        cap_idx = query_text.find("@", idx)
        if cap_idx == -1:
            break

        j = cap_idx + 1
        while j < len(query_text) and (query_text[j].isalnum() or query_text[j] in "._"):
            j += 1
        capture_name = query_text[cap_idx + 1:j]
        idx = j

        kind = _capture_kind(capture_name)
        if kind is None:
            continue

        node_type = _extract_node_type_for_capture(query_text, cap_idx)
        if not node_type:
            continue

        if kind == "class":
            class_types.add(node_type)
        elif kind == "function":
            function_types.add(node_type)
        else:
            inferred_kind = _infer_symbol_kind_from_node_type(node_type)
            if inferred_kind == "class":
                class_types.add(node_type)
            else:
                function_types.add(node_type)

    return class_types, function_types


@lru_cache(maxsize=1)
def language_structure_map(query_dir: Optional[str] = None) -> dict[str, dict[str, set[str]]]:
    """
    Build a mapping from language to its class and function node types
    by reading the queries/*.scm files.
    """
    base_dir = Path(query_dir) if query_dir else QUERY_DIR
    structure: dict[str, dict[str, set[str]]] = {}

    if not base_dir.exists():
        return structure

    for query_file in base_dir.glob("*.scm"):
        language_name = query_file.stem
        class_types, function_types = _extract_types_from_query_text(
            query_file.read_text(encoding="utf-8")
        )

        if not class_types:
            class_types = set(CLASS_NODE_TYPES)
        if not function_types:
            function_types = set(FUNCTION_NODE_TYPES)

        structure[language_name] = {
            "class": class_types,
            "function": function_types,
        }

    return structure


def get_structure_node_types(language_name: Optional[str] = None) -> tuple[set[str], set[str]]:
    """
    Return the class and function node types for a given language.
    If no language is specified, returns the union of types across all query files.
    """
    structure = language_structure_map()
    if not structure:
        return set(CLASS_NODE_TYPES), set(FUNCTION_NODE_TYPES)

    if language_name:
        entry = structure.get(language_name)
        if entry:
            return set(entry["class"]), set(entry["function"])
        return set(CLASS_NODE_TYPES), set(FUNCTION_NODE_TYPES)

    class_union: set[str] = set()
    function_union: set[str] = set()
    for entry in structure.values():
        class_union.update(entry["class"])
        function_union.update(entry["function"])
    return class_union, function_union


def resolve_enclosing_class(
    node,
    source_code: bytes,
    language_name: Optional[str] = None,
    class_node_types: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """
    Determine if this function or method is defined inside a class or struct.
    """
    if class_node_types is None:
        class_node_types, _ = get_structure_node_types(language_name)

    class_node = find_parent_of_type(node, class_node_types)
    if not class_node:
        return None

    return get_identifier_from_children(class_node, source_code)


def resolve_function_name(node, source_code: bytes) -> Optional[str]:
    """
    Extract the function or method name regardless of the specific language syntax.
    """
    return get_identifier_from_children(node, source_code)


def build_qualified_name(
    node,
    source_code: bytes,
    language_name: Optional[str] = None,
    class_node_types: Optional[Iterable[str]] = None,
) -> str:
    """
    Build a fully qualified name, for example:
        function -> login
        method   -> AuthService.login
    """
    func_name = resolve_function_name(node, source_code)
    if not func_name:
        return "<anonymous>"
    owner = resolve_enclosing_class(
        node,
        source_code,
        language_name=language_name,
        class_node_types=class_node_types,
    )

    if owner:
        return f"{owner}.{func_name}"

    return func_name


def resolve_enclosing_function(
    node,
    source_code: bytes,
    language_name: Optional[str] = None,
    function_node_types: Optional[Iterable[str]] = None,
    class_node_types: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """
    Determine which function or method contains a specific call-site.
    """
    if function_node_types is None:
        _, function_node_types = get_structure_node_types(language_name)

    fn_node = find_parent_of_type(node, function_node_types)
    # JS/TS queries register `variable_declarator` as a function node so that
    # `const f = () => …` is a symbol. But most declarators hold plain values:
    # the call in `const x = compute()` belongs to the enclosing function, not
    # to a phantom caller named "x". Keep climbing past those.
    while fn_node is not None and fn_node.type == "variable_declarator":
        value = fn_node.child_by_field_name("value")
        if value is not None and value.type in _FUNCTION_VALUE_TYPES:
            break
        fn_node = find_parent_of_type(fn_node, function_node_types)
    if not fn_node:
        return None

    return build_qualified_name(
        fn_node,
        source_code,
        language_name=language_name,
        class_node_types=class_node_types,
    )

def debug_path_to_root(node):
    """
    Return the AST type chain from the node to the root.
    """
    types = []
    current = node

    while current is not None:
        types.append(current.type)
        current = current.parent

    return " -> ".join(types)