(function_definition
  declarator: (function_declarator
    declarator: (identifier) @symbol.name)) @symbol.definition

(class_specifier
  name: (type_identifier) @symbol.name) @symbol.definition

(struct_specifier
  name: (type_identifier) @symbol.name) @symbol.definition

;; --- Call Sites ---

(call_expression
  function: (field_expression
    field: (field_identifier) @call.name))

(call_expression
  function: (qualified_identifier
    name: (identifier) @call.name))
