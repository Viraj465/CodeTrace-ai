(function_definition
  name: (identifier) @symbol.name) @symbol.definition

(class_definition
  name: (identifier) @symbol.name) @symbol.definition

;; --- Call Sites ---
(call
  function: (identifier) @call.name) @call.expression

(call
  function: (attribute
    attribute: (identifier) @call.name)) @call.
    