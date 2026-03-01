;; ------------------------
;; Function Definitions
;; ------------------------

(function_declaration
  name: (identifier) @function.name) @function.definition

(method_declaration
  name: (field_identifier) @function.name) @function.method


;; ------------------------
;; Struct / Type (Go has no classes)
;; ------------------------

(type_declaration
  (type_spec
    name: (type_identifier) @class.name
    type: (struct_type))) @class.definition

(interface_type) @class.interface

;; --- Call Sites ---
(call_expression
  function: (identifier) @call.name)
(call_expression
  function: (selector_expression
    field: (field_identifier) @call.name))
