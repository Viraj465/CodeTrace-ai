;; ------------------------
;; Function Definitions
;; ------------------------

(function_declaration
  name: (identifier) @function.name) @function.definition

(variable_declarator
  name: (identifier) @function.name
  value: (arrow_function)) @function.arrow

(method_definition
  name: (property_identifier) @function.name) @function.method


;; ------------------------
;; Class Definitions
;; ------------------------

(class_declaration
  name: (identifier) @class.name) @class.definition

(class
  name: (identifier) @class.name) @class.expression


;; --- Call Sites ---

; Standard function call
(call_expression
  function: (identifier) @function.call)
; Method call (e.g., console.log)
(call_expression
  function: (member_expression
    property: (property_identifier) @function.method.call))
