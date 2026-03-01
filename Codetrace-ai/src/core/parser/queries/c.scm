;; ------------------------
;; Function Definitions
;; ------------------------

(function_definition
  declarator: (function_declarator
    declarator: (identifier) @function.name)) @function.definition


;; ------------------------
;; Struct Specifiers (C "Class")
;; ------------------------

(struct_specifier
  name: (type_identifier) @class.name) @class.definition

(type_definition
  declarator: (type_identifier) @class.typedef)

;; --- Call Sites ---

(call_expression
  function: (identifier) @call.name)
