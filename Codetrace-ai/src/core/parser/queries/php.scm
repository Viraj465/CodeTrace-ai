
;; ------------------------
;; Function Definitions
;; ------------------------

(function_definition
  name: (name) @function.name) @function.definition

(method_declaration
  name: (name) @function.method) @function.method

;; ------------------------
;; Class Definitions
;; ------------------------

(class_declaration
  name: (name) @class.name) @class.definition

(interface_declaration
  name: (name) @class.interface) @class.interface

(trait_declaration
  name: (name) @class.name) @class.definition

;; --- Call Sites ---

(function_call_expression
  function: (name) @call.name)

(member_call_expression
  name: (name) @call.name)