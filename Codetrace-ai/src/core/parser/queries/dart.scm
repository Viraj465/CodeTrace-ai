
;; ------------------------
;; Function Definitions
;; ------------------------

(function_signature
  name: (identifier) @function.name) @function.definition

(method_declaration
  name: (identifier) @function.method) @function.method

;; ------------------------
;; Class Definitions
;; ------------------------

(class_definition
  name: (identifier) @class.name) @class.definition

(enum_declaration
  name: (identifier) @class.enum) @class.enum

;; --- Call Sites ---

(method_invocation
  name: (identifier) @call.name)