;; ------------------------
;; Class Definitions
;; ------------------------

(class_declaration
  name: (identifier) @class.name) @class.definition

(interface_declaration
  name: (identifier) @class.name) @class.interface

(enum_declaration
  name: (identifier) @class.name) @class.enum


;; ------------------------
;; Method / Function Definitions
;; ------------------------

(method_declaration
  name: (identifier) @function.name) @function.definition

(constructor_declaration
  name: (identifier) @function.name) @function.constructor

;; --- Call Sites ---

; Method invocation (e.g., obj.method())
(method_invocation
  name: (identifier) @function.method.call)
