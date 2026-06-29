;; ------------------------
;; Class / Type Definitions
;; ------------------------

(class_declaration
  name: (identifier) @class.name) @class.definition

(interface_declaration
  name: (identifier) @class.name) @class.interface

(enum_declaration
  name: (identifier) @class.name) @class.enum

(struct_declaration
  name: (identifier) @class.name) @class.struct

(record_declaration
  name: (identifier) @class.name) @class.record

;; ------------------------
;; Method / Function Definitions
;; ------------------------

(method_declaration
  name: (identifier) @function.name) @function.definition

(constructor_declaration
  name: (identifier) @function.name) @function.constructor

(local_function_statement
  name: (identifier) @function.name) @function.local

;; --- Call Sites ---

; Simple method call: Foo()
(invocation_expression
  function: (identifier) @call.name)

; Member access call: obj.Method()
(invocation_expression
  function: (member_access_expression
    name: (identifier) @call.name))
