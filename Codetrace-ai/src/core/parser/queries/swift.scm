;; ------------------------
;; Class / Type Definitions
;; ------------------------

(class_declaration
  name: (type_identifier) @class.name) @class.definition

(protocol_declaration
  name: (type_identifier) @class.name) @class.interface

(enum_declaration
  name: (type_identifier) @class.name) @class.enum

(struct_declaration
  name: (type_identifier) @class.name) @class.struct

(extension_declaration
  name: (type_identifier) @class.name) @class.extension

;; ------------------------
;; Function / Method Definitions
;; ------------------------

(function_declaration
  name: (simple_identifier) @function.name) @function.definition

(init_declaration) @function.constructor

(deinit_declaration) @function.destructor

;; --- Call Sites ---

; Simple function call: foo()
(call_expression
  function: (simple_identifier) @call.name)

; Member method call: obj.foo()
(call_expression
  function: (navigation_expression
    suffix: (navigation_suffix
      (simple_identifier) @call.name)))
