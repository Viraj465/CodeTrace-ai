;; ------------------------
;; Class / Object Definitions
;; ------------------------

(class_declaration
  name: (type_identifier) @class.name) @class.definition

(interface_declaration
  name: (type_identifier) @class.name) @class.interface

(enum_entry
  name: (type_identifier) @class.name) @class.enum

(object_declaration
  name: (type_identifier) @class.name) @class.object

(companion_object
  name: (type_identifier) @class.name) @class.companion

;; ------------------------
;; Function Definitions
;; ------------------------

(function_declaration
  name: (simple_identifier) @function.name) @function.definition

(anonymous_initializer) @function.init

;; --- Call Sites ---

; Direct call: foo()
(call_expression
  (call_suffix)
  . (simple_identifier) @call.name)

; Navigation call: obj.foo()
(navigation_expression
  (navigation_suffix
    (simple_identifier) @call.name))
