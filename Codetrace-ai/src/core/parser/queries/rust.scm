
;; ------------------------
;; Function Definitions
;; ------------------------

(function_item
  name: (identifier) @function.name) @function.definition

;; ------------------------
;; Struct / Enum / Trait Definitions
;; ------------------------

(struct_item
  name: (type_identifier) @class.name) @class.definition

(enum_item
  name: (type_identifier) @class.name) @class.definition

(trait_item
  name: (type_identifier) @class.interface) @class.interface

;; --- Call Sites ---

(call_expression
  function: (identifier) @call.name)

(call_expression
  function: (field_expression
    field: (field_identifier) @call.name))