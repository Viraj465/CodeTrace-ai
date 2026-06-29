;; ------------------------
;; Function Definitions
;; ------------------------

(function_declaration
  name: (identifier) @function.name) @function.definition

(generator_function_declaration
  name: (identifier) @function.name) @function.definition

(variable_declarator
  name: (identifier) @function.name
  value: (arrow_function)) @function.arrow

(variable_declarator
  name: (identifier) @function.name
  value: (function_expression)) @function.expression

(method_definition
  name: (property_identifier) @function.name) @function.method


;; ------------------------
;; Class Definitions
;; ------------------------

(class_declaration
  name: (type_identifier) @class.name) @class.definition


;; ------------------------
;; TypeScript Type Shapes
;; ------------------------

(interface_declaration
  name: (type_identifier) @interface.name) @interface.definition

(type_alias_declaration
  name: (type_identifier) @type.name) @type.definition

(enum_declaration
  name: (identifier) @enum.name) @enum.definition


;; ------------------------
;; Call Sites
;; ------------------------

(call_expression
  function: (identifier) @function.call)

(call_expression
  function: (member_expression
    property: (property_identifier) @function.method.call))

(call_expression
  function: (member_expression
    property: (private_property_identifier) @function.method.call))