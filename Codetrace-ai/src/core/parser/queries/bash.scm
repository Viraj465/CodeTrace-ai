;; ------------------------
;; Function Definitions
;; ------------------------

; Standard function: foo() { ... }
(function_definition
  name: (word) @function.name) @function.definition

;; --- Call Sites ---

; Direct command call: git, npm, python, etc.
(command
  name: (command_name
    (word) @call.name))

; Function calls within script: my_func arg1 arg2
(command
  name: (command_name
    (word) @call.name))
