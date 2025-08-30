## What Are Assembler Directives?

- **Instructions** → Tell the CPU what to do (e.g., `ADD R0, R1, R2`).
- **Directives** → Tell the **assembler** how to build the program (e.g., where to put data, how to align memory, define constants).

Directives are **not executed** at runtime but are crucial for program correctness.

## Common Assembler Directives in ARM Assembly

### Data Definition Directives

Used to **reserve memory** and **initialize variables**.

| Directive | Meaning                      | Example            |
| --------- | ---------------------------- | ------------------ |
| `.byte`   | Allocate 8-bit data          | `.byte 0x12, 0x34` |
| `.hword`  | Allocate 16-bit data         | `.hword 0x1234`    |
| `.word`   | Allocate 32-bit data         | `.word 0x12345678` |
| `.asciz`  | Null-terminated ASCII string | `.asciz "Hello"`   |

### Section Control Directives

Tell the assembler which section (code/data) the following instructions or data belong to.

| Directive | Purpose                           | Example |
| --------- | --------------------------------- | ------- |
| `.text`   | Start code section (instructions) | `.text` |
| `.data`   | Start data section (variables)    | `.data` |
| `.bss`    | Uninitialized data                | `.bss`  |

### Alignment and Organization

Ensure proper **memory alignment** for efficiency.

| Directive   | Purpose                         | Example                       |
| ----------- | ------------------------------- | ----------------------------- |
| `.align n`  | Align next data to 2^n boundary | `.align 2` (align to 4 bytes) |
| `.org addr` | Set location counter to `addr`  | `.org 0x20000000`             |

### Symbol and Constant Definition

Create symbolic names for values or addresses.

| Directive | Purpose                         | Example                  |
| --------- | ------------------------------- | ------------------------ |
| `.equ`    | Define a constant               | `.equ STACK_SIZE, 0x100` |
| `.set`    | Same as `.equ` (redefinable)    | `.set BUFFER_SIZE, 256`  |
| `.global` | Make a symbol visible to linker | `.global Reset_Handler`  |
| `.extern` | Declare an external symbol      | `.extern main`           |

### Macro and Include

Enable **reusability** and **code modularity**.

| Directive          | Purpose              | Example                                            |
| ------------------ | -------------------- | -------------------------------------------------- |
| `.macro` / `.endm` | Define a macro       | `.macro PUSH reg \n STR \reg, [SP, #-4]! \n .endm` |
| `.include`         | Include another file | `.include "startup.s"`                             |

## Example: Using Directives in Cortex-M Startup Code

```armasm
    .syntax unified
    .cpu cortex-m4
    .thumb

    .section .isr_vector, "a", %progbits
    .word   _estack              ; Initial stack pointer
    .word   Reset_Handler        ; Reset handler

    .text
    .global Reset_Handler
Reset_Handler:
    LDR R0, =_start_data
    BL main
```

Here:

- .section → places vector table in flash.
- .word → stores initial stack pointer and reset vector.
- .global → exposes Reset_Handler to the linker.
- .text → marks the code section.

## Why Directives Matter

- Memory control: Place code and data precisely.
- Portability: Provide symbolic names instead of magic numbers.
- Organization: Separate code, initialized data, and uninitialized data.
- Linking: Export/import functions across files.
- Without directives, large assembly programs would be messy and unmanageable.

## Conclusion

Assembler directives act as the glue between your source code and the machine code layout in memory. They don’t execute at runtime but define how the program is structured. Mastery of directives is crucial when writing startup files, linker scripts, and bare-metal applications for ARM Cortex-M micro-controllers.
