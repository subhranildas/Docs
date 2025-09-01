## Data Definition Instructions in ARM/Thumb-2 Assembly

When programming in ARM or Thumb-2 assembly, **data definition instructions** are used to reserve memory and initialize variables. These instructions are **processed by the assembler** and are not executed at runtime, but they are essential for structuring your program's memory.

## Common Data Definition Instructions

| Instruction | Size   | Description                               | Example                      |
| ----------- | ------ | ----------------------------------------- | ---------------------------- |
| `.byte`     | 8-bit  | Allocate 1 byte per value                 | `.byte 0x12, 0x34`           |
| `.hword`    | 16-bit | Allocate 2 bytes per value                | `.hword 0x1234`              |
| `.word`     | 32-bit | Allocate 4 bytes per value                | `.word 0x12345678`           |
| `DCB`       | 8-bit  | Define constant bytes (1 byte per value)  | `DCB 0x41, 0x42, 0x43`       |
| `DCD`       | 32-bit | Define constant words (4 bytes per value) | `DCD 0x1000, 0x2000, 0x3000` |

## Using DCB and DCD Instructions

### DCB (Define Constant Byte)

- Allocates **1 byte per value** in memory.
- Ideal for **lookup tables, strings, or small flags**.

**Example:**

```armasm
AREA .data, DATA

my_bytes:
    DCB 0x41, 0x42, 0x43   ; ASCII 'A', 'B', 'C'

; Accessing the bytes
LDR R0, =my_bytes          ; Load address of the byte array
LDRB R1, [R0]              ; Load first byte (0x41)
```

LDRB loads a single byte from memory, which matches the DCB size.
DCD (Define Constant Doubleword / Word) Allocates 4 bytes per value.
Commonly used for pointers, lookup tables, and 32-bit constants.

Example:

```armasm
AREA .data, DATA

my_words:
DCD 0x1000, 0x2000, 0x3000 ; 32-bit constants

; Accessing the words
LDR R2, =my_words ; Load address of the word array
LDR R3, [R2, #4] ; Load second word (0x2000)
LDR loads a full 32-bit word, compatible with DCD.
```

Key Points
Memory allocation: DCB and DCD reserve memory and initialize it with constant values.
Access in code: Use LDRB for bytes and LDR for words.
Readability: Makes lookup tables, strings, and constants explicit and maintainable.
Efficiency: Constants are embedded in memory during assembly, reducing runtime setup.

Example:

```armasm
AREA .data, DATA

letters:
DCB 0x41, 0x42, 0x43 ; ASCII 'A', 'B', 'C'

numbers:
DCD 0x100, 0x200, 0x300 ; 32-bit constants

AREA .text, CODE
ENTRY

start:
LDR R0, =letters
LDRB R1, [R0] ; Load 'A'

    LDR R2, =numbers
    LDR R3, [R2, #4]        ; Load 0x200

    ; End program (in a real embedded system, you might loop or branch)

```

## Conclusion

- DCB and DCD are powerful instructions for defining constant data in memory.
- Use them to organize your program's data section efficiently.
- They work seamlessly with load/store instructions, making memory access simple and consistent.
