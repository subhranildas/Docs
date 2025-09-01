## What Are Load/Store Instructions?

- **Load instructions** → Move data **from memory to a register**.
  Example: `LDR R0, [R1]` → Load value at memory address in `R1` into `R0`.

- **Store instructions** → Move data **from a register to memory**.
  Example: `STR R0, [R1]` → Store contents of `R0` into memory at address in `R1`.

ARM processors use a **Load/Store architecture**, meaning memory is only accessed explicitly through these instructions, not as part of arithmetic operations.

## Common Load Instructions

| Instruction     | Purpose                                  | Example               |
| --------------- | ---------------------------------------- | --------------------- |
| `LDR`           | Load 32-bit word from memory             | `LDR R0, [R1]`        |
| `LDRB`          | Load 8-bit unsigned byte                 | `LDRB R0, [R1]`       |
| `LDRH`          | Load 16-bit unsigned halfword            | `LDRH R0, [R1]`       |
| `LDRSB`         | Load 8-bit signed byte (sign-extended)   | `LDRSB R0, [R1]`      |
| `LDRSH`         | Load 16-bit signed halfword              | `LDRSH R0, [R1]`      |
| `LDRD`          | Load two registers (double word, 64-bit) | `LDRD R0, R1, [R2]`   |
| `LDR (literal)` | Load from constant pool (PC-relative)    | `LDR R0, =0x20000000` |

## Common Store Instructions

| Instruction | Purpose                                   | Example             |
| ----------- | ----------------------------------------- | ------------------- |
| `STR`       | Store 32-bit word to memory               | `STR R0, [R1]`      |
| `STRB`      | Store 8-bit byte                          | `STRB R0, [R1]`     |
| `STRH`      | Store 16-bit halfword                     | `STRH R0, [R1]`     |
| `STRD`      | Store two registers (double word, 64-bit) | `STRD R0, R1, [R2]` |

## Addressing Modes in Thumb-2

Load/Store instructions in Thumb-2 support several **addressing modes**:

### Register Offset

Use base register + offset register.

```armasm
LDR R0, [R1, R2]   ; R0 = *(R1 + R2)
```

### Immediate Offset

Use base register + constant offset.

```armasm
STR R0, [R1, #4]   ; *(R1+4) = R0
```

### Pre-indexed (with optional write-back)

Address is calculated before memory access.

```armasm
LDR R0, [R1, #8]! ; R1 = R1+8, then load from [R1] 4. Post-indexed
```

Address is calculated after memory access.

```armasm
STR R0, [R1], #4 ; Store R0, then R1 = R1+4 5. Literal Pool (PC-relative)
```

Load constants directly from flash/ROM.

```armasm
LDR R0, =0x12345678
```

### Load/Store with Shifted Register (e.g., LSL):

```armasm
    LDR R0, [R1, R2, LSL #2]   ; Load word from (R1 + R2*4)
    STR R3, [R4, R5, LSL #1]   ; Store R3 at (R4 + R5*2)
```

LSL #2 → multiply offset by 4 (useful for 32-bit arrays).
LSL #1 → multiply offset by 2 (useful for 16-bit arrays).

### Special Load Instructions: LDRSB and LDRSH:

```armasm
    LDRSB R0, [R1]       ; Load signed 8-bit value into R0 (extends to 32-bit)
    LDRSH R2, [R3, #2]   ; Load signed 16-bit value into R2
```

Example: Array Access with Load/Store
; Suppose R1 = base address of array
; R2 = index

    LDR R0, [R1, R2, LSL #2]   ; Load array[R2] (32-bit word)
    LDRSB R3, [R1, R2]         ; Load signed byte from array
    LDRSH R4, [R1, R2, LSL #1] ; Load signed halfword from array

### Example: Stack Operations Using Load/Store

```armasm
PUSH {R0, R1, LR} ; Store multiple registers on stack
POP {R0, R1, PC} ; Load multiple registers from stack
```

Here, PUSH and POP are pseudo-instructions that internally use STMFD (Store Multiple, Full Descending) and LDMFD (Load Multiple, Full Descending).

Example: Array Access with Load/Store

```armasm
LDR R1, =array ; Load base address of array
MOV R2, #0 ; Index = 0
LDR R0, [R1, R2, LSL #2] ; Load array[0] (word = 4 bytes)
ADD R2, R2, #1 ; Index++
STR R0, [R1, R2, LSL #2] ; Store R0 into array[1]
```

## Why Load/Store Instructions Matter

- Efficient Memory Access: Control over byte, halfword, and word granularity.
- Flexibility: Support for multiple addressing modes.
- Scalability: Enable array indexing, structure access, and stack management.
- Foundation: All higher-level constructs (C variables, arrays, structs) eventually reduce to load/store operations.

## Conclusion

In ARM Thumb-2 assembly, load/store instructions are the backbone of all data movement between registers and memory.
