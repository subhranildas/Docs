## ADR (Address of a Label)

- **Purpose**: Load a **PC-relative address** into a register.
- **Size limitation**: Can reach labels within ±4 KB of the current program counter (PC).
- **Syntax**:

```armasm
ADR <Rd>, <label>
<Rd> → Destination register.
```

<label> → Symbol or memory location whose address you want.

### Example:

```armasm
AREA .data, DATA
my_var:
DCD 0x12345678

    AREA .text, CODE
    ENTRY

start:
ADR R0, my_var ; Load the address of my_var into R0
LDR R1, [R0] ; Load the value at my_var into R1
ADR is efficient for nearby labels because it uses a single instruction with PC-relative offset.
```

## ADRL (Address of a Label, Large Offset)

**Purpose**: Load the address of a distant label beyond ADR’s ±4 KB range.
**Implementation**: Assembler generates multiple instructions to construct the full 32-bit address.

### Example:

```armasm
AREA .data, DATA
my_big_array:
DCD 0x1, 0x2, 0x3, 0x4

    AREA .text, CODE
    ENTRY

start:
ADRL R0, my_big_array ; Load address of my_big_array into R0
LDR R1, [R0] ; Load first word
```

ADRL automatically handles the large offset and may expand into MOVW/MOVT pairs under the hood.

## Key Differences Between ADR and ADRL

### Instruction Range Usage

- ADR ±4 KB from PC Use for nearby labels
- ADRL Full 32-bit address Use for distant labels

## Combined Example

```armasm
AREA .data, DATA
small_var:
DCD 0x11111111
large_var:
DCD 0x22222222

AREA .text, CODE
ENTRY
start:
ADR R0, small_var ; Load address of small_var
LDR R1, [R0] ; Access value

    ADRL R2, large_var     ; Load address of large_var
    LDR R3, [R2]           ; Access value

    B .                    ; Infinite loop (end of program)
```

## Conclusion

- ADR → Quick, single-instruction address load for nearby memory.
- ADRL → Handles distant labels automatically.

Both instructions are essential for position-independent code and efficient memory access in ARM and Thumb-2 assembly.
