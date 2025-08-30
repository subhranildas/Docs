## ARM Instruction Set

- **32-bit fixed-length instructions**
- Originally the **primary instruction set** of ARM CPUs.
- Provides powerful instructions with many addressing modes and features.

### Characteristics

- Instructions are always **4 bytes long**.
- Orthogonal design: almost any instruction can use any register.
- Rich set of instructions (e.g., load/store, arithmetic, logical, multiply, branch).

### Example (ARM mode)

```armasm
ADD R0, R1, R2   ; R0 = R1 + R2
LDR R3, [R4]     ; Load value from memory at address in R4 into R3
STR R5, [R6, #4] ; Store R5 into memory at address (R6 + 4)
```

### Pros & Cons

- Fast and flexible
- Powerful addressing modes
- Larger code size (not memory-efficient)
- Not supported in Cortex-M cores (they use only Thumb)

## Thumb Instruction Set

To improve code density (smaller binaries → less memory usage), ARM introduced the Thumb instruction set.

### Characteristics

- 16-bit fixed-length instructions (half the size of ARM mode).

- Subset of ARM instructions → fewer addressing modes and simpler operations.

- Focused on common operations to keep encoding compact.

### Example (Thumb mode)

```armasm
ADD R0, R1 ; Add R1 to R0
LDR R2, [R3] ; Load word from memory
```

### Pros & Cons

- Smaller code size (~65% smaller than ARM code)
- Ideal for low-memory microcontrollers
- Reduced instruction flexibility (subset of ARM)
- May need more instructions to do complex tasks

## Thumb-2 Instruction Set

Introduced with ARMv6T2 and ARMv7 architectures, Thumb-2 blends the best of ARM and Thumb:
Mix of 16-bit and 32-bit instructions
Provides the code density of Thumb with the power of ARM

### Characteristics

Backward-compatible with Thumb instructions.
Adds 32-bit instructions to handle more complex operations.
Used in all Cortex-M3, M4, and M7 processors.
Offers almost the full functionality of ARM instructions, but with better density.

### Example (Thumb-2 mode)

```armasm
ADD R0, R1, R2 ; 16-bit or 32-bit depending on registers
LDR.W R4, [R5, #100] ; 32-bit wide load instruction
BL MyFunction ; Branch with link (function call) 4. Comparison Table
```

## Comparison

| Feature           | ARM             | Thumb           | Thumb-2           |
| ----------------- | --------------- | --------------- | ----------------- |
| Instruction size  | 32-bit          | 16-bit          | 16-bit + 32-bit   |
| Code density      | Low             | High            | High (optimized)  |
| Instruction power | Full            | Reduced         | Nearly full ARM   |
| Used in Cortex-M  | (not supported) | (Cortex-M0/M0+) | (Cortex-M3/M4/M7) |

## Why Cortex-M Uses Thumb/Thumb-2

- Memory-constrained environments → smaller code size is critical.
- Simplicity → Thumb makes hardware decoding simpler.
- Efficiency → Thumb-2 gives balance between compactness and instruction power.

Thus, all modern Cortex-M microcontrollers use Thumb or Thumb-2 exclusively (no ARM mode).

## Conclusion

The evolution from ARM → Thumb → Thumb-2 represents the trade-off between performance and code size. While full ARM instructions dominate higher-performance processors (e.g., Cortex-A), Cortex-M relies on Thumb/Thumb-2 to achieve efficiency, compactness, and real-time responsiveness in embedded systems.
