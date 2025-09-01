## Arithmetic Instructions

### 1. ADD (Addition)

- **Purpose**: Add two registers or a register and an immediate value.
- **Syntax**:

```armasm
ADD <Rd>, <Rn>, <Operand2>
<Rd> → Destination register

<Rn> → First operand

<Operand2> → Second operand (register or immediate)

Example:

armasm
Copy code
    MOV R0, #5
    MOV R1, #10
    ADD R2, R0, R1   ; R2 = R0 + R1 → 15
2. SUB (Subtraction)
Purpose: Subtract a register or immediate from another register.

Syntax:

armasm
Copy code
SUB <Rd>, <Rn>, <Operand2>
Example:

armasm
Copy code
    MOV R0, #20
    MOV R1, #7
    SUB R2, R0, R1   ; R2 = 20 - 7 → 13
3. MUL (Multiplication)
Purpose: Multiply two registers.

Syntax:

armasm
Copy code
MUL <Rd>, <Rn>, <Rm>
Example:

armasm
Copy code
    MOV R0, #4
    MOV R1, #5
    MUL R2, R0, R1   ; R2 = 4 * 5 → 20
4. MLA (Multiply-Accumulate)
Purpose: Multiply two registers and add a third register.

Syntax:

armasm
Copy code
MLA <Rd>, <Rn>, <Rm>, <Ra>
Example:

armasm
Copy code
    MOV R0, #2
    MOV R1, #3
    MOV R2, #5
    MLA R3, R0, R1, R2 ; R3 = (2*3) + 5 → 11
5. CMP (Compare)
Purpose: Compare two values by subtracting them and updating condition flags (does not store the result).

Syntax:

armasm
Copy code
CMP <Rn>, <Operand2>
Example:

armasm
Copy code
    MOV R0, #10
    MOV R1, #20
    CMP R0, R1    ; Sets flags for R0 - R1 → negative result
6. RSB (Reverse Subtract)
Purpose: Reverse subtraction (Operand2 - Operand1).

Syntax:

armasm
Copy code
RSB <Rd>, <Rn>, <Operand2>
Example:

armasm
Copy code
    MOV R0, #5
    MOV R1, #12
    RSB R2, R0, R1  ; R2 = 12 - 5 → 7
Logical Instructions
1. AND (Bitwise AND)
Purpose: Perform bitwise AND between two registers.

Syntax:

armasm
Copy code
AND <Rd>, <Rn>, <Operand2>
Example:

armasm
Copy code
    MOV R0, #0b1100
    MOV R1, #0b1010
    AND R2, R0, R1   ; R2 = 0b1000
2. ORR (Bitwise OR)
Purpose: Perform bitwise OR between two registers.

Syntax:

armasm
Copy code
ORR <Rd>, <Rn>, <Operand2>
Example:

armasm
Copy code
    MOV R0, #0b1100
    MOV R1, #0b1010
    ORR R2, R0, R1   ; R2 = 0b1110
3. EOR (Bitwise XOR)
Purpose: Perform bitwise XOR between two registers.

Syntax:

armasm
Copy code
EOR <Rd>, <Rn>, <Operand2>
Example:

armasm
Copy code
    MOV R0, #0b1100
    MOV R1, #0b1010
    EOR R2, R0, R1   ; R2 = 0b0110
4. BIC (Bit Clear)
Purpose: Clears bits specified by the second operand in the first operand.

Syntax:

armasm
Copy code
BIC <Rd>, <Rn>, <Operand2>
Example:

armasm
Copy code
    MOV R0, #0b1111
    MOV R1, #0b0101
    BIC R2, R0, R1   ; R2 = 0b1010
5. MVN (Move Not)
Purpose: Move the bitwise NOT of a value into a register.

Syntax:

armasm
Copy code
MVN <Rd>, <Operand2>
Example:

armasm
Copy code
    MOV R0, #0b1010
    MVN R1, R0       ; R1 = 0b0101 (inverted)
Shift and Rotate Instructions
These are often used with arithmetic/logical instructions for flexible operations.

Instruction	Purpose	Example
LSL	Logical shift left (multiply by 2^n)	LSL R2, R0, #2
LSR	Logical shift right (divide by 2^n)	LSR R2, R0, #1
ASR	Arithmetic shift right (preserve sign)	ASR R2, R0, #1
ROR	Rotate right	ROR R2, R0, #4

Example with LSL:

armasm
Copy code
    MOV R0, #3
    LSL R1, R0, #2    ; R1 = 3 << 2 → 12
Combined Example: Arithmetic and Logic Together
armasm
Copy code
    MOV R0, #10
    MOV R1, #5

    ADD R2, R0, R1    ; R2 = 15
    SUB R3, R0, R1    ; R3 = 5
    AND R4, R0, R1    ; R4 = 0
    ORR R5, R0, R1    ; R5 = 15
    EOR R6, R0, R1    ; R6 = 15
    LSL R7, R0, #1    ; R7 = 20
Conclusion
Arithmetic instructions → ADD, SUB, MUL, MLA, CMP, RSB

Logical instructions → AND, ORR, EOR, BIC, MVN

Shift/rotate instructions → LSL, LSR, ASR, ROR

Combine them for complex arithmetic, bit manipulations, and efficient computations in Thumb-2 assembly.
```
