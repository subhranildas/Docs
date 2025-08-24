## Number Systems

Assembly programming deals closely with **numbers**, but computers do not understand numbers the way we do.
Instead, they represent everything — numbers, text, instructions — as **binary (0s and 1s)**.

To program at this level, we need to understand how different number systems work and how to convert between them.

---

## The Common Number Systems

### Decimal (Base 10)

- The system we use in everyday life.
- Digits: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9`
- Place values are powers of 10.

Example:  
`345` in decimal = `(3 × 10²) + (4 × 10¹) + (5 × 10⁰)` = `300 + 40 + 5`

---

### Binary (Base 2)

- The **native language of computers**.
- Digits: `0, 1`
- Place values are powers of 2.

Example:  
`1011₂` = `(1 × 2³) + (0 × 2²) + (1 × 2¹) + (1 × 2⁰)`  
= `8 + 0 + 2 + 1 = 11₁₀`

---

### Hexadecimal (Base 16)

- Shorthand for binary, commonly used in assembly.
- Digits: `0–9` and `A–F` (A=10, B=11, …, F=15).
- Place values are powers of 16.

Example:  
`2F₁₆` = `(2 × 16¹) + (15 × 16⁰)`  
= `32 + 15 = 47₁₀`

---

### Octal (Base 8) _(less common today)_

- Digits: `0–7`
- Place values are powers of 8.
- Historically used in UNIX systems.

Example:  
`57₈` = `(5 × 8¹) + (7 × 8⁰)`  
= `40 + 7 = 47₁₀`

---

## Why Different Number Systems?

- **Binary**: What computers actually use internally.
- **Hexadecimal**: Easier for humans to read/write than long binary strings.
  - Example: `1111 1111₂` = `FF₁₆`
- **Decimal**: Natural for humans, used in daily life.
- **Octal**: Was used for compact binary representation before hex became dominant.

---

## Conversions Between Number Systems

### Decimal ↔ Binary

- **To Binary**: Divide the decimal number by 2 repeatedly, record remainders.  
   Example: Convert `13₁₀` → Binary  
  13 ÷ 2 = 6 remainder 1
  6 ÷ 2 = 3 remainder 0
  3 ÷ 2 = 1 remainder 1
  1 ÷ 2 = 0 remainder 1

yaml
Copy
Edit
Reading remainders bottom to top → `1101₂`

- **To Decimal**: Multiply each binary digit by powers of 2.  
  Example: `1101₂` → `(1×8) + (1×4) + (0×2) + (1×1)` = `13₁₀`

---

### Decimal ↔ Hexadecimal

- **To Hex**: Divide the decimal number by 16 repeatedly, record remainders.  
  Example: `255₁₀` → Hex  
  255 ÷ 16 = 15 remainder 15 (F)
  15 ÷ 16 = 0 remainder 15 (F)

→ `FF₁₆`

- **To Decimal**: Multiply each hex digit by powers of 16.  
  Example: `2F₁₆` = `(2×16) + (15×1)` = `47₁₀`

---

### Binary ↔ Hexadecimal

- Binary digits grouped into **nibbles (4 bits)**, convert directly.  
  Example: `1010 1111₂`  
  → `1010₂ = A₁₆`  
  → `1111₂ = F₁₆`  
  → `AF₁₆`

- Reverse works the same way: each hex digit expands into 4 binary digits.  
  Example: `B9₁₆` → `1011 1001₂`

---

### Binary ↔ Octal

- Binary digits grouped into **triplets (3 bits)**.  
  Example: `110101₂` → `110 101` = `65₈`

---

## Number Representation in Computers

- **Unsigned integers**: Only positive numbers (e.g., `0` to `255` in 8 bits).
- **Signed integers**: Positive and negative numbers using **two’s complement**.
- Example (8-bit):
  - `0000 1010₂ = +10`
  - `1111 0110₂ = -10`

Understanding two’s complement is **critical in assembly**, since CPU arithmetic depends on it.

---

### Quick Reference Table

| Decimal | Binary   | Hex | Octal |
| ------- | -------- | --- | ----- |
| 0       | 0000     | 0   | 0     |
| 1       | 0001     | 1   | 1     |
| 2       | 0010     | 2   | 2     |
| 8       | 1000     | 8   | 10    |
| 10      | 1010     | A   | 12    |
| 15      | 1111     | F   | 17    |
| 16      | 10000    | 10  | 20    |
| 255     | 11111111 | FF  | 377   |

---

## Conclusion

Computers think in **binary**, but humans prefer **decimal**. Hexadecimal provides a **bridge** between the two — compact for humans, yet easy to map back to binary.

---
