## General Idea about Security (Digital signing, hash, encryption)

### Hashing — Ensuring Integrity

Hashing converts any input data (message, file, firmware, password) into a fixed-size unique “fingerprint” called a hash or digest.

#### Properties

- One-way: You cannot reverse a hash to get the original data.
- Deterministic: Same input → same hash.
- Sensitive: Even one bit change → completely different hash.
- Fixed length: Output is fixed (e.g., 256 bits for SHA-256), no matter input size.

#### Common Algorithms

- SHA-256, SHA-512 → modern, secure hashes
- MD5, SHA-1 → older, now considered broken for strong security

- Example

  - "Hello" → 185F8DB32271FE25F561A6FC938B2E264306EC304EDA518007D1764826381969
  - "hello" → 2CF24DBA5FB0A30E26E83B2AC5BCD788...

  One character change → totally different hash.

#### Use Cases

- Check data integrity (firmware update validation, file verification)
- Store password hashes securely (instead of plain passwords)

### Properties

- One-way: You cannot reverse a hash to get the original data.
- Deterministic: Same input → same hash.
- Sensitive: Even one bit change → completely different hash.
- Fixed length: Output is fixed (e.g., 256 bits for SHA-256), no matter input size.

### Common Algorithms

- SHA-256, SHA-512 → modern, secure hashes
- MD5, SHA-1 → older, now considered broken for strong security

### Example

- "Hello" → 185F8DB32271FE25F561A6FC938B2E264306EC304EDA518007D1764826381969
- "hello" → 2CF24DBA5FB0A30E26E83B2AC5BCD788...

One character change → totally different hash.

### Use Cases

- Check data integrity (firmware update vaildation, file verification)
- Store password hashes securely (instead of plain passwords)
- Input to digital signatures
