# Rust Compiler Internals: From Source Code to Native Binary

> Comprehensive documentation of how the Rust compiler (`rustc`) transforms Rust source code into an optimized executable.

---

# Table of Contents

1. Introduction
2. Compiler Pipeline Overview
3. Lexical Analysis (Lexing)
4. Parsing
5. Abstract Syntax Tree (AST)
6. Macro Expansion
7. Name Resolution
8. HIR (High-Level Intermediate Representation)
9. Type Inference and Type Checking
10. THIR (Typed High-Level Intermediate Representation)
11. MIR (Mid-Level Intermediate Representation)
12. Borrow Checker
13. Constant Evaluation
14. MIR Optimizations
15. Monomorphization
16. LLVM IR Generation
17. LLVM Optimizations
18. Assembly Generation
19. Object File Generation
20. Linking
21. Cargo Build System
22. Incremental Compilation
23. Compiler Query System
24. Viewing Intermediate Representations
25. rustc Internal Crates
26. Complete Example Walkthrough
27. Rust vs C/C++ Compilation
28. Hardware-Oriented Perspective
29. Summary
30. References

---

# 1. Introduction

The Rust compiler, `rustc`, is a modern optimizing compiler that:

- Parses Rust source code.
- Performs rigorous static analysis.
- Enforces ownership and borrowing rules.
- Optimizes the program.
- Generates machine code.
- Produces native binaries.

Rust's distinguishing feature is that memory safety is guaranteed at compile time without a garbage collector.

---

# 2. Compiler Pipeline Overview

```text
main.rs
   │
   ▼
Lexing
   │
   ▼
Parsing
   │
   ▼
AST
   │
   ▼
Macro Expansion
   │
   ▼
Name Resolution
   │
   ▼
HIR
   │
   ▼
Type Checking
   │
   ▼
THIR
   │
   ▼
MIR
   │
   ├── Borrow Checker
   ├── Const Evaluation
   └── MIR Optimizations
   │
   ▼
Monomorphization
   │
   ▼
LLVM IR
   │
   ▼
LLVM Optimizations
   │
   ▼
Assembly
   │
   ▼
Object Files
   │
   ▼
Linker
   │
   ▼
Executable Binary
```

---

# 3. Lexical Analysis (Lexing)

The lexer converts raw source code into tokens.

## Input

```rust
fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

## Output Tokens

```text
fn
add
(
a
:
i32
,
b
:
i32
)
->
i32
{
a
+
b
}
```

## Responsibilities

- Remove whitespace.
- Remove comments.
- Identify keywords.
- Identify identifiers.
- Identify literals.
- Identify operators.

## Internal Crate

- `rustc_lexer`

---

# 4. Parsing

The parser consumes tokens and builds a syntax tree.

## Internal Crate

- `rustc_parse`

## Responsibilities

- Validate grammar.
- Report syntax errors.
- Construct AST nodes.

---

# 5. Abstract Syntax Tree (AST)

The AST preserves the syntactic structure.

```text
Function add
 ├── Parameters: a, b
 ├── Return Type: i32
 └── Body:
      BinaryOp(+)
        ├── a
        └── b
```

## Internal Crate

- `rustc_ast`

---

# 6. Macro Expansion

Rust macros are expanded before most semantic analysis.

## Macro Types

- Declarative macros (`macro_rules!`)
- Procedural macros
- Attribute macros
- Derive macros

## Example

```rust
println!("{}", x);
```

Expands into formatting and I/O calls.

## Internal Crates

- `rustc_expand`
- `rustc_proc_macro`

---

# 7. Name Resolution

Associates identifiers with their definitions.

## Example

```rust
add      -> function item
println! -> macro
i32      -> primitive type
```

## Responsibilities

- Scope analysis.
- Module resolution.
- Trait and type resolution.

## Internal Crate

- `rustc_resolve`

---

# 8. HIR (High-Level Intermediate Representation)

HIR is a simplified representation used for semantic analysis.

## Example Desugaring

```rust
for i in 0..10 {
    work(i);
}
```

becomes:

```rust
let mut iter = (0..10).into_iter();
loop {
    match iter.next() {
        Some(i) => work(i),
        None => break,
    }
}
```

## Advantages

- Simplified syntax.
- Easier for analysis.
- Stable internal representation.

## Internal Crate

- `rustc_hir`

---

# 9. Type Inference and Type Checking

The compiler determines types and verifies correctness.

## Example

```rust
let x = 5;
```

Inferred as:

```rust
let x: i32 = 5;
```

## Checks Performed

- Expression types.
- Function argument types.
- Trait bounds.
- Generic constraints.

## Internal Crate

- `rustc_typeck`

---

# 10. THIR (Typed HIR)

THIR is a fully typed representation used as a bridge to MIR.

## Uses

- Exhaustiveness checking.
- Pattern analysis.
- MIR construction.

---

# 11. MIR (Mid-Level Intermediate Representation)

MIR is a control-flow graph representation.

## Example

```text
bb0:
  _1 = const 2
  _2 = const 3
  _3 = add(_1, _2)
  _4 = _3
  return
```

## Characteristics

- SSA-like temporaries.
- Explicit control flow.
- Simpler than AST/HIR.

## Internal Crate

- `rustc_middle::mir`

---

# 12. Borrow Checker

The borrow checker verifies ownership and reference safety using MIR.

## Ownership Rules

1. Every value has one owner.
2. Either one mutable reference or many immutable references.
3. References must not outlive the owner.

## Example Error

```rust
let mut x = 5;
let r1 = &x;
let r2 = &mut x; // error
```

## Analyses

- Move analysis.
- Liveness analysis.
- Region inference.
- Alias checking.

## Internal Crate

- `rustc_borrowck`

---

# 13. Constant Evaluation

Compile-time execution of constant expressions.

## Example

```rust
const N: usize = 4 * 1024;
```

Evaluated to:

```rust
const N: usize = 4096;
```

## Internal Crate

- `rustc_const_eval`

---

# 14. MIR Optimizations

Common optimizations before lowering to LLVM.

- Constant propagation.
- Dead code elimination.
- Simplify branches.
- Copy propagation.
- Inlining.

---

# 15. Monomorphization

Generics are specialized into concrete functions.

## Generic Function

```rust
fn identity<T>(x: T) -> T {
    x
}
```

## Instantiations

```text
identity_i32
identity_bool
identity_f64
```

## Benefits

- Zero-cost abstractions.
- Fully optimized concrete code.

## Trade-Off

- Larger binary size.

---

# 16. LLVM IR Generation

MIR is lowered into LLVM Intermediate Representation.

## Example

```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
  %0 = add i32 %a, %b
  ret i32 %0
}
```

## Internal Crates

- `rustc_codegen_ssa`
- `rustc_codegen_llvm`

---

# 17. LLVM Optimizations

LLVM performs machine-independent and target-specific optimizations.

- Inlining.
- Loop unrolling.
- Vectorization.
- Register allocation.
- Instruction scheduling.

---

# 18. Assembly Generation

LLVM emits target assembly.

## Example (x86-64)

```asm
add:
    lea eax, [rdi + rsi]
    ret
```

---

# 19. Object File Generation

Assembler converts assembly into object files (`.o`).

## Contents

- Machine code.
- Symbol tables.
- Relocations.
- Debug information.

---

# 20. Linking

The linker combines object files and libraries.

## Inputs

- Application object files.
- Rust standard library.
- Native libraries.

## Outputs

- Linux: ELF
- Windows: PE/COFF
- macOS: Mach-O

---

# 21. Cargo Build System

`cargo` is Rust's package manager and build orchestrator.

## Responsibilities

- Dependency resolution.
- Build scripts.
- Feature flags.
- Incremental compilation.
- Running `rustc`.

## Commands

```bash
cargo build
cargo build --release
cargo test
cargo run
```

---

# 22. Incremental Compilation

Only recompiles changed portions.

## Benefits

- Faster edit-build cycles.
- Reuse of cached artifacts.

---

# 23. Compiler Query System

Rust compiler is query-based.

## Example Queries

- Parse crate.
- Type check item.
- Generate MIR.
- Borrow check function.
- Generate LLVM IR.

## Benefits

- Lazy evaluation.
- Incremental compilation.
- Fine-grained caching.

---

# 24. Viewing Intermediate Representations

## HIR

```bash
cargo rustc -- -Z unpretty=hir-tree
```

## MIR

```bash
cargo rustc -- --emit=mir
```

## LLVM IR

```bash
cargo rustc -- --emit=llvm-ir
```

## Assembly

```bash
cargo rustc -- --emit=asm
```

---

# 25. rustc Internal Crates

| Crate                | Purpose              |
| -------------------- | -------------------- |
| `rustc_lexer`        | Tokenization         |
| `rustc_parse`        | Parsing              |
| `rustc_ast`          | AST definitions      |
| `rustc_expand`       | Macro expansion      |
| `rustc_resolve`      | Name resolution      |
| `rustc_hir`          | HIR                  |
| `rustc_typeck`       | Type checking        |
| `rustc_middle`       | Shared compiler data |
| `rustc_borrowck`     | Borrow checker       |
| `rustc_const_eval`   | Const evaluation     |
| `rustc_codegen_ssa`  | Codegen abstraction  |
| `rustc_codegen_llvm` | LLVM backend         |

---

# 26. Complete Example Walkthrough

## Source

```rust
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let x = add(2, 3);
    println!("{}", x);
}
```

## Pipeline Summary

### Lexing

Produces tokens.

### Parsing

Produces AST.

### Macro Expansion

Expands `println!`.

### HIR

Simplified semantic representation.

### Type Checking

Verifies all expressions.

### MIR

Creates control-flow graph.

### Borrow Checker

Ensures safety.

### Monomorphization

Specializes generics (none here).

### LLVM IR

Lower-level representation.

### Optimization

LLVM optimizes code.

### Assembly

Target-specific instructions.

### Linking

Creates final executable.

---

# 27. Rust vs C/C++ Compilation

| Feature                  | Rust   | C/C++  |
| ------------------------ | ------ | ------ |
| Ownership checking       | Yes    | No     |
| Borrow checking          | Yes    | No     |
| Lifetime analysis        | Yes    | No     |
| Monomorphization         | Yes    | Yes    |
| LLVM backend             | Yes    | Often  |
| Memory safety guarantees | Strong | Manual |

---

# 28. Hardware-Oriented Perspective

Each stage can be viewed as a transformation pipeline.

| Compiler Stage      | Hardware Analogy          |
| ------------------- | ------------------------- |
| Lexing              | Instruction decoder       |
| Parsing             | Microcode generation      |
| Type checking       | Static verification       |
| MIR                 | Control/data-flow graph   |
| Borrow checker      | Hazard detection          |
| LLVM optimization   | Logic optimization        |
| Register allocation | Physical register mapping |
| Linking             | Netlist integration       |

This perspective is useful for designing compiler-assisted accelerators and custom ML hardware.

---

# 29. Summary

Rust compilation proceeds through:

1. Lexing
2. Parsing
3. AST generation
4. Macro expansion
5. Name resolution
6. HIR construction
7. Type checking
8. THIR construction
9. MIR generation
10. Borrow checking
11. Const evaluation
12. MIR optimization
13. Monomorphization
14. LLVM IR generation
15. LLVM optimization
16. Assembly generation
17. Object file generation
18. Linking
19. Final executable

Rust guarantees memory safety before machine code is emitted.

---

# 30. References

## Official Documentation

- https://doc.rust-lang.org/book/
- https://doc.rust-lang.org/rustc/
- https://rustc-dev-guide.rust-lang.org/
- https://llvm.org/
- https://doc.rust-lang.org/cargo/

## Recommended Topics to Explore

- Ownership and Borrowing
- Lifetimes
- MIR Internals
- Polonius Borrow Checker
- LLVM Passes
- Linkers (`lld`, `gold`, `mold`)

---

# Appendix A: Frontend, Middle-End, Backend

| Phase             | Components                               |
| ----------------- | ---------------------------------------- |
| Frontend          | Lexer, Parser, AST, Macro Expansion, HIR |
| Semantic Analysis | Type Checking, Trait Solving             |
| Middle-End        | MIR, Borrow Checker, MIR Optimization    |
| Backend           | LLVM IR, LLVM Passes, Assembly           |
| Finalization      | Object Files, Linking                    |

---

# Appendix B: Typical Build Command

```bash
cargo build --release
```

Equivalent to:

```bash
rustc -C opt-level=3 src/main.rs
```

---

# Appendix C: Build Artifacts

| Artifact          | Description         |
| ----------------- | ------------------- |
| `.rs`             | Source code         |
| `.rmeta`          | Metadata            |
| `.o`              | Object file         |
| `.rlib`           | Rust static library |
| `.so/.dll/.dylib` | Shared library      |
| Executable        | Final binary        |

---

# Final Mental Model

```text
Human-Friendly Rust Code
        ↓
Structured Compiler Representations
        ↓
Safety Proofs (Ownership + Borrowing)
        ↓
Optimization
        ↓
Machine Instructions
        ↓
Native Executable
```

Rust's key innovation is proving memory safety at compile time while preserving zero-cost abstractions.
