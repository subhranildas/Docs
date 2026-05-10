# Introduction to Rust

## What is Rust?

Rust is a modern systems programming language designed for Performance,
Memory safety, Concurrency, Reliability.

Rust allows user to write programs that are as fast as C/C++, while preventing
many common bugs such as the following:

- Null pointer dereferences
- Data races
- Buffer overflows
- Use-after-free errors

Rust source code is written in `.rs` files and compiled into machine code by the
Rust compiler.

## Installing Rust

The recommended way to install Rust is through [rust-lang.org](https://rust-lang.org/tools/install/).

## The Rust Toolchain

A standard Rust installation includes the following:

| Tool      | Purpose                         |
| --------- | ------------------------------- |
| `rustc`   | Rust compiler                   |
| `cargo`   | Project manager and build tool  |
| `rustup`  | Toolchain installer and updater |
| `rustfmt` | Automatic code formatter        |
| `clippy`  | Linter for best practices       |

## What is Cargo?

Cargo is Rust's official package manager and build system. Cargo helps with the
following:

- Create projects
- Build programs
- Run programs
- Manage dependencies
- Format code
- Run tests

## Creating First Project

```bash
cargo new hello_rust
cd hello_rust
```

This creates a new project named `hello_rust`.

## Project Structure

```text
hello_rust/
├── Cargo.toml
└── src/
    └── main.rs
```

### `Cargo.toml`

Contains project metadata and dependencies.

```toml
[package]
name = "hello_rust"
version = "0.1.0"
edition = "2024"
```

### `src/main.rs`

Contains your Rust source code.

## Writing Your First Program

```rust
fn main() {
    println!("Hello, Rust!");
}
```

This program prints:

```text
Hello, Rust!
```

## The `main` Function

Every executable Rust program must contain a `main` function.

```rust
fn main() {
    println!("Program starts here");
}
```

Execution always begins in `main()`.

## Printing Output with `println!`

`println!` is a macro used to print text to the terminal.

### Syntax

```rust
println!("Hello, world!");
```

### Important Notes

- Macros end with `!`
- Arguments go inside parentheses
- Strings are enclosed in double quotes
- Statements end with semicolons

## Comments

Comments are ignored by the compiler and are used to explain code.

### Single-Line Comments

```rust
// This is a comment
println!("Hello");
```

### Multi-Line Comments

```rust
/*
This is a
multi-line comment.
*/
```

---

## Building and Running Programs

### Build Only

```bash
cargo build
```

### Build and Run

```bash
cargo run
```

### Check Without Building

```bash
cargo check
```

### Format Code

```bash
cargo fmt
```

These commands are part of Cargo's core workflow.

## Debug vs Release Builds

Rust can compile programs in two modes:

### Debug Build

```bash
cargo build
```

Characteristics:

- Faster compilation
- Includes debug symbols
- Minimal optimization
- Ideal during development

### Release Build

```bash
cargo build --release
```

Characteristics:

- Slower compilation
- Aggressive optimization
- Much faster executable
- Used for deployment

## Common Cargo Commands

| Command                  | Description            |
| ------------------------ | ---------------------- |
| `cargo new project_name` | Create new project     |
| `cargo build`            | Compile project        |
| `cargo run`              | Compile and run        |
| `cargo check`            | Check code quickly     |
| `cargo fmt`              | Format code            |
| `cargo test`             | Run tests              |
| `cargo build --release`  | Optimized build        |
| `cargo clean`            | Remove build artifacts |

## Dependencies

A dependency is an external library your project uses.

Add dependencies in `Cargo.toml`.

```toml
[dependencies]
rand = "0.9"
```

Then use them in your code.

## Example Program

```rust
// This program demonstrates basic Rust syntax.

fn greet() {
    println!("Welcome to Rust!");
}

fn main() {
    let name = "Subhranil";

    greet();

    println!("Hello, {}!", name);
}
```

Expected output:

```text
Welcome to Rust!
Hello, Subhranil!
```

## Key Terminology

| Term        | Meaning                              |
| ----------- | ------------------------------------ |
| Compiler    | Converts source code to machine code |
| Source Code | Human-readable code                  |
| Executable  | Program that runs on a computer      |
| Function    | Reusable block of code               |
| Macro       | Code generator invoked with `!`      |
| Variable    | Named value                          |
| String      | Text data                            |
| Comment     | Ignored explanatory text             |
| Dependency  | External library                     |
| Crate       | Rust package/library                 |

## Typical Beginner Workflow

```bash
cargo new my_project
cd my_project
cargo run
cargo check
cargo fmt
cargo build --release
```

## Why Rust is Popular

Rust is widely used because of it's offerings

- C/C++ performance
- Memory safety without garbage collection
- Excellent tooling
- Great documentation
- Strong package ecosystem
- Fearless concurrency

## Additional Resources

- [Official Website](https://rust-lang.org/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [Rustlings Exercises](https://github.com/rust-lang/rustlings)
- [rust-analyzer](https://rust-analyzer.github.io/)
