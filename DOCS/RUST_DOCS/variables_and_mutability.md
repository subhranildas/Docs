# Rust Variables and Mutability

## What is a Variable?

A variable is a name that stores a value in a program.

```rust
let age = 25;
```

In this example:

- `age` is the variable name.
- `25` is the value.

Variables make programs dynamic by allowing data to be stored and reused.

## Declaring Variables in Rust

The `let` keyword is used to declare a variable.

```rust
let name = "Subhranil";
let score = 98;
let pi = 3.14159;
```

Syntax:

```rust
let variable_name = value;
```

## Type Inference

Rust automatically determines the type of a variable from its initial value.

```rust
let count = 10;      // i32
let price = 99.99;   // f64
let active = true;   // bool
```

This is called **type inference**.

## Explicit Type Annotations

We can specify a variable's type manually using a colon (`:`).

```rust
let count: i32 = 10;
let price: f64 = 99.99;
let letter: char = 'A';
```

Syntax:

```rust
let variable_name: Type = value;
```

## Integer and Floating-Point Types

Rust infers:

- `i32` for whole numbers
- `f64` for decimal numbers

### Integers

```rust
let year = 2026; // i32
```

### Floats

```rust
let temperature = 36.6; // f64
```

---

## String Interpolation with `println!`

Use `{}` placeholders to insert values into strings.

```rust
let name = "Rust";
println!("Hello, {}!", name);
```

Output:

```text
Hello, Rust!
```

### Multiple Values

```rust
let first = "Subhranil";
let last = "Das";

println!("{} {}", first, last);
```

### Positional Arguments

```rust
println!("{0} scored {1}", "Alice", 95);
```

The first argument after the format string has index `0`.

---

## Mutability

Variables are immutable by default in Rust.

```rust
let x = 5;
// x = 6; // error
```

### Mutable Variables

Use `mut` to allow a variable's value to change.

```rust
let mut counter = 0;
counter = 1;
```

### Important Rule

A mutable variable can change its value, but not its type.

```rust
let mut x = 10;
x = 20;      // OK
// x = "hi"; // Error
```

## Constants

Constants are compile-time values that never change. They are burned into the
built binary.

```rust
const MAX_USERS: u32 = 100;
```

### Rules for Constants

- Use the `const` keyword.
- Must include a type annotation.
- Must be initialized immediately.
- Can be declared outside functions.

### Naming Convention

Use `SCREAMING_SNAKE_CASE`.

```rust
const PI: f64 = 3.1415926535;
const COMPANY_NAME: &str = "OpenAI";
```

## Variable Shadowing

Shadowing means declaring a new variable with the same name using `let`.

```rust
let x = 5;
let x = x + 1;
let x = x * 2;

println!("{}", x); // 12
```

### Why Shadowing is Useful

- Transform values.
- Change the type.
- Keep variable names meaningful.

### Example: Type Change

```rust
let spaces = "   ";
let spaces = spaces.len();
```

`spaces` changes from `&str` to `usize`.

### This is NOT allowed with `mut`

```rust
let mut spaces = "   ";
// spaces = spaces.len(); // Error
```

## Scopes and Blocks

A scope is the region where a variable is valid.

```rust
fn main() {
    let x = 5;

    {
        let y = 10;
        println!("{} {}", x, y);
    }

    // println!("{}", y); // Error: y is out of scope
}
```

### Rules

- Variables live until the end of their scope.
- Inner blocks can access outer variables.
- Outer blocks cannot access inner variables.

## Compiler Directives (Attributes)

Attributes tell the compiler how to treat code.

### Allow Unused Variables

#### For a Single Item

```rust
#[allow(unused_variables)]
fn main() {
    let x = 10;
}
```

#### For an Entire File

```rust
#![allow(unused_variables)]
```

## Type Aliases

A type alias creates a new name for an existing type.

```rust
type Kilometers = i32;

let distance: Kilometers = 42;
```

Benefits:

- Improves readability.
- Documents intent.

## Rust Error Codes

Each compiler error has a unique error code.

Example:

```text
error[E0384]: cannot assign twice to immutable variable
```

To get detailed explanations:

```bash
rustc --explain E0384
```

## Unused Variables

Rust warns about variables that are declared but not used.

```rust
let x = 10; // warning if unused
```

### Suppress Warning

Prefix the variable name with `_`.

```rust
let _x = 10;
```

## Practical Example

```rust
const MAX_SCORE: i32 = 100;

type Points = i32;

fn main() {
    let name = "Subhranil";
    let mut score: Points = 85;

    println!("{} scored {}", name, score);

    score = 95;
    println!("Updated score: {}", score);

    let score = score as f64 / MAX_SCORE as f64 * 100.0;
    println!("Percentage: {:.2}%", score);
}
```

## Common Mistakes

### Reassigning Immutable Variables

```rust
let x = 5;
// x = 6; // Error
```

### Missing Type in Constant

```rust
// const PI = 3.14; // Error
const PI: f64 = 3.14;
```

### Using Variable Outside Scope

```rust
{
    let x = 10;
}
// println!("{}", x); // Error
```

## Summary Table

| Concept          | Syntax              | Mutable? | Type Can Change? |
| ---------------- | ------------------- | -------- | ---------------- |
| Variable         | `let x = 5;`        | No       | No               |
| Mutable Variable | `let mut x = 5;`    | Yes      | No               |
| Constant         | `const X: i32 = 5;` | No       | No               |
| Shadowing        | `let x = x + 1;`    | N/A      | Yes              |

---

## Comparison: Mutability vs Shadowing

| Feature              | `mut` | Shadowing |
| -------------------- | ----- | --------- |
| Change value         | Yes   | Yes       |
| Change type          | No    | Yes       |
| Requires `let` again | No    | Yes       |
| Creates new variable | No    | Yes       |

## Additional Resources

- [Variables and Mutability](https://doc.rust-lang.org/book/ch03-01-variables-and-mutability.html)
- [Variable Bindings](https://doc.rust-lang.org/rust-by-example/variable_bindings.html)
- [rustc Error Index](https://doc.rust-lang.org/error_codes/error-index.html)
