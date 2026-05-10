# Rust Data Types

## What Are Data Types?

Every value in Rust has a data type.

A data type tells the compiler:

- What kind of value is being stored.
- How much memory to allocate.
- Which operations are valid.

Examples:

```rust
let age = 25;        // integer
let price = 99.99;   // floating-point
let active = true;   // boolean
let grade = 'A';     // character
```

## Static Typing in Rust

Rust is a **statically typed** language.
This means:

- The compiler must know the type of every variable at compile time.
- Type errors are caught before the program runs.

Benefits:

- Faster programs.
- Better compiler diagnostics.
- Improved reliability.

## Type Inference

Rust can infer types from initial values.

```rust
let x = 5;       // inferred as i32
let pi = 3.14;   // inferred as f64
let ok = true;   // inferred as bool
```

You only need to specify types when inference is ambiguous or when you want to be explicit.

## Scalar Types Overview

A scalar type stores a single value.

Rust has four scalar types:

1. Integers
2. Floating-point numbers
3. Booleans
4. Characters

## Integer Types

Integers are whole numbers without decimal points.

```rust
let count = 42;
let temperature = -10;
```

Rust supports multiple integer sizes.

### Signed Integers

- `i8`
- `i16`
- `i32`
- `i64`
- `i128`
- `isize`

### Unsigned Integers

- `u8`
- `u16`
- `u32`
- `u64`
- `u128`
- `usize`

### Signed vs Unsigned Integers

Signed integers can store both positive and negative numbers.

```rust
let debt: i32 = -500;
```

Unsigned integers store only zero and positive numbers.

```rust
let age: u32 = 30;
```

Because unsigned types do not need to represent negative values, they can represent larger positive values with the same number of bits.

## Integer Sizes and Memory Usage

The number after `i` or `u` indicates the number of bits used. fileciteturn2file0L70-L70

| Type            | Bits | Bytes |
| --------------- | ---: | ----: |
| `i8` / `u8`     |    8 |     1 |
| `i16` / `u16`   |   16 |     2 |
| `i32` / `u32`   |   32 |     4 |
| `i64` / `u64`   |   64 |     8 |
| `i128` / `u128` |  128 |    16 |

Default integer type in Rust: `i32`.

## Integer Ranges

Common ranges:

| Type  |        Minimum |       Maximum |
| ----- | -------------: | ------------: |
| `i8`  |           -128 |           127 |
| `u8`  |              0 |           255 |
| `i16` |        -32,768 |        32,767 |
| `u16` |              0 |        65,535 |
| `i32` | -2,147,483,648 | 2,147,483,647 |
| `u32` |              0 | 4,294,967,295 |

## Floating-Point Types

Floating-point numbers represent decimal values.

```rust
let pi = 3.14159;
let temperature = 36.6;
```

Rust supports the following floating point types:

- `f32`
- `f64`

Default floating-point type: `f64`.

### Floating-Point Precision

Precision determines how many significant digits can be represented.

| Type  | Precision     |
| ----- | ------------- |
| `f32` | ~6–9 digits   |
| `f64` | ~15–17 digits |

Use `f64` for most applications unless memory or performance constraints require `f32`.

## Boolean Type

The Boolean type is `bool`.

Possible values:

- `true`
- `false`

```rust
let is_logged_in = true;
let finished = false;
```

Used in conditions and control flow.

## Character Type

The character type is `char`.

```rust
let grade = 'A';
let heart = '❤';
let emoji = '🦀';
```

Notes:

- Uses single quotes.
- Represents a Unicode scalar value.
- Occupies 4 bytes.

## Choosing the Right Data Type

### Use `i32`

General-purpose integers.

### Use `u32`

Values that cannot be negative (e.g., counts).

### Use `f64`

Most decimal calculations.

### Use `bool`

Logical conditions.

### Use `char`

Single Unicode characters.

## Type Annotations

Explicitly specify types when needed.

```rust
let age: u8 = 25;
let price: f64 = 19.99;
let initial: char = 'S';
let active: bool = true;
```

## Practical Examples

### Integer Arithmetic

```rust
let a: i32 = 10;
let b: i32 = 3;

println!("Sum: {}", a + b);
println!("Difference: {}", a - b);
println!("Product: {}", a * b);
println!("Quotient: {}", a / b);
println!("Remainder: {}", a % b);
```

### Floating-Point Arithmetic

```rust
let x = 2.5;
let y = 1.2;

println!("{}", x + y);
```

### Mixed Types Require Conversion

```rust
let a: i32 = 5;
let b: f64 = 2.5;

let result = a as f64 + b;
```

## Common Mistakes

### Mixing Types

```rust
let a = 5;
let b = 2.5;
// let c = a + b; // Error
```

### Integer Overflow

```rust
let x: u8 = 255;
// let y = x + 1; // Overflow
```

### Using Double Quotes for `char`

```rust
// let c: char = "A"; // Error
let c: char = 'A';
```

## Additional Resources

- [Data Types](https://doc.rust-lang.org/book/ch03-02-data-types.html)
- [Primitives](https://doc.rust-lang.org/rust-by-example/primitives.html)
- [Rust Standard Library Primitive Types](https://doc.rust-lang.org/std/primitive/)
