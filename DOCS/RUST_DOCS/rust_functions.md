# Rust Functions

## What is a Function?

A function is a sequence of steps that performs a specific task.

Functions help us with the following:

- Organize code.
- Reuse logic.
- Improve readability.
- Reduce duplication.

## Why Functions Matter

Without functions, code quickly becomes repetitive and difficult to maintain.
Functions allows us to break large problems into smaller pieces.

## Function Syntax

Functions begin with the `fn` keyword and use `snake_case` names.

```rust
fn greet() {
    println!("Hello!");
}
```

Syntax:

```rust
fn function_name() {
    // body
}
```

## The `main` Function

Every executable Rust program starts with `main()`.

```rust
fn main() {
    println!("Program starts here.");
}
```

The Rust runtime automatically invokes `main()`.

## Calling Functions

Functions are called by writing their name followed by parentheses.

```rust
fn greet() {
    println!("Hello!");
}

fn main() {
    greet();
}
```

Output:

```text
Hello!
```

## Parameters and Arguments

A parameter is a named input in a function definition.
An argument is the actual value passed when calling the function.

### Example

```rust
fn greet(name: &str) {
    println!("Hello, {}!", name);
}

fn main() {
    greet("Subhranil");
}
```

- Parameter: `name: &str`
- Argument: `"Subhranil"`

## Multiple Parameters

Separate parameters with commas.

```rust
fn add(a: i32, b: i32) {
    println!("{} + {} = {}", a, b, a + b);
}
```

## Return Values

A return value is the output of a function.

Specify the return type using `->`.

```rust
fn square(x: i32) -> i32 {
    x * x
}
```

## Explicit Returns

Use the `return` keyword and end with a semicolon.

```rust
fn square(x: i32) -> i32 {
    return x * x;
}
```

## Implicit Returns

The last expression without a semicolon is returned automatically.

```rust
fn square(x: i32) -> i32 {
    x * x
}
```

### Important

```rust
fn square(x: i32) -> i32 {
    x * x; // Error: semicolon turns this into a statement
}
```

## The Unit Type `()`

If a function does not return anything, it returns the unit type `()`.

```rust
fn say_hi() {
    println!("Hi!");
}
```

Equivalent signature:

```rust
fn say_hi() -> () {
    println!("Hi!");
}
```

## Blocks in Functions

A block is a section of code enclosed in `{}`.

Blocks:

- Create a new scope.
- Can contain multiple statements.
- Can evaluate to a final value.

```rust
let result = {
    let x = 5;
    x + 1
};

println!("{}", result); // 6
```

## Assigning Block Results

Because blocks can produce values, they can be assigned to variables.

```rust
let y = {
    let x = 3;
    x * 2
};

println!("{}", y); // 6
```

## Practical Examples

### Example 1: Greeting Function

```rust
fn greet(name: &str) {
    println!("Hello, {}!", name);
}
```

### Example 2: Addition Function

```rust
fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

### Example 3: Explicit Return

```rust
fn max(a: i32, b: i32) -> i32 {
    if a > b {
        return a;
    }
    b
}
```

## Common Mistakes

### Missing Parameter Types

```rust
// fn greet(name) {} // Error
fn greet(name: &str) {}
```

### Missing Return Type

```rust
// fn square(x: i32) { x * x } // Error
fn square(x: i32) -> i32 { x * x }
```

### Semicolon on Final Expression

```rust
fn square(x: i32) -> i32 {
    x * x; // Error
}
```

## Example Program

```rust
fn greet(name: &str) {
    println!("Hello, {}!", name);
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    greet("Subhranil");

    let result = add(10, 20);
    println!("Result = {}", result);

    let value = {
        let x = 5;
        x * 2
    };

    println!("Block value = {}", value);
}
```

# 19. Additional Resources

- [Functions](https://doc.rust-lang.org/book/ch03-03-how-functions-work.html)
- [Example – Functions](https://doc.rust-lang.org/rust-by-example/fn.html)
