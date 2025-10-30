## If you create a circular buffer, what size of buffer might optimized code be slightly faster to execute? why?

Normally, when you implement a circular buffer, you maintain an index (like head or tail) and wrap it around when it reaches the buffer end:

```c
index = (index + 1) % buffer_size;
```

If the buffer size is a power of 2, we can replace this with a bitwise AND, which is extremely fast:

```c
#define BUFFER_SIZE 64  // Power of 2
index = (index + 1) & (BUFFER_SIZE - 1);
```

Because for a power-of-2 size, (BUFFER_SIZE - 1) creates a binary mask that automatically wraps the index.

Example:

- 64 → 0b0100_0000
- 63 → 0b0011_1111 (mask)

So (index & 63) is equivalent to (index % 64) — but much faster.

## Explain when you should use "volatile" in C.

In C, the volatile keyword tells the compiler:

> “This variable can change at any time outside of the normal program flow, so do not optimize or cache it.”

> The variable can change at any time outside the program flow, so the compiler must always read it from memory and cannot optimize it away.

Normally, the compiler assumes:

- A variable doesn’t change unless the code in your program explicitly modifies it.
- It can store the variable in a register, or reorder reads/writes for optimization.
- But sometimes, this assumption is wrong — and that’s when you need volatile.

### When to use

#### Hardware Registers / Memory-mapped I/O

Embedded systems often access peripheral registers (GPIO, UART, timers, ADC, etc.) that can change independently of your code.

```c
#define GPIO_STATUS_REG (*(volatile uint32_t *)0x40020010)

while(!(GPIO_STATUS_REG & 0x01)) {
    // wait for pin to go high
}
```

Without volatile, the compiler might optimize away the repeated reads of the register — thinking it never changes in this loop.

#### Global Variables Modified by Interrupts (ISRs)

- If a global variable is updated inside an ISR but read in main code:

```c
volatile uint8_t buttonPressed = 0;

void EXTI0_IRQHandler(void) {
    buttonPressed = 1;
}

// Main loop
while(1) {
    if(buttonPressed) {
        buttonPressed = 0;
        toggleLED();
    }
}
```

Without volatile, the compiler could cache buttonPressed in a register and never see the ISR’s update.

#### Variables Shared Between Threads / RTOS Tasks

When multiple tasks access a variable without proper locks, volatile ensures:

- Every read/write actually touches memory
- Compiler does not assume previous value is still valid

Note: volatile alone does not make operations atomic. For multi-threaded safety, you still need mutexes or atomic operations.

#### Variables Changed by External Events

Examples:

- Memory-mapped sensors
- DMA buffers
- Flags set by co-processors or external hardware

## Explain when you should use "constant volatile" in C.

### When to Use const volatile

Read-only Hardware Registers

- Some hardware registers are read-only, but their values can change asynchronously due to hardware.

```c
#define ADC_DATA_REG (*(const volatile uint16_t *)0x4001244C)

uint16_t value = ADC_DATA_REG; // Always read the latest ADC value
```

- const prevents accidental writes in your code.
- volatile ensures the compiler always reads the register from memory rather than caching it.

Status Registers That Change Externally

- Example: Flags set by sensors or peripherals, which your program can read but must not write.

```c
const volatile uint32_t STATUS_REG = 0x40021000;
```

- Program can only read STATUS_REG.
- Each read gets the current value — no caching in registers.

Memory-mapped Read-only Buffers Updated by DMA

- If DMA updates a memory buffer that the CPU only reads:

```c
const volatile uint8_t dmaBuffer[128];
```

- const prevents accidental write by CPU.
- volatile ensures CPU always reads the latest DMA-updated values.

> Use const volatile whenever you have a read-only value controlled externally (hardware or DMA), and you want to protect it from accidental writes while ensuring each read reflects the current hardware state.

## What are the basic concepts of how printf() works? List and describe some of the special format characters? Show some simple C coding examples.

printf() is a standard C library function that prints formatted output to stdout.

Format string parsing

- The first argument is a format string (e.g., "Value = %d\n").
- printf() scans it for conversion specifiers (like %d, %f, %s).

Argument handling

- Each specifier corresponds to an argument in the variable argument list.
- printf() reads the values, converts them to text, and inserts them into the output.

Output

- The resulting string is written to the standard output (console, terminal, UART, or file, depending on environment).
- Behind the scenes, printf() uses va_list, va_start, and va_arg macros for handling variable arguments.

### Common Format Specifiers

| Specifier   | Meaning                                 | Example                                          |
| ----------- | --------------------------------------- | ------------------------------------------------ |
| `%d` / `%i` | Signed decimal integer                  | `int x = 42; printf("%d", x);` → `42`            |
| `%u`        | Unsigned decimal integer                | `unsigned int u = 100; printf("%u", u);` → `100` |
| `%x` / `%X` | Unsigned hexadecimal (lower/upper case) | `printf("%x", 255);` → `ff`                      |
| `%o`        | Unsigned octal                          | `printf("%o", 8);` → `10`                        |
| `%f`        | Floating-point (decimal notation)       | `float pi=3.14; printf("%f", pi);` → `3.140000`  |
| `%e` / `%E` | Floating-point scientific notation      | `printf("%e", 12345.0);` → `1.234500e+04`        |
| `%c`        | Single character                        | `printf("%c", 'A');` → `A`                       |
| `%s`        | Null-terminated string                  | `printf("%s", "Hello");` → `Hello`               |
| `%%`        | Literal percent sign                    | `printf("%%");` → `%`                            |

## How do you determine if a memory address is aligned on a 4 byte boundary in C?

In C, checking whether a memory address is aligned to a 4-byte boundary is straightforward — you just check if the lowest 2 bits of the address are zero, because 4 bytes = 2 exponent 2;

- A memory address is aligned to N bytes if it’s divisible by N.
- In binary, 4-byte alignment means the lowest 2 bits are 0.

```c
#define IS_ALIGNED(ptr, N) (((uintptr_t)(ptr) & ((N)-1)) == 0)
```

## Show how to declare a pointer to constant data in C. Show how to declare a function pointer in C.

### Pointer to Constant Data

A pointer to constant data means:
The data cannot be modified through the pointer,
but the pointer can point to a different address later.

#### Syntax

```c
const int *ptr;
```

#### Variations to know

| Declaration           | Meaning                                                                     |
| --------------------- | --------------------------------------------------------------------------- |
| `const int *p;`       | Pointer to **const data**, data can’t change through `p`, but `p` can move. |
| `int *const p;`       | **Constant pointer** to data, pointer can’t move, but data can change.      |
| `const int *const p;` | Both data and pointer are constant.                                         |

### Function Pointer

A function pointer is a variable that stores the address of a function.
Can be used to call functions dynamically or pass functions as parameters.

#### Syntax

```c
return_type (*pointer_name)(parameter_types);
```

#### Example

```c
#include <stdio.h>

void greet(void) {
printf("Hello from greet()!\n");
}

int add(int x, int y) {
return x + y;
}

int main() {

	void (*funcPtr)(void); // pointer to function returning void, no args
	funcPtr = greet; // assign function address
	funcPtr(); // call through pointer

    int (*addPtr)(int, int); // pointer to function returning int
    addPtr = add;
    printf("Sum = %d\n", addPtr(3, 4)); // call function via pointer

    return 0;

}
```

#### Typedef function pointer use case

```c
#include <stdio.h>

// Define a typedef for a function pointer type
typedef int (*MathFunc)(int, int);

// Define two functions matching the typedef
int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }

int main() {
    MathFunc func; // Declare a function pointer variable

    func = add;
    printf("Add: %d\n", func(5, 3));

    func = subtract;
    printf("Subtract: %d\n", func(5, 3));

    return 0;
}

```

## How do you multiply without using multiply or divide instructions for a multiplier constant of 10, 31, 132?

Multiplication by a constant can be rewritten using bit shifts and additions
since shifting left by n equals multiplying by

x∗2<sup>n</sup> = x<<n

So we can build other constants from combinations of powers of 2.

#### Multiply by 10

We know:

10 = 8 + 2 = 2<sup>3</sup> + 2<sup>1</sup>

So:

y = (x << 3) + (x << 1);

Equivalent to x \* 10, no multiply used.

#### Multiply by 31

31 = 32 - 1 = 2<sup>5</sup> - 2<sup>0</sup>

So:

y = (x << 5) - x;

Equivalent to x \* 31.

#### Multiply by 132

132 = 128 + 4 = 2<sup>7</sup> + 2<sup>2</sup>

So:

y = (x << 7) + (x << 2);

Equivalent to x \* 132.

### Example Program

```c
#include <stdio.h>

int main() {
int x = 5;
int y10 = (x << 3) + (x << 1); // x _ 10
int y31 = (x << 5) - x; // x _ 31
int y132 = (x << 7) + (x << 2); // x \* 132

    printf("x*10  = %d\n", y10);
    printf("x*31  = %d\n", y31);
    printf("x*132 = %d\n", y132);

    return 0;

}
```

Output:

```bash
x*10 = 50
x*31 = 155
x*132 = 660
```

### Optimization Insights

- Shifts (<<) are single-cycle on most CPUs — far faster than multiply/divide.
- Compilers often do this automatically for constant multipliers.
- You can apply similar logic for division by constants, using shifts and adds for efficient division (e.g. divide by 8 → x >> 3).

## Why is strlen() sometimes not considered "safe" in C? How to make it safer? What is the newer safer function name?

It assumes the string is null-terminated.

strlen() works by scanning memory byte by byte until it finds a '\0' (null terminator).

```c
size_t strlen(const char *str);
```

If the string is not properly null-terminated, or if a pointer doesn’t actually point to a valid string,
then strlen() will continue reading memory indefinitely — leading to:

- Buffer over-reads
- Segmentation faults
- Potential security vulnerabilities (information leaks)

How to make it safer
Option 1 — Ensure proper null termination

Always make sure every string has a terminating '\0'.
Example:

char buffer[6] = "HELLO"; // includes '\0' at end

Option 2 — Use bounded length scanning

If you receive data from an untrusted source (e.g., serial input, socket, etc.),
you should never trust it to be null-terminated.

Use strnlen() (or its secure variant below) to specify a maximum number of bytes to check.

#### Safer Alternative: strnlen()

```c
size_t strnlen(const char *str, size_t maxlen);
```

It stops scanning after maxlen bytes, even if no '\0' was found.

Example:

```c
char buffer[5] = {'H', 'E', 'L', 'L', 'O'}; // no null terminator
printf("Length = %zu\n", strnlen(buffer, sizeof(buffer))); // ✅ Safe
```

Output:

```bash
5
```

Even though no '\0' exists, it doesn’t crash — it just reports up to maxlen.

#### Newer Secure Function (C11 Annex K)

If your compiler supports the C11 bounds-checking interface (<string.h> Annex K):

```c
errno_t strcpy_s(char *dest, rsize_t destsz, const char *src);
size_t strnlen_s(const char *str, size_t maxsize);
```

- strnlen_s() is the safer, standardized version of strnlen(),
- which also returns 0 if str is NULL (preventing crashes).

## When do you use memmove() instead of memcpy() in C? Describe why?

Both memcpy() and memmove() are used to copy a block of memory,
but they differ in how they handle overlapping memory regions.

When copying overlapping regions, the order of copying matters.

- If you use memcpy() and the source and destination overlap, data can be overwritten before it’s copied — leading to corruption.
- memmove() detects overlap and adjusts the copy direction:

- If destination < source, copy forward (like memcpy()).
- If destination > source, copy backward to avoid overwriting unread data.

## When is the best time to malloc() large blocks of memory in embedded processors? Describe alternate approach if malloc() isn't available or desired to not use it, and describe some things you will need to do to ensure it safely works.

In embedded processors, malloc() should be used rarely and carefully.
But if you must use it, the best time is:

- During system initialization

  - Before the scheduler starts (in RTOS systems), or
  - Before interrupts or critical loops begin.

- Because:

  - There’s no risk of fragmentation yet.
  - You can handle allocation failures gracefully.
  - Memory needs are often known at startup (buffers, tables, etc.).

- Example:

```c
void SystemInit(void) {
buffer = malloc(BUFFER_SIZE);
	if (!buffer) {
	// Handle failure before system runs critical tasks
	}
}
```

Once the system is running, especially in real-time loops, avoid dynamic allocation — it introduces non-determinism (malloc can take variable time or fail unpredictably).

### Why malloc() is risky in embedded systems

- Fragmentation : Multiple allocations and frees of different sizes can fragment memory over time.
- Non-deterministic timing You can’t predict how long malloc() or free() will take.
- Limited memory Embedded RAM is small; allocation failure is likely.
- No OS protection A bad pointer or overflow corrupts heap or other memory regions.

### Alternate Approaches (when malloc() isn’t available or desired)

#### Static allocation (preferred)

Allocate all memory at compile time.

```c
#define BUFFER_SIZE 1024
static uint8_t buffer[BUFFER_SIZE];
```

- Simple, safe, deterministic.
- No runtime failures or fragmentation.
- Common in safety-critical and bare-metal systems.

#### Memory pools (fixed-size blocks)

If you need dynamic-like behavior but deterministic timing, use a fixed-size memory pool.

Example concept:

```c
#define BLOCK_SIZE 64
#define BLOCK_COUNT 32
static uint8_t pool[BLOCK_SIZE * BLOCK_COUNT];
static bool block_in_use[BLOCK_COUNT];

void *pool_alloc(void) {
    for (int i = 0; i < BLOCK_COUNT; i++) {
        if (!block_in_use[i]) {
            block_in_use[i] = true;
            return &pool[i * BLOCK_SIZE];
        }
    }
    return NULL; // Pool exhausted
}
void pool_free(void *ptr) {
    int index = ((uint8_t*)ptr - pool) / BLOCK_SIZE;
    block_in_use[index] = false;
}

```

- Benefits:

  - Constant allocation/free time (O(1)).
  - No fragmentation.
  - Safe for RTOS or ISR contexts (with proper protection).

#### Static buffers + custom allocator

Pre-allocate one large static buffer and manage partitions yourself.
This gives flexibility similar to malloc() but keeps everything deterministic.

```c
#define HEAP_SIZE 4096
static uint8_t heap_area[HEAP_SIZE];
static size_t heap_offset = 0;

void *simple_alloc(size_t size) {
	if (heap_offset + size > HEAP_SIZE) return NULL;
	void *ptr = &heap_area[heap_offset];
	heap_offset += size;
	return ptr;
}
```

No free(), but you can control all allocations predictably.

## Have you ever written code to initialize (configure) low-power self-refreshing DRAM memory after power up (independent of BIOS or other code that did it for the system)? It's likely that most people have never done this.

### DRAM Configuration

- Clock and power setup — enabling the DRAM controller clock and PLLs.
- Reset sequence — asserting and de-asserting DRAM reset pins.
- Mode register programming — configuring CAS latency, burst length, write recovery, and low-power self-refresh settings.
- Timing parameters — setting tRCD, tRP, tRAS, tRFC according to the DRAM datasheet.
- Self-refresh configuration — enabling low-power modes and verifying the DRAM can retain data during self-refresh.

## Write code in C to "round up" any number to the next "power of 2", unless the number is already a power of 2. For example, 5 rounds up to 8, 42 rounds up to 64, 128 rounds to 128. When is this algorithm useful?

### Bit-Twiddling Trick

- Decrement n by 1 → ensures already-powers-of-2 stay the same.
- Fill all bits to the right of the highest set bit using OR and shifts.
- Add 1 → gives the next power of 2.

This works because binary numbers double each time.

```c
#include <stdio.h>
#include <stdint.h>

uint32_t next_power_of_2(uint32_t n) {
    if (n == 0) return 1; // Edge case
    n--;                   // If already power of 2, stay the same
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

int main() {
    uint32_t nums[] = {0, 5, 42, 128, 255};
    for (int i = 0; i < 5; i++) {
        printf("%u -> %u\n", nums[i], next_power_of_2(nums[i]));
    }
    return 0;
}

```

## Implement a Count Leading Zero (CLZ) bit algorithm, but don't use the assembler instruction. What optimizations to make it faster? What are some uses of CLZ?

### The Binary search Method

#### Idea Behind the Method

Instead of checking every bit from MSB to LSB,
We do a binary search on where the first 1-bit could be:

Is the upper 16 bits all zero ?
→ Yes → Then the first 1 is in the lower 16 bits.
→ No → Then it’s in the upper 16 bits.

- Now check smaller and smaller chunks (8, 4, 2, 1 bits).
- Each test halves the possible range — just like binary search.
- That’s why we need only log₂(32) = 5 checks to find the first 1-bit.

```c

#include <stdint.h>

uint32_t clz(uint32_t x) {
    if (x == 0) return 32;
    uint32_t n = 0;
    if ((x >> 16) == 0) { n += 16; x <<= 16; }
    if ((x >> 24) == 0) { n += 8;  x <<= 8; }
    if ((x >> 28) == 0) { n += 4;  x <<= 4; }
    if ((x >> 30) == 0) { n += 2;  x <<= 2; }
    if ((x >> 31) == 0) { n += 1; }
    return n;
}

```

We can also remove the if condition with bit trick

```c
#include <stdint.h>

uint32_t clz_branchless(uint32_t x) {
    if (x == 0) return 32;

    uint32_t n = 0;
    uint32_t mask;

    mask = ((x >> 16) == 0) * 16;
    n += mask;
    x <<= mask;

    mask = ((x >> 24) == 0) * 8;
    n += mask;
    x <<= mask;

    mask = ((x >> 28) == 0) * 4;
    n += mask;
    x <<= mask;

    mask = ((x >> 30) == 0) * 2;
    n += mask;
    x <<= mask;

    mask = ((x >> 31) == 0) * 1;
    n += mask;

    return n;
}


```

### The lookup table (LUT) approach for Count Leading Zeros (CLZ).

It’s a smart trade-off between speed and memory, and it’s commonly used in embedded systems where assembly isn’t allowed but performance still matters.

#### Idea Behind the Lookup Table

Instead of checking all 32 bits, we:

- Split the 32-bit number into 8-bit chunks (bytes).
- Precompute the CLZ count for all possible 8-bit values (0–255).
- Find the first non-zero byte starting from the most significant byte (MSB).
- Use the lookup table to find its CLZ within that byte.
- Add the number of bits skipped from the higher bytes.

```c

#include <stdint.h>

uint32_t clz_lut32(uint32_t x) {
    if (x == 0) return 32;

	/* Precomputed CLZ count for all possible 8-bit values */
    static const uint8_t clz_lut[256] = {
        8,7,6,6,5,5,5,5,4,4,4,4,4,4,4,4,
        3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,
        2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
        /* ... fill up to 255 ... */
        0
    };

    uint32_t n = 0;

    if (x >= 0x01000000) {
        // MSB byte (bits 31–24)
        return clz_lut[x >> 24];
    } else if (x >= 0x00010000) {
        // Bits 23–16
        return 8 + clz_lut[(x >> 16) & 0xFF];
    } else if (x >= 0x00000100) {
        // Bits 15–8
        return 16 + clz_lut[(x >> 8) & 0xFF];
    } else {
        // Bits 7–0
        return 24 + clz_lut[x & 0xFF];
    }
}
```
