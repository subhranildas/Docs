## What Are FLOPs

FLOPs stand for Floating Point Operations per Second.

- Floating-point operations are mathematical operations (like addition, subtraction, multiplication, division) performed on floating-point numbers.
- FLOPs as a metric represent how many such operations a system can perform in one second.

Example:

- If a processor performs 1 million floating-point additions in 0.001 seconds,

\[
FLOPs = \frac{10^6}{10^{-3}} = 10^9 \text{ FLOPs} = 1 \text{ GFLOP}
\]

### Common Scales of FLOPs

| Unit    | Value                 | Typical Usage Example  |
| ------- | --------------------- | ---------------------- |
| 1 GFLOP | \(10^9\) FLOPs/sec    | Mobile GPUs, CPUs      |
| 1 TFLOP | \(10^{12}\) FLOPs/sec | High-end GPUs, servers |
| 1 PFLOP | \(10^{15}\) FLOPs/sec | Supercomputers         |
| 1 EFLOP | \(10^{18}\) FLOPs/sec | Exascale computing     |

FLOPs measure raw compute capability, but raw FLOPs alone don’t guarantee speedup—because memory bandwidth often limits performance.

## Arithmetic Intensity (AI)

**Arithmetic Intensity (AI)** is defined as:

\[
AI = \frac{\text{Number of Floating-Point Operations}}{\text{Memory Bytes Accessed}}
\]

It tells us **how much computation is performed per byte of data moved** from memory.

- **High AI** → Program is compute-bound (performance limited by FLOPs).
- **Low AI** → Program is memory-bound (performance limited by memory bandwidth).

### Example: Vector Addition

```c
for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
}
```

- FLOPs = 1 addition per iteration
- Memory operations per iteration: 2 loads (a[i], b[i]) + 1 store (c[i]) = 3 \* sizeof(float) = 12 bytes

- AI = 1 / 12 ≈ 0.083 FLOPs/byte → memory-bound

### Example: Matrix Multiplication

```c
void dgemm(size_t n, const double *A, const double *B, double *C) {

    size_t i, j, k;
    double sum;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            sum = 0.0;
            for (k = 0; k < n; k++) {
                sum = sum + A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}
```

#### FLOPs Count

Inner loop body:

sum = sum + A[i*n + k] * B[k*n + j]; → 1 multiplication + 1 addition = 2 FLOPs

#### Memory Access

Each array element is a double (8 bytes).

- A: size n², ideally loaded once → n² loads
- B: size n², ideally loaded once → n² loads
- C: size n², usually read + write → 2 \* n² accesses

Total = 4 \* n² elements × 8 bytes = 32 n² bytes

#### Arithmetic Intensity (AI)

- AI = (FLOPs) / (Bytes moved) ​= ( 2*n<sup>3</sup> ) / ( 32*n<sup>2</sup> )​
- AI = n / 16

## The Roofline Model

The Roofline Model connects FLOPs and Arithmetic Intensity:

Attainable Performance = min ( Peak FLOPs, 𝐴𝐼 × Memory Bandwidth )

> AI×Bandwidth < Peak FLOPs → memory-bound

> AI×Bandwidth > Peak FLOPs → compute-bound

This model helps identify whether optimization should target reducing memory traffic or maximizing compute usage.

## Amdahl's Law

Amdahl's Law is used to find the theoretical maximum speedup of a program when part of it can be parallelized across multiple processors.

The formula is:

S(N) = 1 / [ (1-P) + P/N ]

Where:

- \(S(N)\) = speedup with \(N\) processors
- \(P\) = fraction of the program that **can** be parallelized
- \(1-P\) = fraction of the program that is **serial** (cannot be parallelized)
- \(N\) = number of processors (or cores)

### Example: Intel Xeon Phi with 61 cores

- \(1\%\) of the runtime is **serial** → \(1-P = 0.01\)
  So, \(P = 0.99\)
- \(N = 61\) cores

Applying the formula:

- S(61) = {1} / {0.01 + {0.99}/{61}} = 38.125
