## Overview

Following is a program that demonstrates how to perform vector addition (adding two arrays element by element)
using CUDA, NVIDIA’s technology for parallel programming on GPUs (Graphics Processing Units).

Normally, programs run on the CPU — the main processor of your computer.
CUDA allows parts of the program (called kernels) to run on the GPU, which can perform many calculations simultaneously, making it much faster for large data.

## Header Section

```c
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
```

- These are header files that include predefined CUDA and C functions.
- cuda_runtime.h: Contains functions for memory management and kernel launching.
- device_launch_parameters.h: Contains definitions related to how kernels are launched on the GPU.
- stdio.h and stdlib.h: Standard C libraries for input/output and memory allocation.

## Defining the Vector Size

```c
#define SIZE 1024
```

- This defines a constant SIZE = 1024, meaning each vector (array) will contain 1024 elements.
- The keyword #define creates a macro, which is replaced everywhere in the code during compilation (Standard C Drill).

## CUDA Kernel Function

```c
__global__ void vectorAdd(int* A, int* B, int* C, int size) {
int i = threadIdx.x;
C[i] = A[i] + B[i];
}
```

- This is the heart of CUDA programming.

### What is a Kernel?

- A kernel is a function that runs on the GPU, not on the CPU.
- It is marked by the special keyword \_\_global\_\_.

### How it works:

- threadIdx.x gives the thread’s unique index within a block.
- CUDA launches many threads (lightweight tasks) at once.
- Each thread computes one element of the result:

```c
C[i] = A[i] + B[i]
```

So if there are 1024 elements, there are 1024 threads, each doing one addition.

> Each thread runs this same function in parallel; this is what makes GPU computation powerful.

## Helper Function to Compare Results

```c
bool compareBuffer(int* a, int* b, int size) {
	for (int i = 0; i < size; i++) {
		if (a[i] != b[i]) {
		return false;
		}
	}
	return true;
}
```

- This function checks if two arrays contain the same values.
- It’s used later to confirm whether the GPU and CPU calculations match.

## Main Function

Every CUDA program, like a C program, starts with main().

```c
int main() {
```

### Step 1: Declare Variables

```c
int *A, *B, *C, *C*CPU;
int *d*A, \_d_B, \_d_C;
int size = SIZE * sizeof(int);
```

#### Host variables (CPU memory):

- A, B: Input arrays.
- C: Output from GPU.
- C_CPU: Output from CPU (for verification).

#### Device variables (GPU memory):

- d_A, d_B, d_C: Corresponding arrays in GPU memory.
- size = total number of bytes for each array (number of elements × size of each element).

### Step 2: Allocate CPU Memory

```c
A = (int*)malloc(size);
B = (int*)malloc(size);
C = (int*)malloc(size);
C_CPU = (int*)malloc(size);
```

- The malloc() function reserves memory for arrays in CPU RAM.

### Step 3: Allocate GPU Memory

```c
cudaMalloc((void**)&d_A, size);
cudaMalloc((void**)&d_B, size);
cudaMalloc((void\*\*)&d_C, size);

```

- cudaMalloc() works like malloc() but for GPU memory.
- It allocates space in the GPU’s global memory.

### Step 4: Initialize Input Data on CPU

```c
for (int i = 0; i < SIZE; i++) {
	A[i] = i;
	B[i] = SIZE - i;
}
```

- Fills the input arrays:
- A = [0, 1, 2, 3, ...]
- B = [1024, 1023, 1022, ...]

- So each pair sums to the same value: A[i] + B[i] = SIZE.

### Step 5: Copy Data to GPU

```c
cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
```

- Transfers data from the CPU memory (Host) to GPU memory (Device).
- cudaMemcpyHostToDevice indicates the direction of the copy.

### Step 6: Launch the CUDA Kernel

```c
vectorAdd <<<1, 1024 >>> (d_A, d_B, d_C, SIZE);

```

#### Syntax:

- Here, <<<1, 1024>>> means:

  - 1 block of threads.
  - 1024 threads inside that block.
  - Each of the 1024 threads adds one pair of numbers from A and B.

- This line tells the GPU:

  - “Run the vectorAdd function in parallel using 1024 threads.”

### Step 7: Synchronize CPU and GPU

```c
cudaDeviceSynchronize();
```

- Ensures the CPU waits until all GPU operations are finished.
- Without this, the CPU might continue before the GPU completes its task.

### Step 8: Copy Results Back to CPU

```c
cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
```

- Transfers the computed results from GPU memory back to CPU memory.
- Now C on the CPU contains the output from the GPU.

### Step 9: Compute on CPU for Comparison

```c
for (int i = 0; i < SIZE; i++) {
	C_CPU[i] = A[i] + B[i];
}
```

- Performs the same vector addition using the CPU for verification.
- Much slower than GPU for large data, but useful for checking correctness.

### Step 10: Verify the Results

```c
printf("%s", compareBuffer(C, C_CPU, SIZE) ? "Calculation Correct !!\n" : "Error !!\n");
```

- Calls compareBuffer() to compare the GPU and CPU results.
- Prints “Calculation Correct !!” if they match, or “Error !!” if not.

### Step 11: Free All Allocated Memory

```c
free(A);
free(B);
free(C);
free(C_CPU);
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
```

- free() releases CPU memory.
- cudaFree() releases GPU memory.

!>Always free memory after use to avoid memory leaks.

## Output Example

When run, the expected output is:

```bash
Calculation Correct !!
```

- This means both GPU and CPU produced the same result — the CUDA kernel worked correctly.

## The Entire Program

```c
/* ==================================================================================
# A simple Vector Addition Example with CUDA
===================================================================================*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 1024

/* Kernel Function : To be executed in GPU not CPU */
__global__ void vectorAdd(int* A, int* B, int* C, int size) {

	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

/* Function Compare Buffers */
bool compareBuffer(int* a, int* b, int size) {
	for (int i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			return false;
		}
	}
	return true;
}

/* CUDA C always requires a main Function */
int main() {

	/* Declare CPU side and GPU side variables for the Operations */
	int * A, * B, * C, * C_CPU;
	int *d_A, *d_B, *d_C;
	int size = SIZE * sizeof(int);

	/* Allocate Memory for CPU side vaiables */
	A = (int*)malloc(size);
	B = (int*)malloc(size);
	C = (int*)malloc(size);
	C_CPU = (int*)malloc(size);

	/* Allocate memory for GPU side variables */
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	/* Fill data in the variabels Host side */
	for (int i = 0; i < SIZE; i++) {
		A[i] = i;
		B[i] = SIZE - i;
	}

	/* Copy data from host side (CPU Memory) to device side (GPU Memory) */
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	/* Launch The vectorAdd CUDA Kernel */
	vectorAdd <<<1, 1024 >>> (d_A, d_B, d_C, SIZE);

	/* Halt the CPU thread until all preceding GPU operations (kernels, memory transfers, etc.)
	   are finished on the Entire Device (GPU) */
	cudaDeviceSynchronize();

	/* Copy the results from GPU memory to CPU Memory */
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	/* Do the same Operation using the CPU */
	for (int i = 0; i < SIZE; i++) {
		C_CPU[i] = A[i] + B[i];
	}

	printf("%s", compareBuffer(C, C_CPU, SIZE) ? "Calculation Correct !!\n" : "Error !!\n");

	/* Free All the Memories used */
	free(A);
	free(B);
	free(C);
	free(C_CPU);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
```

## This basic workflow of any CUDA program:

- Allocate memory on both CPU and GPU.
- Copy input data from CPU → GPU.
- Launch the GPU kernel with threads.
- Synchronize to wait for GPU completion.
- Copy results back from GPU → CPU.

Once understood, more complex CUDA applications can be built for image processing, deep learning, and scientific computing.
