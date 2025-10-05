## What is GPU Occupancy

**Occupancy** in CUDA refers to how effectively the GPU's **Streaming Multiprocessors (SMs)** are utilized.
It is defined as the **ratio of active warps per SM** to the **maximum possible warps per SM**.

In simpler terms, it measures how many threads are running **concurrently** on a GPU compared to its **hardware capability**.

> **Occupancy = (Active Warps per SM) / (Maximum Warps per SM)**

A higher occupancy means the GPU can **hide memory latency better**, leading to improved performance; up to a certain point.

## GPU Execution Model Refresher

Before understanding occupancy, recall that:

- Each **GPU Device** has multiple **Streaming Multiprocessors (SMs)**.
- Each **SM** can handle multiple **thread blocks**.
- Each block contains multiple **threads**, and threads are grouped into **warps** (usually 32 threads per warp).

Occupancy depends on **how many blocks and warps** can be resident (active) on an SM at the same time.

## Theoretical Occupancy

**Theoretical Occupancy** is the _maximum possible occupancy_ based purely on hardware limits and kernel launch configuration — **without considering real-time scheduling or memory behavior**.

> **Theoretical Occupancy = Active Warps per SM (based on resources) / Max Warps per SM (hardware limit)**

### How to Calculate

1. **Find hardware limits from `cudaDeviceProp`:**

   - `maxThreadsPerMultiProcessor`
   - `maxThreadsPerBlock`
   - `warpSize`
   - `regsPerBlock`, `sharedMemPerBlock`, etc.

2. **Compute warps per block:**

> Warps per Block = Threads per Block / Warp Size

3. **Determine resource limits:**

   - Each block consumes **registers**, **shared memory**, and **thread slots**.
   - Calculate how many blocks can fit on one SM before any resource runs out.

4. **Compute active warps:**

   Active Warps per SM = Blocks per SM \* Warps per Block

5. **Compare to the hardware maximum:**

> Occupancy = Active Warps per SM / Max Warps per SM

### Example

Suppose:

- Warp size = 32
- Maximum threads per SM = 1536
- Maximum threads per block 1024

Now if we run an application with lets say 1024 threads in the configuration

- 32 block with 320 threads per block

  - Then ( Active Warps / Block ) = ( Threads / Block ) / Warp Size or (Threads / Warp) = 384 / 32 = 12
  - We can calculate ( Blocks / SM ) = ( Thread / SM ) / ( Thread / Block ) = 1536 / 320 = 4
  - Therefore ( Active Warps / SM ) = ( Active Warps / Block ) _ ( Blocks / SM ) = 12 _ 4 = 48

  - Maximum ( Maximum Warps / SM ) = ( Max Threads / SM ) / Warp Size or (Threads / Warp) = 1536 / 32 = 48

  - Therefore Occupancy = ( Active Warps / SM ) / ( Max Warps / SM ) = 48 / 48 = 1 = 100%

## Practical Occupancy (Measured)

**Practical Occupancy** refers to the _actual_ occupancy achieved **during runtime**, considering real-world constraints such as:

- Memory bandwidth usage
- Control flow divergence
- Instruction dependencies
- Runtime scheduling

### How to Measure in Practice

We can use CUDA tools to measure actual occupancy:

| **Tool** **Description**                              |                                                                                                                              |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **`nvprof`**                                          | Command-line CUDA profiler (deprecated but still used in legacy systems).                                                    |
| **Nsight Compute**                                    | NVIDIA’s modern profiling tool that reports achieved occupancy, SM utilization, and warp efficiency.                         |
| **`cudaOccupancyMaxActiveBlocksPerMultiprocessor()`** | Runtime API function that estimates the theoretical maximum number of active blocks per SM for a given kernel configuration. |

### Using CUDA API for Estimation

We can estimate occupancy in our code like this:

```cpp
int numBlocksPerSM = 0;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocksPerSM,
    myKernel,
    threadsPerBlock,
    sharedMemPerBlock
);

float occupancy = (numBlocksPerSM * threadsPerBlock) /
                  (float)deviceProp.maxThreadsPerMultiProcessor;
```

This gives an estimated occupancy ratio (0–1).

## Performance Measurement Using CUDA Events

To measure the performance of a CUDA kernel, we can use CUDA events, which allow us to record timestamps on the GPU.
This gives high-resolution timing for our kernel execution.

### Example Application for performance Analysis

```c
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(int* A, int* B, int* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main() {
    int N = 1 << 20;
    int *A, *B, *C;
    cudaMallocManaged(&A, N*sizeof(int));
    cudaMallocManaged(&B, N*sizeof(int));
    cudaMallocManaged(&C, N*sizeof(int));

    // Initialize arrays
    for(int i=0;i<N;i++){ A[i]=i; B[i]=i; }

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Launch kernel
    int threads = 256;
    int blocks = (N + threads - 1)/threads;
    vectorAdd<<<blocks, threads>>>(A, B, C, N);

    // Record stop event
    cudaEventRecord(stop);

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %f ms\n", milliseconds);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

```

- cudaEventCreate() → Creates events for timing.
- cudaEventRecord(event) → Records a timestamp on the GPU.
- cudaEventSynchronize(stop) → Ensures the kernel has finished.
- cudaEventElapsedTime(&ms, start, stop) → Computes elapsed time in milliseconds.

?> Using this process we can measure only kernel execution excluding CPU-GPU memory transfer unless we wrap them in events.
The measurements may vary due to other overheads and due to cold start situations and variable clock speeds.

## Performance Measurement Using NVIDIA Tools

Nsight Compute CLI:

```bash
ncu --target-processes all ./<cuda_application>
```

Gives detailed kernel timing, occupancy, memory throughput, etc.

## Performance Insight

Occupancy Range Performance Impact:

- 0–30% Very low utilization — GPU underused, expect poor performance.
- 30–60% Moderate occupancy — acceptable for memory-bound kernels.
- 60–100% High occupancy — usually ideal for compute-heavy kernels, but diminishing returns may appear beyond ~80%.

> Note: Higher occupancy does not always mean higher performance.
> Too many active threads can increase register pressure or shared memory contention.

## Conclusion

- Theoretical occupancy helps us design efficient kernels,
- Practical occupancy helps us verify if our kernel performs efficiently on the hardware.
