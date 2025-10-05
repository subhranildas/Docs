## Introduction

When we write CUDA programs, we interact with NVIDIA GPUs through a software interface provided by CUDA.

This interface consists of two parts:

Driver API – Low-level, complex control (used in advanced cases).
Runtime API – High-level, user-friendly functions for most developers.

CUDA Runtime APIs, which allow the C/C++ program to:

- Detect and query GPU devices.
- Retrieve detailed hardware specifications.
- Manage and optimize GPU usage dynamically.

## CUDA Runtime Environment

The CUDA Runtime is automatically initialized when we run a CUDA application.
This means we don’t have to manually open or manage GPU contexts (unlike the Driver API).

It provides direct access to GPU features through functions such as:

- cudaGetDeviceCount() – Find how many CUDA-capable GPUs are available.
- cudaGetDeviceProperties() – Retrieve information about each GPU.
- cudaSetDevice() – Choose which GPU to use (in multi-GPU systems).
- cudaGetDevice() – Check which GPU is currently being used.

These functions are defined in the CUDA header files:

```c
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
```

## Detecting CUDA Devices

Before running any computation, the program must first detect available GPUs.

This is done using:

```c
int count;
cudaGetDeviceCount(&count);
```

- The argument is a pointer to an integer (&count).
- After the function executes, count will store the number of GPUs detected.
- If count is 0, it means no CUDA-compatible GPU was found.

Example output:

```bash
Number of GPU Devices found: 1
```

## Querying GPU Properties

Once we know that GPUs exist, we can extract device properties using:

```c
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
```

- prop is a structure of type cudaDeviceProp.

- The second parameter (0) refers to the device index (0 = first GPU).

- The structure cudaDeviceProp contains dozens of hardware attributes, such as:

  - totalGlobalMem: Amount of global memory on the device (in bytes)
  - sharedMemPerBlock: Shared memory per thread block
  - regsPerBlock: Number of registers available per block
  - warpSize: Number of threads in a warp (usually 32)
  - maxThreadsPerBlock: Maximum number of threads a block can contain
  - multiProcessorCount: Number of Streaming Multiprocessors (SMs)
  - l2CacheSize: Size of the Level 2 cache
  - clockRate Core clock speed of the GPU (in kHz)
  - major / minor CUDA Compute Capability version
  - maxThreadsDim[] Max thread dimensions per block (x, y, z)
  - maxGridSize[] Max grid dimensions (x, y, z)

These fields help us understand the hardware limits when designing our CUDA kernels.

### Example: Reading Key GPU Specifications

Once populated, you can print information from the structure:

```c
printf("CUDA Compute Capability: %d.%d\n", prop.major, prop.minor);
printf("Total Global Memory (GB): %.2f\n", (double)prop.totalGlobalMem / (1024.0 \* 1024.0 \* 1024.0));
printf("Multiprocessors: %d\n", prop.multiProcessorCount);
printf("Warp Size: %d\n", prop.warpSize);
```

#### Explanation:

- Compute Capability: Indicates which CUDA features are supported by the hardware.
- Global Memory: Total VRAM available to CUDA programs.
- Multiprocessors: More SMs → higher parallelism.
- Warp Size: Fixed group size of threads that execute together.

## CUDA Hardware Hierarchy

CUDA hardware is organized in a hierarchical structure:

| **Level**             | **Description**                                                                                                             |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `Device`              | The entire GPU (physical hardware that executes all CUDA operations). A computer may have one or more CUDA-capable devices. |
| `Multiprocessor (SM)` | Short for _Streaming Multiprocessor_. Each SM contains many cores that execute groups of threads in parallel.               |
| `Block`               | A group of threads that can **cooperate and share memory** using _shared memory_. Each block runs on a single SM.           |
| `Thread`              | The **smallest unit of execution** in CUDA. Each thread performs computations on individual data elements.                  |

- APIs like maxThreadsPerBlock, maxThreadsDim[], and maxGridSize[] tell us the limits for these levels.

### Example:

```c
printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Block dimension limits: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
printf("Grid dimension limits: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
```

- This helps developers design thread layouts that fit within hardware limits.

## Some Other Useful Runtime APIs

Here are a few additional CUDA Runtime APIs often used in more complex applications:

| **Function**                            | **Purpose**                                      |
| --------------------------------------- | ------------------------------------------------ |
| `cudaSetDevice(int device)`             | Selects a specific GPU to use.                   |
| `cudaGetDevice(int *device)`            | Returns the currently selected GPU.              |
| `cudaDeviceSynchronize()`               | Waits for all GPU operations to finish.          |
| `cudaDeviceReset()`                     | Resets the GPU state and releases all resources. |
| `cudaGetErrorString(cudaError_t error)` | Returns a human-readable error message.          |

These are essential for managing and debugging GPU behavior in runtime.

> Follow [Link](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html) to NVIDIA's official Documentation on runtime APIs.
