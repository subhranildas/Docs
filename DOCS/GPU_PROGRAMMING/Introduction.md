## A Gentle Introduction to GPUs

Following gives a quick history of GPUs, why they are widely used in deep learning, and how they differ from CPUs and other specialized hardware.

## Hardware Comparison

### CPU (Central Processing Unit)

- General-purpose processor
- High clock speed
- Few powerful cores
- Large cache memory
- Low latency, low throughput

### GPU (Graphics Processing Unit)

- Specialized for parallel workloads
- Lower clock speed than CPU
- Many smaller cores
- Smaller cache
- High latency, high throughput

### TPU (Tensor Processing Unit)

- Google’s custom hardware for Deep learning
- Optimized for Tensor operations (e.g., Matrix Multiplication)
- Designed for AI Workloads

### FPGA (Field Programmable Gate Array)

- Reconfigurable hardware for specific tasks
- Extremely low latency
- Very high throughput
- High power consumption & high cost

## A Brief History of NVIDIA GPUs

?> For a deeper dive, watch: _**<a href="https://www.youtube.com/watch?v=kUqkOAU84bA" download>NVIDIA GPU History</a>**_

?> A timeline of Innovation(NVIDIA): _**<a href="https://www.nvidia.com/en-us/about-nvidia/corporate-timeline/" download>NVIDIA History</a>**_

?> For CUDA compute capabilities across generations visit: _**<a href="https://developer.nvidia.com/cuda-gpus" download>CUDA GPU Compute Capability</a>**_

## Why GPUs Excel at Deep Learning

![Von Neumann Architecture](Images/cpu-vs-gpu.png)

- Massive Parallelism
- High Memory Bandwidth
- Optimized for Matrix Math
- Mature Software Ecosystem

## CPU (Host)

- Optimized to minimize the time of a single task
- Performance metric → Latency (seconds per task)

## GPU (Device)

- Optimized to maximize parallel execution
- Performance metric → Throughput (tasks per second)

?> In deep learning, we often need to process millions of small operations at once — GPUs shine here.

## Anatomy of a Typical CUDA Program

- CPU allocates memory on the host.
- CPU copies data to the GPU device.
- CPU launches a kernel (the computation runs on GPU).
- GPU completes processing → results copied back to CPU.

### Puzzle Analogy

#### Think of solving a jigsaw puzzle:

- CPU approach → solve piece by piece, sequentially
- GPU approach → many people place multiple pieces in parallel
- As long as no one interferes with others, the puzzle is solved much faster

## Key Terminology

- Kernels → GPU functions executed on the device (not to be confused with Linux or convolution kernels)
- Threads, Blocks, and Grids → **CUDA’s** way of organizing parallelism
- GEMM → General Matrix Multiplication
- SGEMM → Single-precision (fp32) General Matrix Multiplication
- Host (CPU) → Runs regular functions
- Device (GPU) → Runs Kernels

## Summary

- CPUs are latency-optimized (great for sequential, complex logic)
- GPUs are throughput-optimized (Ideal for Parallel Tasks like matrix operations)
- TPUs and FPGAs are even more specialized hardware for AI
- **CUDA** enables CPUs (host) and GPUs (device) to collaborate efficiently
