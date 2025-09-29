## Prerequisites

Before installing CUDA, make sure your system meets these requirements:

- **Laptop GPU** → NVIDIA RTX 3050 (supports CUDA Compute Capability 8.6)
- **Operating System** → Windows 10/11 (64-bit) or Ubuntu 20.04/22.04 (recommended for Linux users)
- **RAM** → At least 8 GB (16 GB preferred)
- **Storage** → ~5–10 GB free for CUDA toolkit and drivers
- **Compiler** → GCC (Linux) or MSVC (Windows)

## Install NVIDIA Drivers and support Drivers

### On Windows

1.  Go to [NVIDIA Drivers Download](https://www.nvidia.com/Download/index.aspx).
2.  Select:
    - **Product Series**: GeForce RTX 30 Series (Laptop)
    - **Product**: GeForce RTX 3050 Laptop GPU
    - **OS**: Windows 10/11 64-bit
3.  Download and install the **Game Ready Driver (GRD)** or **Studio Driver**.
4.  Install Microsoft Visual Studio Build Tools:

    - cl.exe is part of Microsoft Visual Studio. If you don't have Visual Studio or the Build Tools installed, download and install them from the official Microsoft website.
      During installation, ensure you select the "Desktop development with C++" workload, as this includes the necessary C++ compiler components.
      Locate cl.exe:
      After installation, cl.exe is typically found within a path similar to:

          C:\Program Files (x86)\Microsoft Visual Studio\<version>\VC\Tools\MSVC\<toolset_version>\bin\HostX64\x64

    - Add cl.exe's path to the System PATH Environment Variable.

5.  Reboot after installation.

### On Linux (Ubuntu)

```bash
sudo apt update
sudo ubuntu-drivers autoinstall
```

Verify driver installation:

```bash
nvidia-smi
```

You should see your RTX 3050 listed with driver and CUDA version.

## Install CUDA Toolkit

### Download

- Visit [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads).
- Select your OS, architecture, and version (e.g., CUDA 12.x).

### Windows Installation

1. Run the installer → choose **Express Installation**.
2. Ensure CUDA Toolkit, Drivers, and Nsight tools are selected.
3. Add CUDA paths manually if needed:
   - Add to **Environment Variables → Path**:
     ```
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\libnvvp
     ```

### Ubuntu Installation

```bash
# Example for CUDA 12.2
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.54.03_linux.run
sudo sh cuda_12.2.2_535.54.03_linux.run
```

!> Version might be different, follow the Download link above for accurate command.

Update environment variables:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Verify CUDA Installation

Run:

```bash
nvcc --version
```

You should see the installed CUDA version.

Also check GPU status:

```bash
nvidia-smi
```

---

## Install cuDNN (Optional, for Deep Learning)

1. Go to [NVIDIA cuDNN Download](https://developer.nvidia.com/cudnn).
2. Sign in with NVIDIA Developer account.
3. Download version matching your CUDA installation.
4. Extract and copy files into CUDA directories:

Linux:

```bash
tar -xzvf cudnn-*-linux-x64-v*.tgz
sudo cp cuda/include/* /usr/local/cuda/include/
sudo cp cuda/lib64/* /usr/local/cuda/lib64/
```

Windows → Copy `bin`, `include`, and `lib` files into corresponding CUDA folders.

---

## Write & Run Your First CUDA Program

Create a simple **vector addition program**:

```cpp
// vector_add.cu
#include <stdio.h>

__global__ void add(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    int n = 16;
    int size = n * sizeof(int);
    int h_a[16], h_b[16], h_c[16];

    for (int i = 0; i < n; i++) { h_a[i] = i; h_b[i] = i * 2; }

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    add<<<1, 16>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
```

Compile & run:

```bash
nvcc vector_add.cu -o vector_add
./vector_add
```

## Next Steps

- Learn about **CUDA kernels, threads, blocks, and grids**.
- Try running **cuBLAS** and **cuDNN** libraries.
- Explore **PyTorch** or **TensorFlow GPU versions** for deep learning.

## Quick Reference Commands

```bash
nvidia-smi          # GPU status
nvcc --version      # CUDA version
deviceQuery         # Check CUDA samples
bandwidthTest       # Benchmark GPU memory
```
