## What is an RTOS?

An **RTOS** (Real-Time Operating System) is a type of operating system designed
to handle **real-time tasks**/operations that must be executed within
**strict time constraints**.

> It focuses on **predictability and responsiveness**, ensuring that
high-priority tasks are executed within deterministic time.

**Real Time Operating Systems** are commonly used in **embedded systems** such as automotive ECUs,
medical devices, and IoT products — where **Timing is critical**.


## RTOS vs. GPOS

| **Feature**            | **RTOS (Real-Time OS)**                          | **GPOS (General-Purpose OS)**                 |
|------------------------|--------------------------------------------------|-----------------------------------------------|
|   Goal                 | Deterministic, time-critical task execution      | Maximum throughput, multitasking, convenience |
|   Task Scheduling      | Priority-based, deterministic                    | Fair or round-robin scheduling                |
|   Response Time        | Predictable, low-latency                         | Variable and non-deterministic                |
|   Interrupt Handling   | Very fast, low overhead                          | Slower, handled through complex subsystems    |
|   Footprint            | Lightweight, minimal features                    | Heavyweight, includes drivers, GUI, etc.      |
|   Use Case             | Embedded systems, robotics, real-time control    | PCs, servers, smartphones                     |
|   Examples             | FreeRTOS, Zephyr, VxWorks, Micrium OS            | Windows, Linux, macOS, Android                |


## Real-Life Analogy

- **RTOS** is like a factory manager who ensures every task is done **on time**, in a **fixed sequence**, with **no room for delay**.
- **GPOS** is like a city mayor — trying to keep everything running, but not guaranteeing when each thing will happen.


## When to use What

<!-- tabs:start -->

#### **GPOS**

- You need complex features (like a full GUI, networking stack).
- Timing isn’t ultra-critical.
- You have plenty of memory and processing power.

#### **RTOS**

- Your system must react within a guaranteed time.
- You need consistent task execution (e.g., 10ms cycle).
- Power consumption and resources are limited.

<!-- tabs:end -->

## What is FreeRTOS?

**FreeRTOS** (Free Real-Time Operating System) is an open-source, lightweight
RTOS designed for embedded systems. Developed by Richard Barry and now
maintained by Amazon Web Services (AWS), it's widely used in microcontrollers,
IoT devices, and industrial systems for providing deterministic task scheduling
and multitasking capabilities.

It’s written in **C**, easy to port, and supports a large number of hardware
platforms.

## Common Use Cases
- **IoT Devices** (sensors, smart locks, wearables)
- **Automotive Control Systems**
- **Industrial Automation**
- **Medical Devices**
- **Consumer Electronics**


## FreeRTOS Alternatives

A curated list of open-source and commercial real-time operating systems (RTOS) that can be used instead of FreeRTOS, depending on your project's requirements.


### Open-Source RTOS Alternatives

<!-- tabs:start -->


#### **Zephyr RTOS**

- **Backed by**: Linux Foundation
- **Language**: C
- **Key Features**:
  - Scalable, modular architecture
  - Secure (MPU, ARM TrustZone support)
  - Native support for BLE, networking, filesystems
  - Supports multiple architectures (ARM, RISC-V, x86, etc.)
- **Use Case**: IoT, wearables, edge devices
- **Website**: [https://www.zephyrproject.org](https://www.zephyrproject.org)

#### **Mbed OS**

- **Backed by**: Arm
- **Language**: C++
- **Key Features**:
  - Integrated TLS/crypto stack
  - Real-time kernel, threads, sync
  - Optimized for ARM Cortex-M
- **Use Case**: Secure IoT with Arm chips
- **Website**: [https://os.mbed.com](https://os.mbed.com)

#### **RIOT OS**

- **Backed by**: Community
- **Language**: C
- **Key Features**:
  - Ultra-lightweight for IoT
  - Supports IPv6, 6LoWPAN, RPL, CoAP
  - Real-time, multithreaded kernel
- **Use Case**: Wireless sensor networks
- **Website**: [https://www.riot-os.org](https://www.riot-os.org)


#### **ChibiOS/RT**

- **Language**: C
- **Key Features**:
  - Preemptive real-time kernel
  - Compact and efficient
  - Built-in HAL and driver framework
- **Use Case**: Robotics, industrial control
- **Website**: [http://www.chibios.org](http://www.chibios.org)

#### **NuttX**

- **Backed by**: Apache Incubator
- **Language**: C
- **Key Features**:
  - POSIX-compliant
  - Unix-like features (file systems, networking, shell)
  - Lightweight and configurable
- **Use Case**: Drones, IoT, embedded Unix-style systems
- **Website**: [https://nuttx.apache.org](https://nuttx.apache.org)

<!-- tabs:end -->



### Commercial RTOS Alternatives

<!-- tabs:start -->

#### **ThreadX (Azure RTOS)**
- **Owned by**: Microsoft
- **Key Features**:
  - Fast, deterministic kernel
  - Components: GUIX, NetX, FileX
  - Integrated with Azure IoT
- **Use Case**: Medical, industrial, consumer electronics
- **Website**: [https://learn.microsoft.com/en-us/azure/rtos/](https://learn.microsoft.com/en-us/azure/rtos/)


#### **QNX**
- **Owned by**: BlackBerry
- **Key Features**:
  - Microkernel architecture
  - POSIX-compliant
  - Safety-certified (e.g., ISO 26262 ASIL-D)
- **Use Case**: Automotive, medical, aerospace
- **Website**: [https://blackberry.qnx.com](https://blackberry.qnx.com)


#### **VxWorks**
- **Owned by**: Wind River
- **Key Features**:
  - Mature, safety-critical RTOS
  - POSIX-compliant, multicore support
  - Real-time performance
- **Use Case**: Aerospace, industrial automation
- **Website**: [https://www.windriver.com/products/vxworks](https://www.windriver.com/products/vxworks)

<!-- tabs:end -->


### Some Honorable Mentions

<!-- tabs:start -->

#### **TinyOS**

For ultra-low-power wireless sensor networks

#### **AliOS Things**

Lightweight RTOS from Alibaba for IoT

#### **Amazon RTOS**

AWS-enhanced FreeRTOS for cloud integration

<!-- tabs:end -->

