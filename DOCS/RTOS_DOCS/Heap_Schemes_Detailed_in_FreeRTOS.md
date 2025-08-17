## Memory Management in RTOS

### Introduction

In a **Real-Time Operating System (RTOS)**, memory management is critical to ensure the following:

- **Deterministic behavior** (predictable allocation/deallocation times)
- **Efficient use of limited memory** (especially in embedded systems)
- **Safety and reliability** (prevent memory leaks, fragmentation, corruption)

Memory management in RTOS is often simpler and more predictable than in general-purpose OSes,
but it must handle both **static** and **dynamic** requirements.

---

### Types of Memory in RTOS

#### Static Memory Allocation

- Memory assigned **at compile time**.
- Predictable and safe (no fragmentation risk).
- Common for:
  - Task stacks
  - Global/static variables
  - Fixed-size buffers
- Example: `static uint8_t buffer[128];`

#### Dynamic Memory Allocation

- Memory assigned **at runtime**.
- More flexible but can introduce:
  - **Heap fragmentation**
  - **Unpredictable allocation time**
- Commonly used for:
  - Variable-sized buffers
  - Temporary data structures
  - Dynamically created tasks

---

### Memory Management Models in RTOS

#### Heap-Based Allocation

- Similar to standard `malloc`/`free`, but with RTOS-specific heaps.
- FreeRTOS example: `pvPortMalloc()` and `vPortFree()`
- Different heap schemes (e.g., FreeRTOS `heap_1` to `heap_5`) balance flexibility and determinism.

#### Memory Pools (Fixed Block Allocation)

- Pre-allocated memory divided into **fixed-size blocks**.
- Constant-time allocation/deallocation (O(1)).
- Avoids fragmentation.
- Ideal for real-time constraints.
- Example: CMSIS-RTOS `osPoolCreate()`

#### Partitioned Memory

- Memory divided into regions reserved for specific tasks or modules.
- Prevents one task from exhausting all memory.

#### Stack-Based Allocation

- Each task gets a **dedicated stack**.
- Stack size defined at task creation.
- Important to monitor stack usage to avoid overflows.

---

### Memory Protection

Many RTOS implementations support **Memory Protection Units (MPUs)** to:

- Isolate task memory spaces
- Prevent faulty tasks from corrupting others
- Improve system reliability and security

---

### Memory Management Challenges in RTOS

| Challenge           | Cause                                      | Mitigation                                 |
| ------------------- | ------------------------------------------ | ------------------------------------------ |
| **Fragmentation**   | Frequent dynamic allocations/deallocations | Use fixed-size memory pools                |
| **Memory Leaks**    | Allocated memory not freed                 | Enforce coding discipline, static analysis |
| **Stack Overflow**  | Underestimating task stack size            | Enable stack overflow detection            |
| **Non-determinism** | Variable allocation time                   | Prefer pre-allocation or fixed-block pools |

---

### Best Practices

- Prefer **static allocation** for critical tasks.
- Use **memory pools** for real-time data buffers.
- Monitor stack usage during testing (`uxTaskGetStackHighWaterMark` in FreeRTOS).
- Avoid standard library dynamic allocation (`malloc`, `free`) in real-time code.
- Enable MPU (if available) for safety.
- Pre-allocate resources before entering time-critical sections.

---

### Example (FreeRTOS Memory Pool)

```c
#include "FreeRTOS.h"
#include "queue.h"

#define BLOCK_SIZE   32
#define BLOCK_COUNT  10

static uint8_t ucMemoryPool[BLOCK_SIZE * BLOCK_COUNT];
StaticQueue_t xStaticQueue;
QueueHandle_t xQueue;

void MemoryPoolExample(void) {
    xQueue = xQueueCreateStatic(
        BLOCK_COUNT,         // Number of items
        BLOCK_SIZE,          // Item size
        ucMemoryPool,        // Storage area
        &xStaticQueue        // Static queue buffer
    );

    if (xQueue != NULL) {
        // Use the memory pool as needed
    }
}
```

## FreeRTOS Memory Management

FreeRTOS provides five different heap memory management schemes that offer
varying degrees of complexity, flexibility, and performance. This guide dives
deeper into each heap management scheme with usage details and code examples.

---

### Heap_1: Minimalist Fixed Allocation

#### Overview

- Simplest implementation.
- Only supports allocation, not deallocation.
- Suitable for systems where memory usage is static.

#### Configuration

In `FreeRTOSConfig.h`:

```c
#define configFRTOS_MEMORY_SCHEME 1
#define configTOTAL_HEAP_SIZE ((size_t)(4 * 1024))
```

Ensure `heap_1.c` is included in your project.

#### Example

```c
void vTaskA(void *pvParameters) {
    while (1) {
        printf("Task A running\n");
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

int main() {
    xTaskCreate(vTaskA, "TaskA", 128, NULL, 1, NULL);
    vTaskStartScheduler();
    while (1);
}
```

---

### Heap_2: Simple Allocation with Free

#### Overview

- Adds support for freeing memory (`vPortFree()`).
- No coalescing of adjacent free blocks, so fragmentation can occur.

#### Configuration

```c
#define configFRTOS_MEMORY_SCHEME 2
#define configTOTAL_HEAP_SIZE ((size_t)(4 * 1024))
```

Include `heap_2.c`.

#### Example

```c
void *ptr = pvPortMalloc(100);
if (ptr != NULL) {
    // Use memory
    vPortFree(ptr); // Free it
}
```

---

### Heap_3: C Library Malloc/Free Wrapper

#### Overview

- Simply redirects to the standard `malloc()` and `free()`.
- Useful if your C library provides a well-optimized allocator.

#### Configuration

```c
#define configFRTOS_MEMORY_SCHEME 3
```

Include `heap_3.c`.

#### Example

No special usage—standard dynamic allocation:

```c
void *buffer = pvPortMalloc(256);
if (buffer) {
    memset(buffer, 0, 256);
    vPortFree(buffer);
}
```

---

### Heap_4: Coalescing Allocator

#### Overview

- Supports `malloc()` and `free()` with coalescing of adjacent free blocks.
- Reduces fragmentation.
- Suitable for complex, long-running systems.

#### Configuration

```c
#define configFRTOS_MEMORY_SCHEME 4
#define configTOTAL_HEAP_SIZE ((size_t)(10 * 1024))
```

Include `heap_4.c`.

#### Example

```c
void dynamicAllocationTask(void *pvParameters) {
    void *block1 = pvPortMalloc(100);
    void *block2 = pvPortMalloc(200);

    vPortFree(block1);
    vPortFree(block2); // Freed blocks are coalesced

    vTaskDelete(NULL);
}
```

---

### Heap_5: Multi-Region Support

#### Overview

- Supports multiple non-contiguous memory regions.
- Ideal for fragmented memory architectures like STM32 (SRAM, DTCM, etc.).

#### Configuration

```c
#define configFRTOS_MEMORY_SCHEME 5
```

Include `heap_5.c`.

#### Example

```c
#include "portable.h"

static uint8_t ucHeap1[1024];
static uint8_t ucHeap2[2048];

const HeapRegion_t xHeapRegions[] = {
    { ucHeap1, sizeof(ucHeap1) },
    { ucHeap2, sizeof(ucHeap2) },
    { NULL, 0 } // Terminator
};

void vSetupHeap() {
    vPortDefineHeapRegions(xHeapRegions);
}

void appMain() {
    vSetupHeap();
    void *data = pvPortMalloc(500); // Can be from either region
    vPortFree(data);
}
```

---

### Heap Scheme Summary

| Scheme | Free Support | Coalescing | Multi-Region | Notes                      |
| ------ | ------------ | ---------- | ------------ | -------------------------- |
| heap_1 | ❌           | ❌         | ❌           | Static allocation only     |
| heap_2 | ✅           | ❌         | ❌           | Basic, may fragment        |
| heap_3 | ✅           | C library  | ❌           | Uses `malloc()`/`free()`   |
| heap_4 | ✅           | ✅         | ❌           | Efficient and flexible     |
| heap_5 | ✅           | ✅         | ✅           | Best for fragmented memory |

---

### Conclusion

Choosing the right heap scheme depends on your system's needs:

- Use `heap_1` for static applications with tight memory.
- Use `heap_4` or `heap_5` for advanced applications with dynamic tasks and fragmentation concerns.
- Monitor heap usage with `xPortGetFreeHeapSize()` and `xPortGetMinimumEverFreeHeapSize()` to optimize memory over time.
