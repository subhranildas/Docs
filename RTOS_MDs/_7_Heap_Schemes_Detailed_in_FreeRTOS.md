## 🔍  FreeRTOS Memory Management: In-Depth Guide to Heap Schemes

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

| Scheme   | Free Support  | Coalescing  |  Multi-Region  | Notes                            |
|----------|---------------|-------------|----------------|----------------------------------|
| heap_1   | ❌            | ❌          | ❌             | Static allocation only           |
| heap_2   | ✅            | ❌          | ❌             | Basic, may fragment              |
| heap_3   | ✅            | C library   | ❌             | Uses `malloc()`/`free()`         |
| heap_4   | ✅            | ✅          | ❌             | Efficient and flexible           |
| heap_5   | ✅            | ✅          | ✅             | Best for fragmented memory       |

---

### Conclusion

Choosing the right heap scheme depends on your system's needs:

- Use `heap_1` for static applications with tight memory.
- Use `heap_4` or `heap_5` for advanced applications with dynamic tasks and fragmentation concerns.
- Monitor heap usage with `xPortGetFreeHeapSize()` and `xPortGetMinimumEverFreeHeapSize()` to optimize memory over time.

