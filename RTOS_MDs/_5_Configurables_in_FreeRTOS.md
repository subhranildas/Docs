## What Is FreeRTOSConfig.h?

FreeRTOS is a lightweight, open-source real-time operating system for
microcontrollers and small embedded systems. While FreeRTOS comes with a
sensible default setup, fine-tuning its configuration via FreeRTOSConfig.h is
critical for optimizing performance, resource utilization, and system stability.

FreeRTOSConfig.h is a project-specific header file where you define compile-time
configuration options using #define statements. These options control system
behavior such as task scheduling, memory management, API inclusion, debugging
features, and more.


## Core Configuration Macros

Below are some core configuration Macros.

### configTOTAL_HEAP_SIZE

<!-- tabs:start -->
#### **Description**
Sets the total size (in bytes) of the FreeRTOS heap.
#### **Impact**
Too low causes task and object creation failures; too high wastes RAM.
<!-- tabs:end -->

```c
#define configTOTAL_HEAP_SIZE                   ( ( size_t ) ( 10 * 1024 ) )
```

---

### configMAX_PRIORITIES

<!-- tabs:start -->
#### **Description**
Maximum number of priority levels for tasks.
#### **Impact**
More levels give flexibility; each level adds a bit of memory overhead.
<!-- tabs:end -->

```c
#define configMAX_PRIORITIES                    5
```
?> Notice with `inline code` and additional placeholder text used to
force the content to wrap and span multiple lines.
---

### configTICK_RATE_HZ
#### Description:
Frequency of the RTOS tick interrupt.
#### Impact:
Higher rate = better timing resolution but more CPU overhead.

```c
#define configTICK_RATE_HZ                      ( ( TickType_t ) 1000 )
```

### configUSE_PREEMPTION
#### Description:
Enables (1) or disables (0) preemptive multitasking.
#### Impact:
Preemption improves responsiveness; cooperative mode is simpler to debug.

```c
#define configUSE_PREEMPTION                    1
```

### configUSE_TIME_SLICING

#### Description:
Enables (1) or disables (0) time slicing between multiple tasks with same
priority.
#### Impact:
Preemption is enabled, ready tasks with same priority will share time among
then.

```c
#define configUSE_TIME_SLICING                  1
```

### configMINIMAL_STACK_SIZE

#### Description:
Stack size for the idle task and a reference point for others.
#### Impact:
Too small causes crashes; monitor stack usage to adjust.

```c
#define configMINIMAL_STACK_SIZE                ( ( uint16_t ) 128 )
```

### configMAX_TASK_NAME_LEN
#### Description:
Maximum length of task names (including the null terminator).
#### Impact:
Longer names use more memory but aid debugging.

```c
#define configMAX_TASK_NAME_LEN                 16
```

## Optional Features and Debugging Aids

### configUSE_IDLE_HOOK / configUSE_TICK_HOOK
#### Description:
Enable custom functions during idle and tick interrupts.
#### Impact:
Useful for diagnostics or power-saving features.

```c
#define configUSE_IDLE_HOOK                     0
#define configUSE_TICK_HOOK                     0
```

### configCHECK_FOR_STACK_OVERFLOW
#### Description:
Enables stack overflow detection (option 1 or 2).
#### Impact:
Helps catch task stack issues during development.

```c
#define configCHECK_FOR_STACK_OVERFLOW          2
```

### configUSE_MALLOC_FAILED_HOOK
#### Description:
Run a hook function when heap allocation fails.
#### Impact:
Critical for diagnosing memory issues.

```c
#define configUSE_MALLOC_FAILED_HOOK            1
```

### configASSERT()
#### Description:
Define this to insert your own assert logic.
#### Impact:
Helps catch programming errors early.

```c
#define configASSERT( x ) if( ( x ) == 0 ) { taskDISABLE_INTERRUPTS(); for( ;; ); }
```

## Tracing and Runtime Statistics

### configUSE_TRACE_FACILITY
#### Description:
Enables trace-related functions like uxTaskGetSystemState().
#### Impact:
Useful for debugging and performance monitoring.

```c
#define configUSE_TRACE_FACILITY                1
```

### configGENERATE_RUN_TIME_STATS

#### Description:
Enables collection of runtime execution statistics.
#### Impact:
Adds insight into CPU usage by each task.

```c
#define configGENERATE_RUN_TIME_STATS           1
```

## API Inclusion Settings
These macros allow you to exclude unused API functions to reduce binary size.

```c
#define INCLUDE_vTaskDelay                      1
#define INCLUDE_vTaskDelete                     1
#define INCLUDE_vTaskSuspend                    1
#define INCLUDE_xTaskGetIdleTaskHandle          0
#define INCLUDE_uxTaskGetStackHighWaterMark     1
```

## Additional Useful Macros

### configUSE_TICKLESS_IDLE

#### Description:
Enables tickless low-power mode.
#### Impact:
Saves power on idle but requires accurate timers.

```c
#define configUSE_TICKLESS_IDLE                 1
```

### configNUM_THREAD_LOCAL_STORAGE_POINTERS

#### Description:
Enables per-task storage pointers (thread-local storage).
#### Impact:
Useful for thread-safe libraries or middleware.

```c
#define configNUM_THREAD_LOCAL_STORAGE_POINTERS 5
```
