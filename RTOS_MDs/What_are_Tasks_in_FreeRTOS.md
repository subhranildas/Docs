## Tasks and Threads in RTOS

### Introduction

In a **Real-Time Operating System (RTOS)**, the core concept revolves around executing **multiple independent activities** concurrently.  
These activities are implemented as **tasks** (also called **threads** in some systems).

- **Task**: A unit of execution managed by the RTOS scheduler.
- **Thread**: Often used interchangeably with "task," but in some contexts, a thread is a lightweight execution unit within a process.

In embedded RTOS environments, **task** is the more common term.

---

### Characteristics of a Task in RTOS

Each task in an RTOS typically has:

1. **Task Function** – The code that the task executes.
2. **Stack** – Private memory for local variables, return addresses, and function calls.
3. **Priority** – Determines scheduling preference.
4. **State Information** – Managed by the RTOS (ready, running, blocked, suspended).
5. **Task Control Block (TCB)** – Data structure storing task-related information.

---

### Task States in RTOS

A task can be in one of the following states:

- **Running** – Currently executing on the CPU.
- **Ready** – Ready to run but waiting for CPU time.
- **Blocked/Waiting** – Waiting for an event, message, or time delay.
- **Suspended** – Inactive until explicitly resumed.

---

### Threads vs Tasks in RTOS

| Feature         | Task (RTOS context)               | Thread (General OS context)                |
| --------------- | --------------------------------- | ------------------------------------------ |
| **Usage**       | Embedded & real-time applications | Desktop, server, and some embedded systems |
| **Memory**      | Usually has its own stack         | Shares address space with other threads    |
| **Overhead**    | Lightweight                       | Lightweight compared to full processes     |
| **Terminology** | Common in RTOS                    | Common in multi-threaded OS                |

---

## Tasks in FreeRTOS?

A **task** in FreeRTOS is an independent unit of execution — much like a thread
in a traditional operating system. It is essentially a function that runs
concurrently with other tasks, allowing you to divide your application into
multiple logical blocks.

Each task has:

- It's **own stack**
- It's **own execution context** (registers, program counter)
- A **priority level**
- A **Task Control Block (TCB)** for internal management

This means that multiple tasks can run "simultaneously" on a single-core
microcontroller — with the help of the FreeRTOS **scheduler**, which switches
between them rapidly.

### How Do Tasks Work in FreeRTOS?

FreeRTOS uses **cooperative** and **preemptive** multitasking to manage tasks:

- In **preemptive mode**, the highest-priority Ready task always runs first.
- In **cooperative mode**, tasks must voluntarily yield control.

As mentioned in the previous section.

### Task States in FreeRTOS

A task can be in one of the following five states:

| **State** | **Description**                                |
| --------- | ---------------------------------------------- |
| Running   | Task is currently executing                    |
| Ready     | Task is ready to run when scheduled            |
| Blocked   | Task is waiting for a timeout, delay, or event |
| Suspended | Task is paused and must be resumed manually    |
| Deleted   | Task has been removed and memory reclaimed     |

### Task State Transitions

- vTaskDelay() → moves task to Blocked state
- vTaskSuspend() → moves task to Suspended
- vTaskResume() → brings it back to Ready

### Context Switching

FreeRTOS uses a tick interrupt (from a hardware timer like SysTick)
in this subroutine the the scheduler checks whether any higher priority task can
be switched in or not, if it finds a task that can be switched in then it
pends the PendSV handler which does the context Stitching operation.

#### Switching Operation Steps

- Save the context (CPU registers, stack pointer) of the current task
- Load the context of the next highest-priority task
- Resume execution from where that task left off

This allows multiple tasks to "share" CPU time even on single-core systems.

### Code Examples

#### Task Creation Function Declaration

```c
xTaskCreate(TaskFunction, "TaskName", StackDepth, Parameters, Priority, &TaskHandle);
```

#### Example Task Creation and Execution

```c

void vLEDTask(void *pvParameters) {
    while(1) {
        ToggleLED();
        vTaskDelay(pdMS_TO_TICKS(500)); // Delay 500ms
    }
}

int main(void) {
    xTaskCreate(vLEDTask, "LED", 128, NULL, 1, NULL);
    vTaskStartScheduler(); // Start multitasking
}

```

### What Happens Behind the Scenes?

#### When **xTaskCreate()** is called:

- FreeRTOS allocates a stack for the task
- Initializes a Task Control Block (TCB) to store task metadata
- Adds the task to the Ready list
- Once **vTaskStartScheduler()** is called, the task can be scheduled for execution
- The scheduler will switch between tasks based on priority and state if
  **preemptive** scheduling is used.

---
