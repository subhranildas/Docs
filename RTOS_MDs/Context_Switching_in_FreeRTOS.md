## What is Context Switching?

A **context switch** is the sequential execution of the following actions:

- **Saving** the state of the currently running task
- **Restoring** the state of the next task to run; so that execution can resume where the new task last left off.

This switching is what creates the **illusion of parallelism** on a single-core
CPU.

#### The “Context” Includes the following thing:

- CPU registers (R0–R12)
- Stack pointer (SP)
- Link register (LR)
- Program counter (PC)
- Processor status (APSR)
- Floating Point registers (if avilable)

## When Does Context Switching Happen?

FreeRTOS switches context in several scenarios:

1. **Preemption**: A higher-priority task becomes ready (In Preemptive mode)
2. **Blocking**: A task goes into Blocked or Suspended state
3. **Voluntary Yield**: A task explicitly calls `taskYIELD()`
4. **Interrupts**: An ISR wakes a higher-priority task

## How Does FreeRTOS Perform a Context Switch?

A **typical sequence** of a context switch in FreeRTOS on an ARM Cortex-M:

### Step 1: PendSV Interrupt is Triggered

FreeRTOS uses a special interrupt called **PendSV (Pendable Service Call)** to
handle context switches.

- The scheduler triggers `PendSV` when it decides to switch tasks.
- This ensures switching happens **outside of critical code paths**.

```c
#define portYIELD() portNVIC_INT_CTRL_REG = portNVIC_PENDSVSET_BIT
```

### Step 2: Current Task Context is Saved

When the PendSV_Handler runs:

#### Automatically saved by hardware:

- R0–R3, R12, LR, PC, xPSR
- Pushed onto the current task’s stack

#### Manually saved by software (FreeRTOS assembly code):

- R4–R11
- Stored on the task’s stack

The stack pointer (SP) now points to the saved context.

### Step 3: Scheduler Chooses the Next Task

- The FreeRTOS scheduler selects the highest priority Ready task.
- The kernel updates the global pointer to the new Task Control Block (TCB).

> Each task’s TCB holds a pointer to its top-of-stack in the saved context.

### Step 4: New Task Context is Restored

The context of the new task is loaded:

- Software restores R4–R11 from the task’s stack
- Hardware automatically restores R0–R3, R12, LR, PC, and xPSR when
  PendSV_Handler exits
- Execution continues as if the new task had never been paused.

## What about critical sections?

During context switching interrupts are disabled to avoid race conditions.
This ensures atomicity when saving and restoring context.

> FreeRTOS uses macros like taskENTER_CRITICAL() and taskEXIT_CRITICAL() to
> protect critical regions.
