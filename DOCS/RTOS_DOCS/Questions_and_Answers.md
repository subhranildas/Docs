FreeRTOS Questions:

## What is FreeRTOS, and why is it used in embedded systems?

### What:

FreeRTOS is a lightwaight open-source operating system designed for
microcontrollers and Embedded Systems.

### Why:

Deterministic RTOS, guarantees task execution within specific time constraints.
Preemptive Scheduling - Higher Priority Task always runs first.
Small Footprint of Kernel.
Static and Dynamic Task Creation.
Queues(FIFO Data Transfer).
Semaphore/Mutexes(Resource Locking).
Hardware Portabality.
Memory Management Options(5).
Low Power Support(Tickless Idle Mode, deep sleep when No Task is running).
MISRA C compliant.
Supported by AWS with Multiple different network layers.

## What are the key features of FreeRTOS?

Preemptive and Cooperative Scheduling.
Small and Efficient Kernel.
Dynamic and static Task Creation with configurable priority.
Multiple heap allocation Schemes.
Inter Task Communication synchronization with Queues, Semaphores, Event Groups,
Streme and Message Buffer.
Software Timers(Runs in a dedicated timer Task with configurable priority).
Tickless Idle Mode(CPU enters sleep mode when idle).
Deferred interrupt processing(using xQueueSendFromISR()).
Hardware Portability(40+ Architecture Supported).
MISRA C compliant.
Free for commercial Use(MIT License).
AWS IoT integration, secure Cloud Connectivity.

## How does FreeRTOS differ from other RTOS like RTX, Zephyr, or ThreadX?

ThreadX is from Microsoft Azure RTOS ecosystem. Rich in middleware like, FileX,
NetX, USBX, GUIX etc. Advanced block and byte-pool allocators unlike simple heap
in freeRTOS. Memory footprint is slightly higher than FreeRTOS(10-20KB).
Proprietary Software.

Zephyr is from linux foundation with rich middleware Support(BlueTooth, USB,
Networking), Advanced MMU/MPU Support, Preemptive and deadline Based. very wide
support for architecture(ARM, RISC-V, x86, etc.).

## Is FreeRTOS a preemptive or cooperative scheduler? Can it be configured for both?

FreeRTOS as a preemptive scheduler but it can also be configured as a
cooperative scheduler and also in hybrid mode.

It is controlled by two configuration Macros:

Hybrid Mode:
#define configUSE_PREEMPTION 1 // Enable preemption
#define configUSE_TIME_SLICING 0 // Disable time-slicing
(same-priority tasks don't auto-switch)

Preemptive Mode:
#define configUSE_PREEMPTION 1 // Enable preemption
#define configUSE_TIME_SLICING 1 // Enable time-slicing
(same-priority tasks share time and higher priority tasks always run first)

Cooperative Mode:
#define configUSE_PREEMPTION 0 // Disable preemption
#define configUSE_TIME_SLICING 0 // Disable time-slicing
(same-priority tasks don't auto-switch)

## What is the difference between task and process in FreeRTOS?

Process does not really exist in the context of FreeRTOS, but if we had to
consider this, then it's a concept of general computing with lots of complexity,

Microcontrollers dont have MMU(Memory Management Unit) which is a requirement
of Processes.

## How do you create a task in FreeRTOS? (Explain xTaskCreate() and xTaskCreateStatic())

xTaskCreate():
Allocates task Stack and TCB(Task Control Block) from the FreeRTOS heap.
Best for application where memory usage varies.
Benefit is system is not constrained by heap fragmentation.

xTaskCreateStatic():
Requires the user to manually allocate stack and TCB memory(Usually Global Arrays)
Good for deterministic Systems and safety critical applications and memory
constrained devices.

### Declaration of Stack and TCB:

StackType_t xTaskStack[TASK_STACK_SIZE];
StaticTask_t xTaskTCB;

Note: taskStack is basically a stack that is word size bit wide. For ARM word
size is 32 bit. So if stack depth is 2 then 8 bytes are allocated for ARM.
Could be different for different architecture.

configSTACK_DEPTH_TYPE says how wide the stack is. It is a configurable in
FreeRTOSConfig.h

## What is a task priority in FreeRTOS? How many priority levels are available?

Task priority is basically how much priority a task is given for execution
during scheduling by the scheduler. If preemptive scheduling is selected, then
the scheduler will always execute the ready high priority task.

Priority levels are customizable up to 255.

configMAX_PRIORITIES is the configurable which is used to set number of priority
levels.

Too many Task priority values could lead to RAM's overconsumption. If many tasks
are allowed to execute with varying task priorities, it may decrease the system's
overall performance.

## What happens if two tasks have the same priority in FreeRTOS ?

In this case the switching is controlled by the scheduler. In preemptive
Scheduling if time slicing is on then the scheduler will share time between
tasks of same priority, otherwise the first task to run will run forever.

Only way to get out of this is if the first task decides to give other task an
opportunity to run through some taskdelay or taskyeald

## How do you delete a task in FreeRTOS?

vTaskDelete call with the task handle.

If a task is deleted by another task, then the task is deleted in the API itself.
If a task deletes itself then the idle task is responsible for freeing the
memory of the deleted task and this is why Idle Task should not be starved.

Memory allocated inside a task is not cleared by the kernel, should be cleared
before deleting a task.

## What is the idle task in FreeRTOS, and what is its purpose?

The Idle task is a default system task automatically created by FreeRTOS when
the scheduler starts. It has the lowest priority(0) and runs when no other task
is active.

- Does memory cleanup, if tasks are dynamically allocated then the Idle task
  frees their stack and TCB(Task Control Block) memory.
  This requires configUSE_IDLE_HOOK = 1 to enable cleanup.
- Low power Mode, configUSE_TICKLESS_IDLE = 1, puts the cpu in sleep mode.
- User defined HOOK function, users can add custom code via the Idle Task Hook
  void vApplicationIdleHook(void){}
- It is recommended to avoid infinite loop in the idle task hook function.

## How can you implement a periodic task in FreeRTOS ?

There are many ways to do that.

1. Using vTaskDelay in a task, this will suspend the task for fixed time but
   does not compensate for the scheduling delays, therefore it will drift.
2. Using vTaskDelayUntil, Best for precise timing(compensates for execution time
   drift).
3. Using Software Timer, this wont be a task but will run periodically. Timers
   run in the RTOS timer task, priority is set by configTIMER_TASK_PRIORITY
   not recommended for higher priority tasks. Callback runs as a lower priority
   task.
4. Using Task + Queue(Event driven Periodicity):
   Best for triggering tasks externally. An ISR can be triggered periodically and
   from the ISR we can push some data into a queue. A task can be created and that
   task can be blocked until something is received in the queue.

## What is vTaskDelay() vs. vTaskDelayUntil()?

1. vTaskDelay suspends the Task for a fixed number of ticks from the moment
   it is called. Periodic and may drift.

vTaskDelayUntil ensures stric periodicity by compensating for the task execution
time.

## What is task state in FreeRTOS (Ready, Running, Blocked, Suspended)?

FreeRTOS tasks can exist in 4 primary states, managed by the scheduler.
Understanding these states helps optimize task behavior and system performance.

1. Running(Active):
   Current Task Executing in the CPU.
   Only one task per core can be in this state at a time.

2. Ready(Runnable):
   Task is ready to run but not currently executing(Waiting for CPU time).
   Some situations where a task cam go into ready state:

   - Task is just Created(xTaskCreate()).
   - A delayed task expires.

3. Blocked(Suspended):
   Task is waiting for an event or timeout and not consuming CPU cycles.
   vTaskDelay(), vTaskDelayUntil().
   xQueueReceive(), xSemaphoreTake().

4. Suspended():
   Task is explicitly suspended and is ignored by the scheduler.
   xTaskSuspend()
   Only Resumes when explicitly resumed( vTaskResume() ).
   Not waiting for any event, just inactive.

## How does FreeRTOS handle task stack overflow?

There are two ways to handle Stack Overflow in FreeRTOS:

Method 1: HEAP fill pattern
FreeRTOS fills the stack with a known pattern when the task is created.
On each context switch, the last valid stack location is checked.
If overflow happens the pattern will not match and thats how over flow is
detected.

    Problem with this is, detection happens post overflow.

Method 2: Stack Pointer Validation
The stack pointer is checked during context switches.
If the SP exceeds the stack's valid range, overflow is detected.

    More reliable, catches preemption before corruption.

Overflow detection can be enabled by setting
configCHECK_FOR_STACK_OVERFLOW to 1

Callback Function:
void vApplicationStackOverflowHook(TaskHandle_t xTask, char \*pcTaskName);

Preventive Method:

Check minimum free Stack ever Recorded:
uxTaskGetStackHighWaterMark() and keep it 20% of total stack size during test.
avoid deep recursion, use loops and dynamic allocation for large buffers.

## What is the daemon task (e.g., for timers and queues)?

The Daemon Task also called the Timer Service Task or RTOS Timer Task is a
system-created background task in FreeRTOS that manages software timers and
other deferred operations. It is essential for handling asynchronous events.

What it does: - Manages timers created with xTimerCreate(). - Runs timer expiration callbacks in the task context(Not in an ISR). - Process Deferred interrupt handling. xQueueSendFromISR() signals the
daemon task. - Assists in low-power tick-less idle mode.

## Explain the FreeRTOS scheduler (preemptive, time-slicing).

FreeRTOS offers two scheduler types:

1. Preemptive Scheduler:

   - The highest priority task always runs first.
   - If a higher-priority task becomes ready due to an interrupt or event, it
     immediately preempts the current task.
   - Ensures time critical tasks get CPU time as soon as possible.
   - Configured by setting configUSE_PREEMPTION = 1 in FreeRTOSConfig.h

2. Cooperative Scheduler:

   - Tasks must explicitly yield control ( taskYIELD() ) or block (e.g., on a
     delay or a semaphore) to allow other tasks to run.
   - Simpler but risks starvation if a task doesn't yield.
   - Configured by setting configUSE_PREEMPTION = 0 in FreeRTOSConfig.h

## What is context switching in FreeRTOS ?

The context switch in FreeRTOS is the process of saving the current state(context)
of a running task and restoring the state of a new task so it can execute. This
happens when:

    - A higher-priority task becomes ready (preemption)
    - The current task blocks (e.g. calls vTaskDelay(), waits on a semaphore).
    - An explicit yield( taskYIELD() ) occurs.
    - The scheduler's time slice expires (round-robin scheduling).

During a context switch the following happens:

    - The processor's register values are stored (e.g., Program Counter, Stack
    Pointer, CPU registers) are saved to the current task's stack.
    - The tasks state is updated in the task control block.
    - Scheduler checks the ready tasks list and picks the highest-priority task
    - If multiple tasks share the same priority, round-robin scheduling may apply.
    - The saved registers are loaded back into the CPU.
    - The Program Counter (PC) is updated to resume execution where the task left off.
    - The Stack Pointer is updated to point to the new task's stack.
    - The CPU continues executing the new task from it's last saved state.
    - FreeRTOS uses PendSV, a low-priority interrupt, to handle context switching cleanly without disrupting real-time ISRs.
    - If no higher priority task is ready then the scheduler just takes care of
    time of delayed tasks and does not switch if no higher priority task becomes
    ready.

## How does FreeRTOS decide which task to run next ?

FreeRTOS uses a priority-based scheduler to determine the next task to run.
The decision depends on:

1. Task Priority.
2. Task State(Ready, Blocked, Suspended)
3. Scheduling Policy(configUSE_PREEMPTION and configUSE_TIME_SLICING).

## What is tickless idle mode in FreeRTOS?

Tickless Idle Mode is a power-saving feature in FreeRTOS that disables the
periodic SysTick interrupt when the system is idle(No Active Tasks except the
idle Task).

Entering the Tickless Idle:

    - When the idle task runs and no other tasks are active, FreeRTOS:
        1. Calculates the nearest task delay or wakeup time.
        2. Disables the SysTick timer.
        3. Configures a low-power timer(e.g., RTC, LPTIM) to wake up the MCU at
           the right time.
        4. Enters sleep Mode.

Waking up from the Tickless Mode:

    - The MCU wakes up due to:
        1. An external interrupt
        2. A timer interrupt(when delayed task is yet to run).
        3. Re-enables the SysTick timer.
        4. Corrects the system time since ticks were skipped during sleep.
        5. Resumes normal Scheduling.

## What is configTICK_RATE_HZ, and how does it affect scheduling?

configTICK_RATE_HZ is a macro defined in FreeRTOSConfig.h that sets the frequency
of the RTOS tick interrupt(SysTick). It determines how often the FreeRTOS scheduler
checks for task switches and updates system time.

It affects scheduling, how often the scheduler evaluates which task to run. More
frequency reduces time to response to events.

Time base delays are also affected, they take input as system ticks.

Round Robin time slicing also gets affected. Tasks of equal priority share time
equivalent to one tick interval.

Increasing may induce more cpu load but response time is quicker.

## What is a mutex in FreeRTOS, and how is it different from a binary semaphore?

Both Mutexes and binary semaphores are sinchronization mechanisms but they serve
different purposes.

Purpose of MUTEX:

Ensures exclusive access to shared resource.
Prevents Priority inversion.

- Only the task that takes the mutex can give it.
- Priority Inheritance, if a higher priority task waits for a mutex held by a
  low priority task, the low priority task temporarily inherits the higher priority
  to finish faster.
- Used for resource Protection.

Purpose of Binary Semaphore:

Used for task synchronization. signaling between tasks or ISRs.
Does not manage resource ownership.

- No Ownership: Any task/ISR can give the semaphore, and any task can take it.
- No Priority Inheritance.
- Used for event signaling(e.g., notify a task that data is ready).

## How do you use a semaphore in FreeRTOS? (Counting vs. Binary)

Binary Semaphore:

A flag like semaphore with only 2 States: - 0 or 1

Counting Semaphore:

A token-based semaphore with a configurable count.
Represents multiple available resources.

    - Tells count of available resources.
    - Starts with max available count and reduces if taken, increases if given
      back.

## What is priority inversion, and how does FreeRTOS handle it?

Priority inversion happens a high priority task is forced to wait for a low
priority task that holds a shared resource, while medium-priority tasks preempts
the low-priority task delaying the high-priority task indefinitely.

How FreeRTOS Handles Priority inversion:

- FreeRTOS uses priority inheritance for mutexes to mitigate inversion.

## What is a recursive mutex in FreeRTOS?

A recursive mutex is a special type of mutex that allows the same task to lock
it multiple times without deadlocking itself. Each lock must be matched by an
equal number of unlocks before the mutex is released for other tasks.

- Has Priority inheritance.
- Needed when a function that locks a mutex calls another function that also
  needs the same mutex.

## What are queues in FreeRTOS, and how are they used for inter-task communication?

Queues are thread safe FIFO data structure used to pass messages or data between
tasks, or between tasks and interrupts. They are the primary means of inter task
communication in FreeRTOS.

Sends data to the Queue, receiver gets blocked until queue is empty.

Use Cases:

- Sending sensor data from a reading task to a processing task.(Task to Task)
- Notifying a task that a button was pressed.(ISR to Task).
- Variable size of Queue.

## How do event groups work in FreeRTOS?

Event Groups are synchronization mechanism in FreeRTOS that allows tasks to wait
for multiple events (flags) and responds when any or all of them occur. They
are particularly useful for :

- Task coordination( Wait for multiple sensors to finish reading ).
- Efficient event signaling ( Better that multiple semaphores/binary signals ).
- 24 bits per event group.
- Can be set from ISR.

## What is the difference between xQueueSend() and xQueueSendToBack()?

There is no functional difference. xQueueSendToFront() is different.

## What is a task notification, and how is it faster than queues/semaphores?

Task Notifications are a lightweight method for sending signals or data directly
to a task without using queues, semaphores, or event groups. They are up to 45%
faster and use less RAM than traditional methods.

Each task has a private 32-bit notification value.
Another task or ISR can update this value and optionally wake the waiting task.
Supports 4 operations:
Set a bit (like Event Flags).
Increment a counter.
Overwrite the value.

Can replace binary semaphore with xTaskNotifyGive() and xTaskNotifyTake()
Can replace event groups by using xTaskNotify() and xTaskNotifyWait()
A 32 bit value can be send with same approach above.

## How does FreeRTOS handle dynamic memory allocation?

FreeRTOS provides flexible memory management strategies to allocate memory for
tasks, queues, semaphores, and other kernel objects. Unlike general-purpose OSes
FreeRTOS is designed for resource constrained embedded systems, so its memory
allocation methods are highly configurable.

FreeRTOS supports 5 heap management schemes. Including the corresponding file in
a project does the job.

heap_1:
Only Malloc(Simplest), Minimal overhead deterministic, no memory reclaiming.

heap_2:
Supports dynamic allocation, Fragmentation risk, non-deterministic.

heap_3:
Uses standard library, Slower, depends on system malloc.

heap_4:
Improved heap_2 with adjacent free block merging, Reduces fragmentation, best
for most applications. Slightly larger code size.

heap_5:
basically heap_4 with non-contiguous memory regions. Works with scattered RAM
(e.g., SRAM1, SRAM2), More complex setup.

## What are the different heap management schemes in FreeRTOS (heap_1 to heap_5)?

## Same as before.

## When would you use pvPortMalloc() instead of standard malloc()?

malloc()/free():
Not inherently thread-safe. Concurrent calls can corrupt the heap.
Requires additional locks(e.g., mutex) to protect allocations, adding overhead.
Execution time varies. Unsuitable for real-time systems where timing must be
predictable.
Uses a single heap defined by the compiler.

pvPortMalloc()/vPortFree():
Built-in thread safety (designed for RTOS use).
No need for external synchronization.
Optimized for embedded systems with fixed or deterministic behavior.
heap_4 and heap_5 merges adjacent free blocks to minimize fragmentation.
better suited for long running embedded systems.
Supports multiple non-contiguous memory regions.
Works without a full C library.
Comes with heap monitoring APIs (xPortGetFreeHeapSize(),
xPortGetMinimumEverFreeHeapSize())

## How can you detect memory leaks in FreeRTOS?

FreeRTOS Trace Hook Macros:

Enabling FreeRTOS Trace Hook macros in FreeRTOSConfig.h
traceMALLOC(pvAddress, uiSize)
traceFREE(pvAddress)

## How does FreeRTOS handle ISRs (Interrupt Service Routines)?

FreeRTOS requires configurable interrupt Priorities(e.g., ARM Cortex-M NVIC).
configMAX_SYSCALL_INTERRUPT_PRIORITY
Defines the highest interrupt priority level that can call FREERTOS-from-ISR
APIs.
Interrupts above this priority can not use FreeRTOS APIs.

    configKERNEL_INTERRUPT_PRIORITY
    Interrupt priority of the kernel(Mostly set to lowest 255(for ARM Cortex-M))

## What is a deferred interrupt handler (using a task or a binary semaphore)?

A deferred Interrupt Handler is a design pattern used in real-time systems to
minimize time spent in interrupt Service Routines(ISRs) by offloading non-urgent
processing to a task.

This ensures:
Low interrupt latency (Critical for hard real time systems)
Thread safe processing of interrupt data.
Reduces jitter in task scheduling.

## Can be achieved using a Queue/semaphore/task notification

## What are critical sections in FreeRTOS (taskENTER_CRITICAL() vs taskEXIT_CRITICAL())?

Critical sections in FreeRTOS are code blocks where interrupts are temporarily
disabled or scheduler preemption is prevented to ensure atomic operations. They
protect Shared resource(e.g., global variables, peripherals) from race conditions.

caused by:

    interrupts
    Task switches

There are two levels of protection:

Task Level Critical section:
Disables scheduler preemption (but not interrupts).
Protects against other tasks but ISRs can still run.

Interrupt level Critical Section:
Disables all interrupts(or up to configMAX_SYSCALL_INTERRUPT_PRIORITY).
Protects against ISRs and Tasks.

## What is configMAX_SYSCALL_INTERRUPT_PRIORITY?

Defines the highest interrupt priority level that can call FREERTOS-from-ISR
APIs.
Interrupts above this priority can not use FreeRTOS APIs.

## Why should you avoid using vTaskDelay() inside an ISR?

ISRs must be non blocking.
No task is running during an ISR(The interrupted tasks state is frozen).
Calling will trigger an error.
ISR might never Complete and halt the system.

## Deffering to a task is one way to handle the delay and subsequent processing.

## How do software timers work in FreeRTOS?

FreeRTOS provides software timers to schedule functions to run after a specified
delay or at a fixed interval, without requiring a dedicated hardware timer.

Here's how they work:

xTImerStart()
xTimerStop()
xTimerReset()
xTimerChangePeriod()

The timer task priority can be set by user
Jitter can occur if timer task is blocked by a higher priority task.
Callback run in the Timer task context(Not in ISR Context).

## What is the difference between one-shot and auto-reload timers?

In one-shot timer fires once after a delay.
In auto-reload mode the timer repeats at a fixed interval.

## Can FreeRTOS timers expire while the scheduler is suspended?

Timer interrupts are queued by the timer service task/daemon task but they are
serviced by the scheduler, if the scheduler is suspended then the corresponding
callback is not run until the scheduler resumes.

## How do you debug a stack overflow in FreeRTOS?

1. Enable Stack overflow detection:

   configCHECK_FOR_STACK_OVERFLOW 0/1/2
   1 -> check only on task switch
   2 -> fill stack with a pattern and verify it.

2. Implement a stack overflow hook function:
   vApplicationStackOverflowHook(TaskHandle_t xTask, char \*pcTaskName){}

3. Check task stack Usage:
   Use uxTaskGetStackHighWaterMark() to track stack usage.
   Gives the minimum available stack ever.

4. Segger system view to to check stack allocation in real time.

## What is uxTaskGetStackHighWaterMark() used for ?

This is an API provided by the FreeRTOS to check the minimum free stack size
ever.

## How can you measure task execution time in FreeRTOS ?

Enable configGENERATE_RUN_TIME_STATES in FreeRTOSConfig.h
Implement

portCONFIGURE_TIMER_FOR_RUN_TIME_STATS()
portGET_RUN_TIME_COUNTER_VALUE()

## What is tracing in FreeRTOS, and how is it useful?

Tracing in FreeRTOS refers to recording and analyzing runtime behavior of tasks,
interrupts, and kernel events( context switches, queue operations). It helps debug
complex issues.

FreeRTOS supports tracing through:

Event Hooks: Custom callbacks for key kernel events.
Real-time data sent to a host PC.
Recorded traces stored in RAM/flash for later analysis.

## What is FreeRTOSConfig.h, and what are the key configurations?

FreeRTOSConfig.h is the configuration file which defines all the configurations
for the Kernel.

vPortSVCHandler():
Used to launch the very first task.

xPortPendSVHandler():
Used to achieve Context switching between tasks.
Triggered by pending the PendSV system exception of ARM

xPortSysTickHandler():
This Implements the RTOS Tick management and Triggered periodically by the
Systick timer.
This systick handler triggers the pendSV handler if required.

## How do you port FreeRTOS to a new micro-controller?

Porting FreeRTOS is divided into a couple of steps:

- Obtain FreeRTOS SourceCode.
- Set up the Development Environment.
  Choose a toolchain(GCC, Keil, IAR).

- Create a new Port Folder in FreeRTOS/Source/portable/[Compiler]/new_MCU
- Copy an existing port for similar core architecture.
- in port.c implement vPortStartFirstTask()
- in port.c implement xPortStartScheduler()
- in port.c vPortYield()
- in portmacro.h define portSTACK_TYPE(Ex. uint32_t).
- in portMacro.h define portBYTE_ALLIGNMENT (stack alignment).
- in portMacro.h portTICK_PERIOD_MS (SysTick timer configuration).
- Critical Section macros(portENTER_CRITICAL(), portEXIT_CRITICAL())
- in portasm.s if needed pendSV handler for arm.
- Configure a hardware timer (usually SysTick for ARM) to generate the FreeRTOS
  tick (typically 1ms).
- Implement printf for debugging.
- Implement configASSERT, vApplicationStackOverflowHook
- If possible add Example ports in FreeRTOS/Demo/

## What is configUSE_PREEMPTION and configUSE_TIME_SLICING?

configUSE_PREEMPTION:

    This is responsible for preemption, if active the scheduler will preempt
    current task if a higher priority task becomes active.

configUSE_TIME_SLICING:

    This is responsible for time slicing between tasks of same priority. If
    active scheduler will switchout and switch in tasks of similar priority.

## What is the role of the SysTick timer in FreeRTOS ?

The SysTick timer plays a critical role in FreeRTOS as it generates the system
tick interrupt, which is the heartbeat of the RTOS scheduler.

- On each tick the scheduler checks if a higher-priority task is ready to run.
- If yes, it preempts the current task(if configUSE_PREEMPTION = 1)
- vTaskDelay() - Pauses a task for N ticks.
- vTaskDelayUntil() - Delays a task until an absolute time.
- Timeout calculation (e.g., xQueueSend() with timeout).
- Idle Task processing.
- configGENERATE_RUN_TIME_STATES uses SysTick to track CPU usage per task.
- SysTIck must not block Higher Priority interrupts.
- On Cortex-M, set priority higher than PendSV but lower than critical ISRs.

## What is MPU (Memory Protection Unit) support in FreeRTOS?

Memory Protection Unit(MPU) support in FreeRTOS allows the RTOS to enforce
memory access restrictions for tasks, improving system reliability and security
by isolating tasks from unauthorized memory access.

- Prevents tasks from corrupting Kernel memory.
- Other task's memory.
- Critical peripherals.

Key Features:

- Each Task runs in it's own protected memory regions.
- Tasks run in unprivilaged mode by default while the kernel in privileged mode
- Tasks can only access memory assigned to them.
- Detects stack overflow via MPU Violation.
- Supports dynamic region updates per task.

#define configENABLE_MPU 1 // Enable MPU support
#define configENABLE_FPU 1 // If using FPU (optional)
#define configTOTAL_MPU_REGIONS 8 // Depends on MCU (e.g., Cortex-M7 has 16)
#define configTEXTPROTECTION_ENABLED 1 // Execute-only code protection

Create a restricted Task:

    xTaskCreateRestricted(&xTaskParams, &xTask);

If a task violates MPU rules:

    Memory Fault handler Triggers.
    FreeRTOS can kill the offending task and log the error.

## How does FreeRTOS work with multi-core processors (SMP)?

FreeRTOS can work on multi-core Processors using two approaches.

SMP -> Symmetric Multiprocessing.
AMP -> Asymmetric Multiprocessing.

Single FreeRTOS kernel schedules tasks across all cores.
Shared memory allows tasks to communicate.

Global task scheduler - Tasks can run on any core.
Spinlock for multicore - safe critical sections.
Core affinity - pin tasks to specific cores.

xTaskCreatePinnedToCore() -> create task pinned to a particular core.

#define configUSE_CORE_AFFINITY 1 // Allow task-core binding
#define configNUM_CORES 2 // Dual-core
#define configUSE_SMP 1 // Enable SMP
#define configRUN_MULTIPLE_PRIORITIES 1 // Allow different priorities per core

## Real world implementations: FreeRTOS SMP on ESP32

---

## How would you design a real-time system with multiple sensors using FreeRTOS?

Considerations:

There are a couple of things to consider.

1. Time based task (Not Time Critical): Something like a Temp Sensor
2. ACCL Sensor on interrupt:

There will be a task to take the temperature sensor value with a taskDelay or
task delay until.

There will be a ACCL task waiting on a semaphore for data availability. This
semaphore will be given from an ISR and the task would be woken. the task will
read the value from the sensor.

There will be a Processing task which will Process the data from the sensors.
This task will wait on a queue.
Each temp reading task will send the data once read to this queue.
The Processing task will wait on queue receive.

## How do you handle task starvation in FreeRTOS?

Task Starvation: Task starvation in FreeRTOS is a condition where a task never
gets CPU time or gets it very infrequently because higher priority tasks are
always running.

Whys to Prevent Task Starvation:

1. Blocking higher priority tasks with Queues and Delays.
2. Adjusting Priority wisely.
3. Enable Time slicing.
4. Objective is to increase the Idle time without compromising on requirements.

## Explain a scenario where you used queues, semaphores, and event groups together.

Lets consider a situation where we want to send some data to a device outside
the system over UART or some other communication method.

1. Two sensors lets say temp and Accl
2. Both share the same I2C bus.
3. Therefore each task collects data at a particular interval.
4. Each Task uses a mutex to block the I2C bus for use.
5. The I2C transaction happens in callback mode. Therefore after the transaction
   API call the application takes a semaphore which is given in the callback.
6. There is a separate Processing Task which filters and packs the data in proper
   format before sending it.
7. A Event group is used to make sure both the data is available. with
   xEventGroupWaitBits, after that the sensor data is collected from queue, send
   by the data reading tasks. Then the data is processed and sent.

## What is Privileged Mode in Cortex-M controller ?

In ARM Cortex-M controllers, privileged mode is a state where the processor has
access to all system resources and can execute any instruction. It's typically
used by the operating system or system firmware for critical tasks.
Unprivileged (user) mode limits access to certain resources, providing a
safety net against malicious code.

Purpose: Privileged mode is designed for code that needs to control the system,
such as an operating system or system firmware. It grants full access to all
memory, peripherals, and instructions.

Access Control: Unprivileged mode restricts access to certain memory regions
and peripherals, preventing user applications from making system-level changes.

Switching: Switching between privileged and unprivileged modes is typically
handled through system calls (SVCs) or interrupts. A privileged instruction
(SVC) can trigger a mode switch, and an interrupt can also cause the processor
to enter privileged mode.

Stack Pointer: In privileged mode, the processor uses the Main Stack Pointer
(MSP) as the default stack pointer. In unprivileged mode, the Process Stack
Pointer (PSP) is used, as explained by Arm Developer.

Thread and Handler Modes: Cortex-M controllers have two modes of operation,
Thread mode and Handler mode. Thread mode can be either privileged or
unprivileged, while Handler mode is always privileged. Handler mode is
entered as a result of exceptions, and all code running in Handler mode is
privileged.
