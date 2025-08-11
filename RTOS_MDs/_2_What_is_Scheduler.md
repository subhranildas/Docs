## What is a Scheduler?

A **scheduler** is the part of an operating system that decides **which task
(or thread)** should be **executed next** by the processor. Since a
microcontroller has only one core in most cases, it can only execute one task at a time. The scheduler creates an illusion of
**parallelism** by switching between tasks quickly — this process is known as
**context switching**.

> The scheduler is like a traffic controller for your tasks. It decides who gets
to run, for how long, and in what order.

## Role of Scheduler in FreeRTOS

FreeRTOS is a lightweight, real-time operating system for microcontrollers that
supports multitasking. The **FreeRTOS scheduler** is responsible for:
- Managing multiple **Tasks**
- Determining **when to switch** between tasks
- Ensuring **priority-based task execution**
- Performing **context switching** between tasks

## Types of Scheduling in FreeRTOS

FreeRTOS provides two types of scheduling:


<!-- tabs:start -->

#### **Preemptive Scheduling**
- Tasks can be interrupted mid-execution if a **higher-priority task becomes
  ready** to run.
- Ensures that **high-priority tasks get immediate CPU time**.
- Ideal for real-time systems with strict timing requirements.

#### **Cooperative Scheduling**
- A task must explicitly yield control.
- Simpler to manage, but tasks must behave cooperatively.
- Useful for simpler applications.

<!-- tabs:end -->
