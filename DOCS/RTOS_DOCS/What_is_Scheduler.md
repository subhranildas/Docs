## What is a Scheduler?

A **scheduler** is the part of an operating system that decides **which task
(or thread)** should be **executed next** by the processor. Since a
microcontroller has only one core in most cases, it can only execute one task at a time. The scheduler creates an illusion of
**parallelism** by switching between tasks quickly — this process is known as
**context switching**.

> The scheduler is like a traffic controller for your tasks. It decides who gets
> to run, for how long, and in what order.

## Scheduler Criteria

In operating systems, **scheduling criteria** are the metrics used to **evaluate and compare CPU scheduling algorithms**.
These help in determining the efficiency, responsiveness, and fairness of an algorithm.

---

### Common Scheduling Criteria

| **Criterion**                  | **Description**                                                                             | **Goal**        |
| ------------------------------ | ------------------------------------------------------------------------------------------- | --------------- |
| **CPU Utilization**            | Percentage of time the CPU is actively processing.                                          | 🔼 Maximize     |
| **Throughput**                 | Number of processes completed per unit of time.                                             | 🔼 Maximize     |
| **Turnaround Time (TAT)**      | Time from submission to completion of a process. <br>`TAT = Completion Time - Arrival Time` | 🔽 Minimize     |
| **Waiting Time (WT)**          | Total time a process spends in the ready queue. <br>`WT = Turnaround Time - Burst Time`     | 🔽 Minimize     |
| **Response Time**              | Time from submission to first CPU response/output (not completion).                         | 🔽 Minimize     |
| **Fairness**                   | Ensures no process is starved; all get a fair share of CPU time.                            | ✅ Ensure Equal |
| **Predictability**             | Consistent and predictable performance, especially important for real-time systems.         | 🔼 Increase     |
| **Context Switching Overhead** | Time wasted when switching between processes (not doing useful work).                       | 🔽 Minimize     |

---

<!-- ### Informal Algorithm Comparison

| **Algorithm**   | **Throughput** | **Waiting Time** | **Response Time** | **Fairness** | **CPU Utilization** |
|-----------------|----------------|------------------|-------------------|--------------|----------------------|
| FCFS            | Moderate       | High (if bursts vary) | Poor (for late arrivals) | Fair (but naive) | Good |
| SJF             | High           | Low              | Moderate          | May starve longer jobs | Good |
| Round Robin     | Moderate       | Moderate         | Good              | Very Fair     | Lower (due to overhead) |
| Priority        | Varies         | Varies           | Varies            | Poor (can starve) | Good |

--- -->

### Choosing the Right Criteria for Specific Systems

- **Batch Systems**: Focus should be on **Throughput** and **CPU Utilization**.
- **Interactive Systems**: Focus should be on **Response Time** and **Fairness**.
- **Real-time Systems**: **Predictability** and **Deadline Adherence** should be prioritized.

---

### Some Common CPU Scheduling Algorithms

#### 1. FCFS (First Come First Serve)

- **Type**: Non-preemptive
- **Description**: Processes are scheduled in the order they arrive.
- **Pros**: Simple and easy to implement.
- **Cons**: Can lead to long wait times (convoy effect).

#### 2. Round Robin (RR)

- **Type**: Preemptive
- **Description**: Each process gets a fixed time slice (quantum); then it's moved to the back of the queue.
- **Pros**: Fair and responsive; good for time-sharing systems.
- **Cons**: High context-switch overhead with small time quantum.

---

#### 3. Weighted Round Robin (WRR)

- **Type**: Preemptive
- **Description**: Extends Round Robin by assigning weights to processes, allowing higher-weight processes to get more CPU time.
- **Pros**: Better control over resource allocation.
- **Cons**: More complex than basic RR.

---

#### 4. Rate Monotonic Scheduling (RMS)

- **Type**: Real-time, Preemptive
- **Description**: Assigns priorities based on the periodicity of tasks — the shorter the period, the higher the priority.
- **Pros**: Proven to be optimal for fixed-priority real-time scheduling under certain conditions.
- **Cons**: Not suitable for dynamic or aperiodic tasks.

---

#### 5. Shortest Job First (SJF)

- **Type**: Can be Preemptive or Non-preemptive
- **Description**: Executes the process with the shortest burst time next.
- **Pros**: Minimizes average waiting time.
- **Cons**: Can cause starvation for long processes.

---

> Refer to the next section **Scheduling Algorithms** for details about some commonly used scheduling algorithms.

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
