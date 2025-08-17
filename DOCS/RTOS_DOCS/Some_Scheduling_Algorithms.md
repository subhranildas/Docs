## FCFS (First Come First Serve) Scheduling Algorithm

### Introduction

**First Come First Serve (FCFS)** is the simplest CPU scheduling algorithm. As the name suggests, the process that arrives **first** is executed **first**.
It operates in a **non-preemptive** manner — once a process starts, it runs to completion.

### Key Characteristics

- **Non-preemptive**
- **Processes are executed in the order they arrive**
- **Simple to implement using a queue**

### Terminology

| Term                      | Meaning                                                |
| ------------------------- | ------------------------------------------------------ |
| **Thread/Process**        | Identifier for the task                                |
| **Arrival Time (AT)**     | Time at which the process enters the ready queue       |
| **Burst Time (BT)**       | CPU time required for execution                        |
| **Service Time (ST)**     | Time when the process starts execution                 |
| **Waiting Time (WT)**     | Time process waits in the queue: `WT = ST - AT`        |
| **Turnaround Time (TAT)** | Total time from arrival to completion: `TAT = WT + BT` |

### Example

#### Input Data

| Thread | Arrival Time (AT) | Burst Time (BT) |
| ------ | ----------------- | --------------- |
| T1     | 0                 | 5               |
| T2     | 2                 | 3               |
| T3     | 4                 | 1               |
| T4     | 6                 | 2               |

#### Execution Order

Since FCFS executes based on arrival time, the order is:

**T1 → T2 → T3 → T4**

### Scheduling Table

| Thread | Arrival Time (AT) | Burst Time (BT) | Service Time (ST) | Waiting Time (WT) | Turnaround Time (TAT) |
| ------ | ----------------- | --------------- | ----------------- | ----------------- | --------------------- |
| T1     | 0                 | 5               | 0                 | 0                 | 5                     |
| T2     | 2                 | 3               | 5                 | 3                 | 6                     |
| T3     | 4                 | 1               | 8                 | 4                 | 5                     |
| T4     | 6                 | 2               | 9                 | 3                 | 5                     |

### Gantt Chart

---

## Round Robin (RR) Scheduling Algorithm

### Introduction

**Round Robin (RR)** is a widely used CPU scheduling algorithm, especially in time-sharing systems. Each process is assigned a fixed time slice (called a **quantum**), and processes are executed in a cyclic order. If a process does not finish within its quantum, it is moved to the back of the queue.

### Key Characteristics

- **Preemptive**
- **Each process gets equal time slices (quantum)**
- **Fair and responsive for interactive systems**
- **Simple to implement using a circular queue**

### Terminology

| Term                      | Meaning                                                      |
| ------------------------- | ------------------------------------------------------------ |
| **Quantum**               | Fixed time slice allotted to each process                    |
| **Context Switch**        | The act of saving the state of a process and loading another |
| **Arrival Time (AT)**     | Time at which the process enters the ready queue             |
| **Burst Time (BT)**       | CPU time required for execution                              |
| **Remaining Time (RT)**   | Time left for process to complete                            |
| **Waiting Time (WT)**     | Total time process spends waiting                            |
| **Turnaround Time (TAT)** | Total time from arrival to completion                        |

### Example

#### Input Data

Assume a quantum of **2 units**.

| Thread | Arrival Time (AT) | Burst Time (BT) |
| ------ | ----------------- | --------------- |
| T1     | 0                 | 5               |
| T2     | 2                 | 3               |
| T3     | 4                 | 1               |
| T4     | 6                 | 2               |

#### Execution Order

Processes are executed in the order they arrive, each for a maximum of 2 units per turn:

**T1 → T2 → T3 → T4 → T1 → T2 → T4 → T1**

#### Scheduling Table

| Thread | Arrival Time (AT) | Burst Time (BT) | Waiting Time (WT) | Turnaround Time (TAT) |
| ------ | ----------------- | --------------- | ----------------- | --------------------- |
| T1     | 0                 | 5               | 6                 | 11                    |
| T2     | 2                 | 3               | 4                 | 7                     |
| T3     | 4                 | 1               | 2                 | 3                     |
| T4     | 6                 | 2               | 3                 | 5                     |

_Note: Waiting and turnaround times are calculated based on the order and quantum._

### Gantt Chart

| T1  | T2  | T3  | T4  | T1  | T2  | T4  | T1  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0   | 2   | 4   | 5   | 7   | 9   | 11  | 13  |

---
