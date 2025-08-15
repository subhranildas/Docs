## Communication and Synchronization in RTOS

### Introduction

In a **Real-Time Operating System (RTOS)**, tasks often need to **share data**, **coordinate execution**, and **access common resources**.  
To achieve this, RTOS kernels provide **communication** and **synchronization** mechanisms.
These mechanisms ensure:

- Data consistency
- Avoidance of race conditions
- Proper sequencing of dependent tasks
- Efficient CPU utilization

---

### Communication in RTOS

#### Purpose

Inter-task communication is used when tasks need to **exchange data** or **send events** without direct shared memory conflicts.

#### Common Communication Mechanisms

1. **Message Queues**

   - Stores messages in FIFO order.
   - Used for passing structured data between tasks.
   - Can be blocking or non-blocking.

2. **Mailboxes**

   - Holds a single message at a time.
   - Often used for lightweight, point-to-point communication.

3. **Pipes / Streams**

   - Byte-oriented data flow channels.
   - Good for streaming sensor data or logging.

4. **Event Flags / Event Groups**

   - Tasks wait for specific event bits to be set.
   - Supports multiple event conditions (OR/AND logic).

5. **Shared Memory**
   - Multiple tasks access a common memory region.
   - Requires synchronization mechanisms (e.g., mutexes) to ensure safety.

---

### Synchronization in RTOS

#### Purpose

Synchronization ensures **orderly access to shared resources** and **correct sequencing of dependent tasks**.

#### Common Synchronization Mechanisms

1. **Semaphores**

   - **Binary Semaphore**: Acts like a lock/unlock signal.
   - **Counting Semaphore**: Tracks resource availability count.
   - Often used for signaling between tasks or between ISR and task.

2. **Mutexes (Mutual Exclusion)**

   - Prevents multiple tasks from accessing a resource at the same time.
   - Supports **priority inheritance** to avoid **priority inversion**.

3. **Condition Variables**

   - Allows tasks to wait for a certain condition to become true.
   - Often used with mutexes.

4. **Barriers**
   - Blocks tasks until a predefined number of tasks reach the barrier point.

---

### Communication vs Synchronization

| Aspect                | Communication                      | Synchronization                         |
| --------------------- | ---------------------------------- | --------------------------------------- |
| **Purpose**           | Exchange data/events between tasks | Coordinate timing and resource access   |
| **Example**           | Message Queue, Mailbox             | Semaphore, Mutex                        |
| **Data Transfer**     | Yes                                | No (only signaling)                     |
| **Blocking Behavior** | Can be blocking or non-blocking    | Usually blocking until condition is met |

---

### Best Practices

- **Use mutexes** for shared data protection, not semaphores.
- **Avoid busy-wait loops**; use blocking mechanisms for efficiency.
- **Prioritize ISR-to-task communication** using semaphores or queues.
- **Avoid deadlocks** by defining a consistent lock acquisition order.
- **Minimize critical section duration** to reduce priority inversion risk.

---

## FreeRTOS Inter-Task Communication and Synchronization

In embedded systems powered by FreeRTOS, efficient communication and
synchronization between tasks are critical for system reliability,
responsiveness, and robustness. FreeRTOS offers a set of powerful primitives to
achieve these goals, including **Queues**, **Semaphores**, **Mutexes**, and
**Event Groups**.

---

### FreeRTOS Queues: Communication Between Tasks

#### What Are FreeRTOS Queues?

FreeRTOS Queues are used for sending messages or data from one task to another.
They act like thread-safe FIFO (First-In-First-Out) buffers.

#### Why Use FreeRTOS Queues?

- Task-to-task data transfer.
- ISR (Interrupt Service Routine) to task communication.
- Avoids shared memory issues by decoupling sender and receiver.

#### Example: Sending Sensor Data from ISR to Logger Task

```c
#include "FreeRTOS.h"
#include "queue.h"
#include "task.h"

QueueHandle_t xSensorQueue;

void vLoggerTask(void *pvParameters) {
    int sensorData;
    while (1) {
        if (xQueueReceive(xSensorQueue, &sensorData, portMAX_DELAY)) {
            printf("Logging sensor data: %d\n", sensorData);
        }
    }
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
    static int value = 0;
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xQueueSendFromISR(xSensorQueue, &value, &xHigherPriorityTaskWoken);
    value++;
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

int main() {
    xSensorQueue = xQueueCreate(10, sizeof(int));
    xTaskCreate(vLoggerTask, "Logger", 128, NULL, 2, NULL);
    vTaskStartScheduler();
    while (1);
}
```

---

### FreeRTOS Semaphores: Task Synchronization

#### What Are FreeRTOS Semaphores?

Semaphores in FreeRTOS help coordinate execution between tasks or between ISRs
and tasks. A binary semaphore is typically used for signaling, while a counting
semaphore can allow multiple resources.

#### Why Use FreeRTOS Semaphores?

- Event notification (e.g., button press).
- Synchronize access between ISRs and tasks.
- Signal task completion.

#### Example: Binary Semaphore for Button Press Notification

```c
SemaphoreHandle_t xButtonSemaphore;

void vButtonTask(void *pvParameters) {
    while (1) {
        if (xSemaphoreTake(xButtonSemaphore, portMAX_DELAY)) {
            printf("Button Pressed!\n");
        }
    }
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xSemaphoreGiveFromISR(xButtonSemaphore, &xHigherPriorityTaskWoken);
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

int main() {
    xButtonSemaphore = xSemaphoreCreateBinary();
    xTaskCreate(vButtonTask, "ButtonTask", 128, NULL, 2, NULL);
    vTaskStartScheduler();
    while (1);
}
```

---

### FreeRTOS Mutexes: Protecting Shared Resources

#### What Are FreeRTOS Mutexes?

Mutexes are used to ensure **mutual exclusion**, allowing only one task at a
time to access shared resources like UART, memory, or I2C buses.

#### Why Use FreeRTOS Mutexes?

- Prevent race conditions.
- Protect shared peripherals and memory.
- Priority inheritance prevents priority inversion.

#### Example: Multiple Tasks Logging to UART

```c
SemaphoreHandle_t xUartMutex;

void vTask1(void *pvParameters) {
    while (1) {
        if (xSemaphoreTake(xUartMutex, portMAX_DELAY)) {
            printf("Task 1 writing to UART\n");
            vTaskDelay(pdMS_TO_TICKS(100));
            xSemaphoreGive(xUartMutex);
        }
        vTaskDelay(pdMS_TO_TICKS(200));
    }
}

void vTask2(void *pvParameters) {
    while (1) {
        if (xSemaphoreTake(xUartMutex, portMAX_DELAY)) {
            printf("Task 2 writing to UART\n");
            vTaskDelay(pdMS_TO_TICKS(50));
            xSemaphoreGive(xUartMutex);
        }
        vTaskDelay(pdMS_TO_TICKS(150));
    }
}

int main() {
    xUartMutex = xSemaphoreCreateMutex();
    xTaskCreate(vTask1, "Task1", 128, NULL, 2, NULL);
    xTaskCreate(vTask2, "Task2", 128, NULL, 2, NULL);
    vTaskStartScheduler();
    while (1);
}
```

---

### FreeRTOS Event Groups: Bitwise Task Synchronization

#### What Are FreeRTOS Event Groups?

Event Groups allow tasks to synchronize on multiple events using **bit flags**.
This is ideal for scenarios where tasks must wait for multiple conditions to be
true.

#### Why Use FreeRTOS Event Groups?

- Wait for multiple events (AND/OR logic).
- Lightweight and flexible.
- Better than polling flags manually.

#### Example: Task Waits for WiFi and Sensor Ready

```c
#include "event_groups.h"

#define WIFI_READY     (1 << 0)
#define SENSOR_READY   (1 << 1)

EventGroupHandle_t xEventGroup;

void vInitWiFi(void *pvParameters) {
    // Simulate WiFi Init
    vTaskDelay(pdMS_TO_TICKS(500));
    xEventGroupSetBits(xEventGroup, WIFI_READY);
}

void vInitSensor(void *pvParameters) {
    // Simulate Sensor Init
    vTaskDelay(pdMS_TO_TICKS(800));
    xEventGroupSetBits(xEventGroup, SENSOR_READY);
}

void vAppTask(void *pvParameters) {
    EventBits_t uxBits;
    uxBits = xEventGroupWaitBits(xEventGroup, WIFI_READY | SENSOR_READY, pdTRUE, pdTRUE, portMAX_DELAY);
    if ((uxBits & (WIFI_READY | SENSOR_READY)) == (WIFI_READY | SENSOR_READY)) {
        printf("WiFi and Sensor are ready. Starting main application...\n");
    }
}

int main() {
    xEventGroup = xEventGroupCreate();
    xTaskCreate(vInitWiFi, "WiFiInit", 128, NULL, 2, NULL);
    xTaskCreate(vInitSensor, "SensorInit", 128, NULL, 2, NULL);
    xTaskCreate(vAppTask, "AppTask", 128, NULL, 1, NULL);
    vTaskStartScheduler();
    while (1);
}
```

---

### Conclusion

FreeRTOS provides a rich set of inter-task communication and synchronization tools:

| Feature      | Best For                                 | Synchronization | Communication |
| ------------ | ---------------------------------------- | --------------- | ------------- |
| Queues       | Data transfer between tasks or from ISRs | ✗               | ✅            |
| Semaphores   | Signaling between tasks/ISR              | ✅              | ✗             |
| Mutexes      | Exclusive access to shared resources     | ✅              | ✗             |
| Event Groups | Multi-condition task synchronization     | ✅              | ✗             |

Choosing the right tool for your use case improves task coordination and overall system stability. FreeRTOS makes this easier with simple and powerful APIs tailored for real-time embedded applications.
