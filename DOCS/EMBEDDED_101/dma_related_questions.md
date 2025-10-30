## Explain how DMA works. What are some of the issues that you need to worry about when using DMA?

DMA is a hardware feature that allows data transfer between memory and peripherals (or memory to memory) without CPU intervention.
Normally, the CPU would do something like:

while (count--)
*dst++ = *src++;

But with DMA, the CPU tells a dedicated DMA controller to perform this transfer autonomously.

### How DMA Works — Step by Step:

- CPU sets up the DMA controller:

  - Source address (e.g., ADC data register, or memory buffer)
  - Destination address (e.g., RAM buffer, or UART data register)

- Transfer size (number of bytes/words)

  - Direction (peripheral → memory, memory → peripheral, or memory ↔ memory)
  - Transfer mode (single, circular, burst, etc.)

- DMA transfer starts:

  - A peripheral event (like ADC conversion complete or UART TX empty) triggers the DMA transfer.
  - The DMA controller reads data from the source and writes it to the destination directly via the system bus, bypassing the CPU.

- DMA completion:
  - After all transfers, the DMA generates an interrupt (e.g., “transfer complete” or “half complete”).
  - The CPU can then process the data (for example, analyze ADC samples or refill a TX buffer).

### Common Issues to Watch Out For

#### Memory alignment and data size

- DMA often requires word-aligned addresses.
- If your source/destination isn’t properly aligned, transfers may fail or give corrupted data.

Example: Transferring bytes to a 32-bit peripheral register can cause misalignment faults.

#### Cache coherence

- On MCUs with data cache (like Cortex-M7):
- The DMA reads/writes directly to RAM, bypassing the CPU cache.
- The CPU might see stale data if the cache isn’t invalidated or cleaned.

Fix: use SCB_CleanDCache_by_Addr() before TX and SCB_InvalidateDCache_by_Addr() after RX.

#### Buffer overrun / underrun

- If the peripheral produces data faster than the DMA can transfer, or consumes data faster than the DMA provides it, you get data loss.
- Example: UART RX overrun or I2S underrun.

#### Memory region accessibility

- Not all memory regions are DMA-accessible.
- For example, on STM32:
  - DMA cannot access Flash directly (only RAM and peripheral memory).
- Some peripherals only work with specific DMA channels or streams.

#### Circular and double-buffer mode pitfalls

- Circular mode is great for continuous streams (e.g., ADC sampling), but:
- The buffer content keeps being overwritten.
- You must manage “which half” of the buffer is valid at any moment (using half-transfer interrupts).

#### Bus contention

- The DMA controller and CPU share the same memory bus.
- Heavy DMA activity can slow down CPU memory access (e.g., when both access SRAM heavily).

#### Interrupt handling and synchronization

- DMA operations are asynchronous.
- You must wait for the “transfer complete” interrupt before using the data.
- Accessing data early may give partial or invalid results.

## What is Scatter-Gather DMA? What is Ping-Pong DMA?

Scatter-Gather DMA is a DMA mode that allows non-contiguous memory regions to be transferred as one logical operation by using a linked list of DMA descriptors.

### How It Works

- You create a list of descriptors, where each descriptor contains:

  - Source address
  - Destination address
  - Transfer size
  - Pointer to next descriptor

- The DMA engine follows the chain automatically — when one block finishes, it fetches the next descriptor and continues.
