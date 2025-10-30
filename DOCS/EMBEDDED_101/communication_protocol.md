## What is I2S? Where is it used? Why might you want to use I2S in an embedded system? Have you ever used it?

### What is I2S?

- I²S (Inter-IC Sound) is a digital audio interface standard designed for transmitting PCM audio data between ICs.
- Standard: Philips / NXP I2S specification.
- Purpose: Connect digital audio devices like codecs, DACs, ADCs, and DSPs.

- Not a general-purpose bus: Unlike I²C or SPI, I²S is audio-specific.

- Main Signals in I2S

| Signal                | Description                                |
| --------------------- | ------------------------------------------ |
| **SCK / BCLK**        | Serial clock for audio data bits           |
| **WS / LRCK**         | Word select (Left/Right channel indicator) |
| **SD / SDIN / SDOUT** | Serial data (audio samples)                |
| **MCLK**              | Optional master clock for DAC/ADC          |

### Where is I2S Used?

- Audio codecs in embedded systems (microphones, DACs, ADCs)
- MP3/WAV playback devices
- Digital microphones / MEMS microphones
- Voice processing in smart assistants, intercoms, and phones
- Audio DSPs and amplifiers

### Why Use I2S in an Embedded System?

| Reason                 | Explanation                                                             |
| ---------------------- | ----------------------------------------------------------------------- |
| **High-quality audio** | Transmits PCM audio directly; no analog signal degradation.             |
| **Simplified design**  | Digital interface avoids need for ADC/DAC on MCU side.                  |
| **Multiple channels**  | Easily supports stereo or multi-channel audio.                          |
| **Hardware support**   | Many MCUs and SoCs have **dedicated I2S peripherals** for DMA transfer. |
| **Synchronization**    | Word select and clock signals ensure audio data is properly timed.      |

### Example Use in Embedded System

- MCU → I2S DAC → Speaker: Playing audio.
- I2S MEMS microphone → MCU: Capturing audio for voice recognition.
- Often paired with DMA to transfer audio buffers efficiently without CPU overhead.

## What is CAN, LIN, FlexRay? Where are they used? Have you ever used any?

| Protocol | Speed              | Topology         | Features                      | Typical Use                        |
| -------- | ------------------ | ---------------- | ----------------------------- | ---------------------------------- |
| CAN      | 1 Mbps (FD higher) | Multi-master bus | Error detection, robust       | Engine, brakes, body electronics   |
| LIN      | ~20 Kbps           | Master-slave     | Low cost, simple              | Windows, seats, HVAC               |
| FlexRay  | 10 Mbps            | Dual-channel bus | Deterministic, fault-tolerant | Safety-critical automotive systems |

## What is ARINC 429? Where is it commonly used? Have you ever used it?

- ARINC 429 is a data bus standard for aircraft avionics.
- It is a unidirectional, point-to-point serial bus used for digital communication between avionics equipment.
- Developed by Aeronautical Radio, Inc. (ARINC).

## How SPI daisy chaining work?

### Concept

- In SPI daisy chaining, multiple slave devices are connected in series, not parallel.
- The MISO (data out) of one device connects to the MOSI (data in) of the next.
- Only one chip select (CS) line is used for all devices.
- Common clock line for all devices.

### How It Works

- The master sends a long data stream equal to the total bits of all devices.
- Each clock pulse shifts data through every device’s internal shift register:

  - Data for the last slave exits first.
  - Data for the first slave enters last.

- After transmission, a latch (CS rising edge) updates all devices simultaneously.

## Why is SPI faster than I2C? what is clock stretching?

| Feature               | **SPI**                          | **I²C**                                                     |
| --------------------- | -------------------------------- | ----------------------------------------------------------- |
| **Signaling**         | Push-pull (driven high & low)    | Open-drain (pulled high via resistors)                      |
| **Clock control**     | Master drives clock continuously | Slave can pause via _clock stretching_                      |
| **Protocol overhead** | None — just raw bytes            | Extra overhead: addressing, ACK bits                        |
| **Speed**             | Tens of MHz (commonly 10–50 MHz) | Standard (100 kHz), Fast (400 kHz), up to 3.4 MHz (HS mode) |
| **Lines needed**      | 4 (MOSI, MISO, SCK, CS)          | 2 (SDA, SCL)                                                |

## How will you figure out if 2 salve of same addresses are connected to I2C?

If two slaves share an I²C address:

- There will be bus errors, ACK conflicts, or corrupted reads.( Error detection is IP dependent)

- To detect it: we can run an I²C scan + check waveforms.
- To fix it: we can change address or use a multiplexer.

## Why does high pullup increases rise time?

When the line is released (logic high), the capacitive load of the bus charges through the pull-up resistor.

The rise time depends on the RC time constant:

t<sub>rise</sub>​≈0.693 ⋅ R<sub>pullup​</sub> ⋅ C<sub>bus​</sub>

Higher resistance → slower charging of the bus capacitance → longer rise time.
Slower rise time can limit maximum data rate and cause signal integrity issues.

## What are two of the hardware protocols used to communicate with SD cards? Which will most likely work with more microcontrollers?

### Hardware Protocols for SD Cards

| Protocol                              | Description                                                                                                             | Notes                                                                                                                          |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **SD (1-bit / 4-bit) Bus Mode**       | Uses dedicated data lines (DAT0–DAT3), CMD, and CLK pins. Native SD mode; allows 1-bit or 4-bit parallel data transfer. | Mostly used in **dedicated SD controllers**. Higher throughput in 4-bit mode.                                                  |
| **SPI (Serial Peripheral Interface)** | SD card operates as an SPI slave; uses MOSI, MISO, SCK, CS lines.                                                       | Widely supported because **almost every MCU has an SPI peripheral**. Lower speed than native SD mode, but easier to implement. |
