## Introduction

Over the years, I²C (Inter-Integrated Circuit) has been the go-to protocol for connecting sensors, memory devices, and controllers.
Growing demand for higher speed, power efficiency, and interoperability, the MIPI Alliance introduced I³C (Improved Inter-Integrated Circuit).

This blog explains the fundamentals of the I3C protocol, its advantages, differences from I²C, and the working mechanism—with diagrams for better understanding.

## Versions of I3C Protocol

### I3C v1.0 (2016)

The very first version released by the MIPI Alliance.
It was designed to replace I²C in sensor connectivity while being backward compatible.
Provides a unified protocol that works where I²C was traditionally used, but faster and more power-efficient.

Key Features:

- Two-wire bus (SDA, SCL).
- Backward compatibility with I²C devices.
- Dynamic Address Assignment (DAA) for devices.
- In-Band Interrupts (IBI) → No extra interrupt pin required.
- Hot-Join support (devices can join after initialization).
- SDR Mode (up to 12.5 Mbps).
- Basic HDR-DDR support (initial high-speed mode).

### I3C v1.1 (2019)

This was an enhancement release to improve scalability and flexibility.
Made I3C suitable for complex embedded systems (mobile, IoT, automotive sensors).

Improvements Over v1.0:

- HDR Modes expanded:
- HDR-DDR (Double Data Rate).
- HDR-TSP (Ternary Symbol Polarity).
- HDR-TSL (Ternary Symbol Legacy) → backward compatible with legacy I²C traffic.
- Group Addressing – Controller can broadcast commands to groups of devices.
- Improved multi-controller support (bus handoff and shared bus scenarios).
- Increased robustness for larger device networks.
- Optional timing control improvements for better signal integrity.

### I3C Basic (2018, maintained separately)

This is a subset of I3C v1.0/v1.1 published as a freely available specification (unlike the full standard, which requires MIPI membership).
Encourage adoption outside the MIPI ecosystem, especially for IoT and embedded developers who don’t want licensing barriers.

Key Features:

- Designed to replace I²C in open ecosystems.
- Backward compatibility with I²C.
- Dynamic Address Assignment.
- In-Band Interrupts.
- Hot-Join.
- Standard Data Rate (up to 12.5 Mbps).
- Excludes HDR modes (to keep it simpler).

### I3C v1.1.1 (2021)

A maintenance and clarification release that refined the specification.
Ensures smoother industry adoption with fewer implementation errors.

Key Points:

- Fixed ambiguities in v1.1.
- Improved compatibility guidelines for I²C and mixed systems.
- Clarified multi-controller operation rules.
- Strengthened documentation of electrical/timing characteristics.

### I3C v1.1.2 (2022)

Another maintenance release with incremental improvements.
Polishes the protocol for scalability in complex systems.

Key Points:

- Expanded clarifications on HDR usage.
- Enhanced hot-join handling.
- Added guidance for robust large networks.

### I3C v1.1.3 (2023)

Latest update release before v2.0.

Key Points:

- Introduced minor electrical enhancements.
- Expanded documentation on multi-controller arbitration.
- Added recommendations for low-power states in IoT.

### I3C v2.0 (2023/2024)

The most recent major release (announced mid-2023, rolling into 2024 adoption).
This version significantly expands the capabilities of I3C.
Evolves I3C into a universal sensor and peripheral interconnect standard, competing with SPI in performance while keeping I²C simplicity.

Key Features:

- Improved Bus Speeds: Higher SDR and HDR rates.
- Advanced Multi-Drop Support: better handling of very large sensor networks.
- Improved Multi-Controller Operation: smoother handoff and concurrent bus ownership.
- Standardized Profiles for automotive, industrial, and IoT systems.
- Enhanced error handling and robustness.
- Refined power-saving mechanisms (optimized for ultra-low-power IoT).

---

## Bus Conditions and Dynamic Definition in I3C

The bus condition refers to the logical state of the shared bus lines (SDA and SCL) and how those states define events like start, stop, idle, or data transfer.
Unlike static or predefined bus conditions, I3C introduces the concept of dynamic definitions, where certain bus conditions (e.g., addressing, interrupts, or hot-join events) are defined at runtime depending on the devices present and the roles assigned.

---

### Standard Bus Conditions

![Start, Stop & Repeated Start Conditions](Images/i2c_start_stop_r_start.svg)

#### Idle Condition

The bus is considered idle when both SDA and SCL lines are released (logic high).
During this no device is driving the bus.

#### Start Condition (S)

A Start condition occurs when SDA transitions from HIGH → LOW while SCL remains HIGH.
This indicates the beginning of a transaction.

#### Stop Condition (P)

A Stop condition occurs when SDA transitions from LOW → HIGH while SCL is HIGH.
This Indicates the end of a transaction; bus is released to idle.

#### Repeated Start Condition (Sr)

Similar to Start, but issued without releasing the bus (no Stop in between).
Allows the controller to maintain control for back-to-back operations.

---

### Dynamic Bus Definitions in I3C

Unlike I²C, which has static addressing and fixed interpretation of bus conditions, I3C supports dynamic behavior on the same physical bus.
This is where Dynamic Bus Conditions come into play.

#### Dynamic Address Assignment (DAA)

After power-up, I3C devices only have a provisional identifier (PID).
The controller performs a bus-wide broadcast to discover devices.
Devices respond dynamically, and the controller assigns unique 7-bit dynamic addresses at runtime.
Dynamic Bus Condition: The meaning of address phases is defined dynamically, depending on the assigned address.

#### In-Band Interrupts (IBI)

In I²C, interrupts require dedicated pins.
In I3C, targets can assert an interrupt request over SDA while the bus is idle.
The bus dynamically redefines Start + IBI request as an interrupt condition instead of normal data transfer.

#### Hot-Join

Devices can join the bus dynamically after it has been initialized.
They pull SDA low during a defined bus condition (while SCL is free), signaling the controller to start the Hot-Join sequence.
The bus redefines this condition dynamically as a join request rather than a protocol violation.

#### Multi-Controller Handoff

In I3C, multiple controllers can coexist.
A special bus condition sequence allows a controller to release bus ownership, and another to dynamically acquire it.
This dynamic definition enables seamless handover without requiring a full reset.

<!--
1. What is I3C?

I3C is a communication protocol developed by the MIPI Alliance to standardize sensor connectivity and improve upon the limitations of I²C and SPI.

Backwards compatible with I²C (supports legacy devices).

Provides higher data rates (up to 33.3 Mbps HDR mode).

Lower power consumption through in-band interrupts and efficient bus management.

Designed for mobile, IoT, automotive, and industrial systems.

2. I3C Bus Overview

The I3C bus, like I²C, is a two-wire bus:

SDA (Serial Data Line) – bidirectional data line.

SCL (Serial Clock Line) – clock signal from the controller.

But unlike I²C, I3C allows dynamic addressing, in-band interrupts, and multi-controller capability.

Diagram 1: I3C Bus Basic Layout
Controller (Master)
|

---

| | |
Device Device Device
(Target) (Target) (Target)

(All devices share SDA and SCL lines, but I3C allows smarter arbitration and faster data transfers compared to I²C.)

3. Key Features of I3C

Backward Compatibility with I²C

I3C devices can coexist with I²C devices on the same bus.

Default startup mode is I²C-compatible.

Higher Speed

Standard Data Rate (SDR): up to 12.5 Mbps.

High Data Rate (HDR): up to 33.3 Mbps.

Compared to I²C max ~3.4 Mbps.

Dynamic Address Assignment (DAA)

Unlike I²C’s fixed 7-bit/10-bit addresses, I3C assigns addresses dynamically during initialization.

In-Band Interrupts (IBI)

Targets can signal the controller directly over SDA (no extra GPIO needed).

Hot-Join Capability

Devices can join the bus dynamically after startup.

Diagram 2: I3C vs I²C Bus Signals
I²C: I3C:

- Open Drain - Push-Pull (for higher speeds)
- Static Addr - Dynamic Addr
- Extra INT pin - In-band Interrupts

(Visually: Show two side-by-side buses, one with extra INT line for I²C devices, and one with I3C devices using SDA for interrupts.)

4. I3C Communication Flow
   Step 1: Bus Initialization

Controller powers up in I²C-compatible mode.

Performs Dynamic Address Assignment (DAA) to give each device a unique address.

Step 2: Data Transfer

Data is exchanged in SDR mode by default.

For faster operations, it can switch to HDR modes (HDR-DDR, HDR-TSP, HDR-TSL).

Step 3: Interrupts and Hot-Join

Targets can send IBI to the controller without needing extra wires.

New devices can hot-join and request an address dynamically.

Diagram 3: Dynamic Address Assignment
Controller: "Who is on the bus?"
Device A: "I’m here → Assign 0x20"
Device B: "I’m here → Assign 0x21"
Device C: "I’m here → Assign 0x22"

(Visually: Show Controller polling devices, assigning addresses dynamically in sequence.)

5. I3C High Data Rate (HDR) Modes

I3C supports multiple HDR modes for boosting speed:

HDR-DDR (Double Data Rate) – Transfers data on both clock edges.

HDR-TSP (Ternary Symbol Polarity) – Uses three signal levels for encoding.

HDR-TSL (Ternary Symbol Legacy) – Compatible with I²C devices.

These modes allow I3C to reach up to 33.3 Mbps, far beyond I²C.

Diagram 4: SDR vs HDR Timing
SDR: Data changes only on rising/falling edges
HDR: Data encoded on both edges → Higher throughput

(Visually: Two timing diagrams showing SDR using one edge per cycle, HDR using both edges.)

6. Advantages of I3C Over I²C and SPI
   Feature I²C SPI I³C
   Wires 2 4+ 2
   Speed Up to 3.4 Mbps Up to 50 Mbps Up to 33.3 Mbps (HDR)
   Addressing Static Chip Select Dynamic
   Interrupt Handling Extra Pin Extra Pin In-band (over SDA)
   Power Efficiency Moderate High Optimized (push-pull driving)
   Multi-controller Limited Yes Yes
7. Real-World Applications of I3C

Smartphones/Tablets → Sensors (accelerometers, gyros, magnetometers).

Automotive → In-vehicle sensor fusion for ADAS.

IoT Devices → Low-power multi-sensor hubs.

Wearables → Optimized power usage and fewer pins.

8. Conclusion

I3C is a game-changer in sensor communication, offering the simplicity of I²C with the performance of SPI. With features like dynamic addressing, in-band interrupts, hot-join, and HDR modes, it is rapidly becoming the go-to protocol for modern embedded and IoT applications.

For engineers designing the next generation of connected devices, I3C is worth exploring as a future-proof solution. -->
