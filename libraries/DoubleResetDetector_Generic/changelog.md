## DoubleResetDetector_Generic

[![arduino-library-badge](https://www.ardu-badge.com/badge/DoubleResetDetector_Generic.svg?)](https://www.ardu-badge.com/DoubleResetDetector_Generic)
[![GitHub release](https://img.shields.io/github/release/khoih-prog/DoubleResetDetector_Generic.svg)](https://github.com/khoih-prog/DoubleResetDetector_Generic/releases)
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/khoih-prog/DoubleResetDetector_Generic/blob/master/LICENSE)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#Contributing)
[![GitHub issues](https://img.shields.io/github/issues/khoih-prog/DoubleResetDetector_Generic.svg)](http://github.com/khoih-prog/DoubleResetDetector_Generic/issues)

---
---

## Table of Contents

* [Changelog](#changelog)
  * [Releases v1.8.1](#releases-v181)
  * [Releases v1.8.0](#releases-v180)
  * [Releases v1.7.3](#releases-v173)
  * [Releases v1.7.2](#releases-v172)
  * [Releases v1.7.1](#releases-v171)
  * [Major Releases v1.7.0](#major-releases-v170)
  * [Major Releases v1.6.0](#major-releases-v160)
  * [Major Releases v1.5.0](#major-releases-v150)
  * [Major Releases v1.4.0](#major-releases-v140)
  * [Major Releases v1.3.0](#major-releases-v130)
  * [Major Releases v1.2.0](#major-releases-v120)
  * [Major Releases v1.1.0](#major-releases-v110)
  * [Releases v1.0.3](#releases-v103)
  * [Releases v1.0.2](#releases-v102)
  * [Releases v1.0.1](#releases-v101)
  * [Releases v1.0.0](#releases-v100)

---
---

## Changelog

### Releases v1.8.1

1. Add waitingForDRD() function to signal in DRD wating period. Check [DRD is waiting for a double reset? #14](https://github.com/khoih-prog/ESP_DoubleResetDetector/discussions/14)
2. Add example [checkWaitingDRD](https://github.com/khoih-prog/DoubleResetDetector_Generic/tree/master/examples/checkWaitingDRD) to demo how to use the new feature.
3. Add support to `Sparkfun Pro nRF52840 Mini`
4. Update `Packages' Patches`

### Releases v1.8.0

1. Update to be compatible with new `FlashStorage_SAMD`, `FlashStorage_STM32` and `FlashStorage_STM32F1` libraries.
2. Use more efficient `FlashStorage_STM32` and `FlashStorage_STM32F1` libraries for STM32
3. Add support to more `SAMD` and `STM32` boards
4. Update `Packages' Patches`

### Releases v1.7.3

1. Update `platform.ini` and `library.json` to use original `khoih-prog` instead of `khoih.prog` after PIO fix
2. Update `Packages' Patches`

### Releases v1.7.2

1. Back to using auto LittleFS size to fix bug and to permit 8 LittleFS files instead of 4 in previous release. There is a bug somewhere in the [`ArduinoCore-mbed mbed_portenta core`](https://github.com/arduino/ArduinoCore-mbed) and we have to live with it.

### Releases v1.7.1

1. Change the default and minimum LITTLEFS_PORTENTA_H7_SIZE_KB to 1024KB, instead at maximum available size. This is done to avoid the cases where the code size is larger, and LittleFS size has to be changed, reformatted, leading to data loss.

### Major Releases v1.7.0

1. Add support to **Portenta_H7** using [**Arduino mbed_portenta core**](https://github.com/arduino/ArduinoCore-mbed).
2. Update `Packages' Patches` for **Portenta_H7**

### Major Releases v1.6.0

1. Add support to **MBED nRF52840-based boards such as Nano_33_BLE, Nano_33_BLE_Sense**, etc. using [**Arduino-mbed mbed_nano** core](https://github.com/arduino/ArduinoCore-mbed)

### Major Releases v1.5.0

1. Add support to Realtek RTL8720DN, RTL8722DM, RTL8722CSM, etc., using [`Arduino AmebaD core`](https://github.com/ambiot/ambd_arduino)

### Major Releases v1.4.0

1. Permit more control over LittleFS for RP2040 Arduino mbed core, such as `FORCE_REFORMAT`, `MBED_RP2040_INITIALIZED`, `DRD_MBED_LITTLEFS_NEED_INIT`.


### Major Releases v1.3.0

1. Add support to RP2040-based boards, such as **Nano RP2040 Connect**, using [**Arduino mbed OS for Nano boards**](https://github.com/arduino/ArduinoCore-mbed) and LittleFS.
1. Add support to RP2040-based boards, such as **RASPBERRY_PI_PICO, ADAFRUIT_FEATHER_RP2040 and GENERIC_RP2040**, using [**Arduino-mbed RP2040** core](https://github.com/arduino/ArduinoCore-mbed) and LittleFS.


### Major Releases v1.2.0

1. Add support to RP2040-based boards, such as **RASPBERRY_PI_PICO, ADAFRUIT_FEATHER_RP2040 and GENERIC_RP2040**, using [Earle Philhower's arduino-pico core](https://github.com/earlephilhower/arduino-pico) and LittleFS.

### Major Releases v1.1.0

1. Use new efficient [**FlashStorage_STM32** library](https://github.com/khoih-prog/FlashStorage_STM32). 
2. Add support to new **STM32 core v2.0.0** and STM32L5


### Releases v1.0.3

1. Clean-up all compiler warnings possible.
2. Add Table of Contents
3. Modify Version String 

### Releases v1.0.2

1. Fix not-detected DRD bug for SAMD21 and SAMD51 boards.

### Releases v1.0.1

1. Support **nRF52** boards such as **Adafruit NRF52840_FEATHER, NRF52832_FEATHER, NRF52840_FEATHER_SENSE, NRF52840_ITSYBITSY, NRF52840_CIRCUITPLAY, NRF52840_CLUE, NRF52840_METRO, NRF52840_PCA10056, PARTICLE_XENON, NINA_B302_ublox**, etc.
2. nRF52 boards to save data in **Adafruit's InternalFS/LittleFS**.

### Releases v1.0.0

1. Support boards such as **AVR, Teensy, SAM DUE, SAMD21, SAMD51 and STM32F/L/H/G/WB/MP1, etc.**
2. **AVR Mega, Teensy, STM32F/L/H/G/WB/MP1** to save data in **EPPROM**
3. **SAMD** to save data in EEPROM-simulated [**FlashStorage_SAMD**](https://github.com/khoih-prog/FlashStorage_SAMD)
4. **SAM DUE** to save data in [**DueFlashStorage**](https://github.com/sebnil/DueFlashStorage)


