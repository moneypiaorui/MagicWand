## DoubleResetDetector_Generic

[![arduino-library-badge](https://www.ardu-badge.com/badge/DoubleResetDetector_Generic.svg?)](https://www.ardu-badge.com/DoubleResetDetector_Generic)
[![GitHub release](https://img.shields.io/github/release/khoih-prog/DoubleResetDetector_Generic.svg)](https://github.com/khoih-prog/DoubleResetDetector_Generic/releases)
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/khoih-prog/DoubleResetDetector_Generic/blob/master/LICENSE)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#Contributing)
[![GitHub issues](https://img.shields.io/github/issues/khoih-prog/DoubleResetDetector_Generic.svg)](http://github.com/khoih-prog/DoubleResetDetector_Generic/issues)

<a href="https://www.buymeacoffee.com/khoihprog6" title="Donate to my libraries using BuyMeACoffee"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Donate to my libraries using BuyMeACoffee" style="height: 50px !important;width: 181px !important;" ></a>
<a href="https://www.buymeacoffee.com/khoihprog6" title="Donate to my libraries using BuyMeACoffee"><img src="https://img.shields.io/badge/buy%20me%20a%20coffee-donate-orange.svg?logo=buy-me-a-coffee&logoColor=FFDD00" style="height: 20px !important;width: 200px !important;" ></a>

---
---

## Table of Contents

* [Important Notes for Portenta_H7](#important-notes-for-portenta_h7)
* [Why do we need this DoubleResetDetector_Generic library](#why-do-we-need-this-doubleresetdetector_generic-library)
  * [Features](#features)
  * [Currently supported Boards](#currently-supported-boards)
* [Changelog](changelog.md)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
  * [Use Arduino Library Manager](#use-arduino-library-manager)
  * [Manual Install](#manual-install)
  * [VS Code & PlatformIO](#vs-code--platformio)
* [Packages' Patches](#packages-patches)
  * [1. For Adafruit nRF52840 and nRF52832 boards](#1-for-adafruit-nRF52840-and-nRF52832-boards)
  * [2. For Teensy boards](#2-for-teensy-boards)
  * [3. For Arduino SAM DUE boards](#3-for-arduino-sam-due-boards)
  * [4. For Arduino SAMD boards](#4-for-arduino-samd-boards)
      * [For core version v1.8.10+](#for-core-version-v1810)
      * [For core version v1.8.9-](#for-core-version-v189-)
  * [5. For Adafruit SAMD boards](#5-for-adafruit-samd-boards)
  * [6. For Seeeduino SAMD boards](#6-for-seeeduino-samd-boards)
  * [7. For STM32 boards](#7-for-stm32-boards) 
    * [7.1. For STM32 boards to use LAN8720](#71-for-stm32-boards-to-use-lan8720)
    * [7.2. For STM32 boards to use Serial1](#72-for-stm32-boards-to-use-serial1)
  * [8. For RP2040-based boards using Earle Philhower arduino-pico core](#8-for-rp2040-based-boards-using-earle-philhower-arduino-pico-core)
    * [8.1. To use BOARD_NAME](#81-to-use-board_name)
    * [8.2. To avoid compile error relating to microsecondsToClockCycles](#82-to-avoid-compile-error-relating-to-microsecondstoclockcycles)
  * [9. For Portenta_H7 boards using Arduino IDE in Linux](#9-for-portenta_h7-boards-using-arduino-ide-in-linux)
* [Purpose](#purpose)
* [How It Works](#how-it-works)
* [Examples](#examples)
  * [ 1. minimal](examples/minimal)
  * [ 2. checkWaitingDRD](examples/checkWaitingDRD) **New**
  * [Many other libraries are depending on this library's DRD feature](#many-other-libraries-are-depending-on-this-librarys-drd-feature)
    * [ 1. BlynkEthernet_WM](https://github.com/khoih-prog/BlynkEthernet_WM)
    * [ 2. Blynk_Esp8266AT_WM](https://github.com/khoih-prog/Blynk_Esp8266AT_WM)
    * [ 3. Blynk_WiFiNINA_WM](https://github.com/khoih-prog/Blynk_WiFiNINA_WM)
    * [ 4. WiFiManager_NINA_Lite](https://github.com/khoih-prog/WiFiManager_NINA_Lite)
    * [ 5. BlynkEthernet_STM32_WM](https://github.com/khoih-prog/BlynkEthernet_STM32_WM)
    * [ 6. ESP_AT_WM_Lite](https://github.com/khoih-prog/ESP_AT_WM_Lite)
    * [ 7. WIOTerminal_WiFiManager](https://github.com/khoih-prog/WIOTerminal_WiFiManager)
    * [ 8. Ethernet_Manager](https://github.com/khoih-prog/Ethernet_Manager)
    * [ 9. Ethernet_Manager_STM32](https://github.com/khoih-prog/Ethernet_Manager_STM32)
    * [10. WiFiManager_Generic_Lite](https://github.com/khoih-prog/WiFiManager_Generic_Lite)
* [Example checkWaitingDRD](#example-checkWaitingDRD)
* [Debug Terminal Output Samples](#debug-terminal-output-samples)
  * [1. SAMD_WiFiNINA on SAMD NANO_33_IOT using FlashStorage_SAMD](#1-samd_wifinina-on-samd-nano_33_iot-using-flashstorage_samd)
    * [1.1 DRD Not Detected](#11-drd-not-detected)
    * [1.2 DRD detected and Config Portal opened](#12-drd-detected-and-config-portal-opened)
  * [2. minimal on RASPBERRY_PI_PICO using LittleFS](#2-minimal-on-raspberry_pi_pico-using-littlefs)
    * [2.1 No DRD](#21-no-drd)
    * [2.2 DRD](#22-drd)
  * [3. minimal on MBED RASPBERRY_PI_PICO using LittleFS](#3-minimal-on-mbed-raspberry_pi_pico-using-littlefs)
    * [3.1 No DRD](#31-no-drd)
    * [3.2 DRD](#32-drd)
  * [4. minimal on Nano RP2040 Connect using LittleFS](#4-minimal-on-nano-rp2040-connect-using-littlefs)
    * [4.1 No DRD](#41-no-drd)
    * [4.2 DRD](#42-drd)
  * [5. minimal on Rtlduino RTL8720DN using FlashStorage](#5-minimal-on-rtlduino-rtl8720dn-using-flashstorage)
    * [5.1 No DRD](#51-no-drd)
    * [5.2 DRD](#52-drd)
  * [6. minimal on Nano 33 BLE using LittleFS](#6-minimal-on-nano-33-ble-using-littlefs)
    * [6.1 No DRD](#61-no-drd)
    * [6.2 DRD](#62-drd) 
  * [7. minimal on PORTENTA_H7_M7 using LittleFS](#7-minimal-on-portenta_h7_m7-using-littlefs)
    * [7.1 No DRD](#71-no-drd)
    * [7.2 DRD](#72-drd) 
* [Libraries using ESP_DoubleResetDetector or DoubleResetDetector_Generic library](#libraries-using-esp_doubleresetdetector-or-doubleresetdetector_generic-library)
* [Debug](#debug)
* [Troubleshooting](#troubleshooting)
* [Issues](#issues)
* [TO DO](#to-do)
* [DONE](#done)
* [Contributions and Thanks](#contributions-and-thanks)
* [Contributing](#contributing)
* [License](#license)
* [Copyright](#copyright)
---
---

### Important Notes for Portenta_H7

The LittleFS of the new **Portenta_H7** board currently tested OK with only **maximum 8 files**. The files, from 9 and up, somehow strangely can't be written and / or read. This is possibly a bug in the [`ArduinoCore-mbed mbed_portenta core`](https://github.com/arduino/ArduinoCore-mbed). The same behaviour is observed from core v2.0.0 up to v2.4.1.

If LittleFS size is reduced to 1024KB, test is OK with only **maximum 4 files**.

Beware and keep cheking for updates.

---
---

### Why do we need this [DoubleResetDetector_Generic library](https://github.com/khoih-prog/DoubleResetDetector_Generic)

#### Features

[**DoubleResetDetector_Generic**](https://github.com/khoih-prog/DoubleResetDetector_Generic) is a library for the **Arduino AVR, Teensy, SAM DUE, SAMD21, SAMD51, STM32F/L/H/G/WB/MP1, nRF52, RASPBERRY_PI_PICO, RTL8720DN, Nano_33_BLE, Portenta_H7, etc.** boards to detects a double reset, within configurable timeout (default 10s) seconds, so that an alternative start-up mode can be used. Example use cases are to allow re-configuration of a device's WiFi / MQTT / Blynk credentials.

The **RP2040-based boards, such as Nano RP2040 Connect, RASPBERRY_PI_PICO, ADAFRUIT_FEATHER_RP2040 and GENERIC_RP2040**, are currently supported using [Earle Philhower's arduino-pico core](https://github.com/earlephilhower/arduino-pico) or [**Arduino mbed OS for RP2040/Nano boards**](https://github.com/arduino/ArduinoCore-mbed).

This library is based on, modified, bug-fixed and improved from [`Stephen Denne's DoubleResetDetector`](https://github.com/datacute/DoubleResetDetector) and [`Khoi Hoang's ESP_DoubleResetDetector`](https://github.com/khoih-prog/ESP_DoubleResetDetector) to add support for AVR, Teensy, SAM DUE, SAMD, STM32, nRF52, RTL8720DN, RP2040-based boards, Nano_33_BLE, Portenta_H7, etc.

Currently, [`DoubleResetDetector`](https://github.com/datacute/DoubleResetDetector) only supports ESP8266 using RTC memory, and [`ESP_DoubleResetDetector`](https://github.com/khoih-prog/ESP_DoubleResetDetector) only ESP8266 and ESP32.
 
This library can be used to detect a double reset within a predetermined time to force the program to enter a special operation such as Config Portal, Clear Default Data, etc., using :

 1. EEPROM for AVR, Teensy boards.
 2. [`FlashStorage_SAMD library`](https://github.com/khoih-prog/FlashStorage_SAMD) for SAMD21 and SAMD51 boards (ZERO, MKR, **NANO_33_IOT**, M0, M0 Pro, AdaFruit Itsy-Bitsy M4, etc.)
 3. [`FlashStorage_STM32F1 library`](https://github.com/khoih-prog/FlashStorage_STM32F1) for STM32F1 and STM32F3 boards
 4. [`FlashStorage_STM32 library`](https://github.com/khoih-prog/FlashStorage_STM32) for STM32F/L/H/G/WB/MP1 boards
 5. [`DueFlashStorage library`](https://github.com/sebnil/DueFlashStorage) for SAM DUE
 6. [`Adafruit's LittleFS/InternalFS`](www.adafruit.com) for nRF52
 7. LittleFS for RP2040-based boards, such as **RASPBERRY_PI_PICO, ADAFRUIT_FEATHER_RP2040 and GENERIC_RP2040** using [Earle Philhower's arduino-pico core](https://github.com/earlephilhower/arduino-pico)
 8. LittleFS for RP2040-based boards, such as **Nano RP2040 Connect, RASPBERRY_PI_PICO, ADAFRUIT_FEATHER_RP2040 and GENERIC_RP2040** using [**Arduino mbed OS for RP2040/Nano boards**](https://github.com/arduino/ArduinoCore-mbed).
 9. [`FlashStorage_RTL8720 library`](https://github.com/khoih-prog/FlashStorage_RTL8720) for Realtek RTL8720DN, RTL8722DM, RTL8722CSM, etc.
10. LittleFS for MBED nRF52840-based boards, such as **Nano_33_BLE, Nano_33_BLE_Sense** using [**Arduino mbed OS for Nano boards**](https://github.com/arduino/ArduinoCore-mbed).
11. LittleFS for Portenta_H7 boards, using [**Arduino mbed_portenta core for Portenta boards**](https://github.com/arduino/ArduinoCore-mbed).

---

#### Currently supported Boards

This [**DoubleResetDetector_Generic** library](https://github.com/khoih-prog/DoubleResetDetector_Generic) currently supports these following boards:

 1. **nRF52 boards**, such as 
  - **AdaFruit Feather nRF52832, nRF52840 Express, BlueFruit Sense, Itsy-Bitsy nRF52840 Express, Metro nRF52840 Express, NINA_B302_ublox, NINA_B112_ublox, etc.**
  - - **Sparkfun Pro nRF52840 Mini**
 
 2. **SAM DUE**
 
 3. **SAMD21**
  - Arduino SAMD21: ZERO, MKRs, NANO_33_IOT, etc.
  - Adafruit SAMD21 (M0): ItsyBitsy M0, Feather M0, Feather M0 Express, Metro M0 Express, Circuit Playground Express, Trinket M0, PIRkey, Hallowing M0, Crickit M0, etc.
  - Seeeduino:  LoRaWAN, Zero, Femto M0, XIAO M0, Wio GPS Board, etc.
  
 4. **SAMD51**
  - Adafruit SAMD51 (M4): Metro M4, Grand Central M4, ItsyBitsy M4, Feather M4 Express, Trellis M4, Metro M4 AirLift Lite, MONSTER M4SK Express, Hallowing M4, etc.
  - Seeeduino: Wio Terminal, Grove UI Wireless
  
 5. **Teensy (4.1, 4.0, 3.6, 3.5, 3,2, 3.1, 3.0, LC)**
 
 6. **AVR Mega1280, 2560, ADK.**
 
 7. **STM32F/L/H/G/WB/MP1 boards**

- Nucleo-144
- Nucleo-64
- Discovery
- Generic STM32F0, STM32F1, STM32F2, STM32F3, STM32F4, STM32F7 (with 64+K Flash): x8 and up
- STM32L0, STM32L1, STM32L4, **STM32L4**
- STM32G0, STM32G4
- STM32H7
- STM32WB
- STM32MP1
- LoRa boards
- 3-D printer boards
- Generic Flight Controllers
- Midatronics boards

 8. RP2040-based boards, such as **RASPBERRY_PI_PICO, ADAFRUIT_FEATHER_RP2040 and GENERIC_RP2040**, using [Earle Philhower's arduino-pico core](https://github.com/earlephilhower/arduino-pico).

 9. RP2040-based boards, such as **Nano RP2040 Connect**, using [**Arduino mbed OS for Nano boards**](https://github.com/arduino/ArduinoCore-mbed).
 
10. RP2040-based boards, such as **RASPBERRY_PI_PICO, ADAFRUIT_FEATHER_RP2040 and GENERIC_RP2040**, using [**Arduino-mbed RP2040** core](https://github.com/arduino/ArduinoCore-mbed)

11. Realtek **RTL8720DN, RTL8722DM, RTL8722CSM, etc.**, using [`Arduino AmebaD core`](https://github.com/ambiot/ambd_arduino)

12. **MBED nRF52840-based boards such as Nano_33_BLE, Nano_33_BLE_Sense**, etc. using [**Arduino-mbed mbed_nano** core](https://github.com/arduino/ArduinoCore-mbed)

13. **Portenta_H7** using [**Arduino-mbed mbed_nano** core](https://github.com/arduino/ArduinoCore-mbed)


---
---

## Prerequisites

 1. [`Arduino IDE 1.8.19+` for Arduino](https://github.com/arduino/Arduino). [![GitHub release](https://img.shields.io/github/release/arduino/Arduino.svg)](https://github.com/arduino/Arduino/releases/latest)
 2. [`Arduino Core for STM32 v2.2.0+`](https://github.com/stm32duino/Arduino_Core_STM32) for STM32 boards. [![GitHub release](https://img.shields.io/github/release/stm32duino/Arduino_Core_STM32.svg)](https://github.com/stm32duino/Arduino_Core_STM32/releases/latest)
 3. [`Teensy core 1.56+`](https://www.pjrc.com/teensy/td_download.html) for Teensy (4.1, 4.0, 3.6, 3.5, 3,2, 3.1, 3.0, LC) boards
 4. [`Arduino SAM DUE core 1.6.12+`](https://github.com/arduino/ArduinoCore-sam) for SAM DUE ARM Cortex-M3 boards
 5. [`Arduino SAMD core 1.8.13+`](https://www.arduino.cc/en/Guide/ArduinoM0) for SAMD ARM Cortex-M0+ boards. [![GitHub release](https://img.shields.io/github/release/arduino/ArduinoCore-samd.svg)](https://github.com/arduino/ArduinoCore-samd/releases/latest)
 6. [`Adafruit SAMD core 1.7.9+`](https://www.adafruit.com/) for SAMD ARM Cortex-M0+ and M4 boards (Nano 33 IoT, etc.). [![GitHub release](https://img.shields.io/github/release/adafruit/ArduinoCore-samd.svg)](https://github.com/adafruit/ArduinoCore-samd/releases/latest)
 7. [`Seeeduino SAMD core 1.8.2+`](https://github.com/Seeed-Studio/ArduinoCore-samd) for SAMD21/SAMD51 boards (XIAO M0, Wio Terminal, etc.). [![Latest release](https://img.shields.io/github/release/Seeed-Studio/ArduinoCore-samd.svg)](https://github.com/Seeed-Studio/ArduinoCore-samd/releases/latest/)
 8. [`Adafruit nRF52 v1.3.0+`](https://www.adafruit.com) for nRF52 boards such as Adafruit NRF52840_FEATHER, NRF52832_FEATHER, NRF52840_FEATHER_SENSE, NRF52840_ITSYBITSY, NRF52840_CIRCUITPLAY, NRF52840_CLUE, NRF52840_METRO, NRF52840_PCA10056, PARTICLE_XENON, **NINA_B302_ublox**, etc. [![GitHub release](https://img.shields.io/github/release/adafruit/Adafruit_nRF52_Arduino.svg)](https://github.com/adafruit/Adafruit_nRF52_Arduino/releases/latest)
 9. [`Earle Philhower's arduino-pico core v1.12.0+`](https://github.com/earlephilhower/arduino-pico) for RP2040-based boards such as **RASPBERRY_PI_PICO, ADAFRUIT_FEATHER_RP2040 and GENERIC_RP2040**, etc. [![GitHub release](https://img.shields.io/github/release/earlephilhower/arduino-pico.svg)](https://github.com/earlephilhower/arduino-pico/releases/latest)
10. [`Arduino mbed_rp2040 core 2.7.2+`](https://github.com/arduino/ArduinoCore-mbed) for Arduino (Use Arduino Board Manager) RP2040-based boards, such as **Arduino Nano RP2040 Connect, RASPBERRY_PI_PICO, etc.**. [![GitHub release](https://img.shields.io/github/release/arduino/ArduinoCore-mbed.svg)](https://github.com/arduino/ArduinoCore-mbed/releases/latest)
11. [`Arduino mbed_nano core 2.7.2+`](https://github.com/arduino/ArduinoCore-mbed) for Arduino (Use Arduino Board Manager) MBED nRF52840-based boards such as **Nano_33_BLE, Nano_33_BLE_Sense**. [![GitHub release](https://img.shields.io/github/release/arduino/ArduinoCore-mbed.svg)](https://github.com/arduino/ArduinoCore-mbed/releases/latest)
12. [`ArduinoCore-mbed mbed_portenta core 2.7.2+`](https://github.com/arduino/ArduinoCore-mbed) for Arduino **Portenta_H7**. [![GitHub release](https://img.shields.io/github/release/arduino/ArduinoCore-mbed.svg)](https://github.com/arduino/ArduinoCore-mbed/releases/latest)
13. [`Arduino AmebaD core 3.1.2+`](https://github.com/ambiot/ambd_arduino) for Realtek RTL8720DN, RTL8722DM and RTL8722CSM, etc. boards. [![GitHub release](https://img.shields.io/github/release/ambiot/ambd_arduino.svg)](https://github.com/ambiot/ambd_arduino/releases/latest)

14. [`FlashStorage_SAMD library v1.3.2+`](https://github.com/khoih-prog/FlashStorage_SAMD) for SAMD21 and SAMD51 boards (ZERO, MKR, NANO_33_IOT, M0, M0 Pro, AdaFruit Itsy-Bitsy M4, etc.). [![GitHub release](https://img.shields.io/github/release/khoih-prog/FlashStorage_SAMD.svg)](https://github.com/khoih-prog/FlashStorage_SAMD/releases/latest)
15. [`FlashStorage_STM32F1 library v1.1.0+`](https://github.com/khoih-prog/FlashStorage_STM32F1) for STM32F1/F3 boards. To install. check [![arduino-library-badge](https://www.ardu-badge.com/badge/FlashStorage_STM32F1.svg?)](https://www.ardu-badge.com/FlashStorage_STM32F1)
16. [`FlashStorage_STM32 library v1.2.0+`](https://github.com/khoih-prog/FlashStorage_STM32) for STM32F/L/H/G/WB/MP1 boards. To install. check [![arduino-library-badge](https://www.ardu-badge.com/badge/FlashStorage_STM32.svg?)](https://www.ardu-badge.com/FlashStorage_STM32)
17. [`DueFlashStorage library v1.0.0+`](https://github.com/sebnil/DueFlashStorage) for SAM DUE. To install, check [![arduino-library-badge](https://www.ardu-badge.com/badge/DueFlashStorage.svg?)](https://www.ardu-badge.com/DueFlashStorage)
18. [`FlashStorage_RTL8720 library v1.1.0+`](https://github.com/khoih-prog/FlashStorage_RTL8720) for Realtek RTL8720DN, RTL8722DM, RTL8722CSM, etc. boards. [![GitHub release](https://img.shields.io/github/release/khoih-prog/FlashStorage_RTL8720.svg)](https://github.com/khoih-prog/FlashStorage_RTL8720/releases/latest)

---
---

## Installation

### Use Arduino Library Manager

The best and easiest way is to use `Arduino Library Manager`. Search for `DoubleResetDetector_Generic`, then select / install the latest version.
You can also use this link [![arduino-library-badge](https://www.ardu-badge.com/badge/DoubleResetDetector_Generic.svg?)](https://www.ardu-badge.com/DoubleResetDetector_Generic) for more detailed instructions.

### Manual Install

Another way to install is to:

1. Navigate to [DoubleResetDetector_Generic](https://github.com/khoih-prog/DoubleResetDetector_Generic) page.
2. Download the latest release `DoubleResetDetector_Generic-master.zip`.
3. Extract the zip file to `DoubleResetDetector_Generic-master` directory 
4. Copy whole `DoubleResetDetector_Generic-master` folder to Arduino libraries' directory such as `~/Arduino/libraries/`.

### VS Code & PlatformIO:

1. Install [VS Code](https://code.visualstudio.com/)
2. Install [PlatformIO](https://platformio.org/platformio-ide)
3. Install [**DoubleResetDetector_Generic** library](https://registry.platformio.org/libraries/khoih-prog/DoubleResetDetector_Generic) by using [Library Manager](https://registry.platformio.org/libraries/khoih-prog/DoubleResetDetector_Generic/installation). Search for **DoubleResetDetector_Generic** in [Platform.io Author's Libraries](https://platformio.org/lib/search?query=author:%22Khoi%20Hoang%22)
4. Use included [platformio.ini](platformio/platformio.ini) file from examples to ensure that all dependent libraries will installed automatically. Please visit documentation for the other options and examples at [Project Configuration File](https://docs.platformio.org/page/projectconf.html)

---
---

### Packages' Patches

#### 1. For Adafruit nRF52840 and nRF52832 boards

**To be able to compile, run and automatically detect and display BOARD_NAME on nRF52840/nRF52832 boards**, you have to copy the whole [nRF52 Packages_Patches](Packages_Patches/adafruit/hardware/nrf52/1.3.0) directory into Adafruit nRF52 directory (~/.arduino15/packages/adafruit/hardware/nrf52/1.3.0). 

Supposing the Adafruit nRF52 version is 1.3.0. These files must be copied into the directory:
- `~/.arduino15/packages/adafruit/hardware/nrf52/1.3.0/platform.txt`
- `~/.arduino15/packages/adafruit/hardware/nrf52/1.3.0/boards.txt`
- `~/.arduino15/packages/adafruit/hardware/nrf52/1.3.0/cores/nRF5/Udp.h`
- `~/.arduino15/packages/adafruit/hardware/nrf52/1.3.0/cores/nRF5/Print.h`
- `~/.arduino15/packages/adafruit/hardware/nrf52/1.3.0/cores/nRF5/Print.cpp`
- `~/.arduino15/packages/adafruit/hardware/nrf52/1.3.0/variants/NINA_B302_ublox/variant.h`
- `~/.arduino15/packages/adafruit/hardware/nrf52/1.3.0/variants/NINA_B302_ublox/variant.cpp`
- `~/.arduino15/packages/adafruit/hardware/nrf52/1.3.0/variants/NINA_B112_ublox/variant.h`
- `~/.arduino15/packages/adafruit/hardware/nrf52/1.3.0/variants/NINA_B112_ublox/variant.cpp`
- **`~/.arduino15/packages/adafruit/hardware/nrf52/1.3.0/cores/nRF5/Udp.h`**

Whenever a new version is installed, remember to copy these files into the new version directory. For example, new version is x.yy.z
These files must be copied into the directory:

- `~/.arduino15/packages/adafruit/hardware/nrf52/x.yy.z/platform.txt`
- `~/.arduino15/packages/adafruit/hardware/nrf52/x.yy.z/boards.txt`
- `~/.arduino15/packages/adafruit/hardware/nrf52/x.yy.z/cores/nRF5/Udp.h`
- `~/.arduino15/packages/adafruit/hardware/nrf52/x.yy.z/cores/nRF5/Print.h`
- `~/.arduino15/packages/adafruit/hardware/nrf52/x.yy.z/cores/nRF5/Print.cpp`
- `~/.arduino15/packages/adafruit/hardware/nrf52/x.yy.z/variants/NINA_B302_ublox/variant.h`
- `~/.arduino15/packages/adafruit/hardware/nrf52/x.yy.z/variants/NINA_B302_ublox/variant.cpp`
- `~/.arduino15/packages/adafruit/hardware/nrf52/x.yy.z/variants/NINA_B112_ublox/variant.h`
- `~/.arduino15/packages/adafruit/hardware/nrf52/x.yy.z/variants/NINA_B112_ublox/variant.cpp`
- **`~/.arduino15/packages/adafruit/hardware/nrf52/x.yy.z/cores/nRF5/Udp.h`**

#### 2. For Teensy boards
 
 **To be able to compile and run on Teensy boards**, you have to copy the files in [**Packages_Patches for Teensy directory**](Packages_Patches/hardware/teensy/avr) into Teensy hardware directory (./arduino-1.8.19/hardware/teensy/avr/boards.txt). 

Supposing the Arduino version is 1.8.19. These files must be copied into the directory:

- `./arduino-1.8.19/hardware/teensy/avr/boards.txt`
- `./arduino-1.8.19/hardware/teensy/avr/cores/teensy/Stream.h`
- `./arduino-1.8.19/hardware/teensy/avr/cores/teensy3/Stream.h`
- `./arduino-1.8.19/hardware/teensy/avr/cores/teensy4/Stream.h`

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz
These files must be copied into the directory:

- `./arduino-x.yy.zz/hardware/teensy/avr/boards.txt`
- `./arduino-x.yy.zz/hardware/teensy/avr/cores/teensy/Stream.h`
- `./arduino-x.yy.zz/hardware/teensy/avr/cores/teensy3/Stream.h`
- `./arduino-x.yy.zz/hardware/teensy/avr/cores/teensy4/Stream.h`

#### 3. For Arduino SAM DUE boards
 
 **To be able to compile and run on SAM DUE boards**, you have to copy the whole [SAM DUE](Packages_Patches/arduino/hardware/sam/1.6.12) directory into Arduino sam directory (~/.arduino15/packages/arduino/hardware/sam/1.6.12). 

Supposing the Arduino SAM core version is 1.6.12. This file must be copied into the directory:

- `~/.arduino15/packages/arduino/hardware/sam/1.6.12/platform.txt`

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz
This file must be copied into the directory:

- `~/.arduino15/packages/arduino/hardware/sam/x.yy.zz/platform.txt`

#### 4. For Arduino SAMD boards
 
 ***To be able to compile, run and automatically detect and display BOARD_NAME on Arduino SAMD (Nano-33-IoT, etc) boards***, you have to copy the whole [Arduino SAMD Packages_Patches](Packages_Patches/arduino/hardware/samd/1.8.13) directory into Arduino SAMD directory (~/.arduino15/packages/arduino/hardware/samd/1.8.13).
 
#### For core version v1.8.10+

Supposing the Arduino SAMD version is 1.8.13. Now only one file must be copied into the directory:

- `~/.arduino15/packages/arduino/hardware/samd/1.8.13/platform.txt`

Whenever a new version is installed, remember to copy this files into the new version directory. For example, new version is x.yy.zz

This file must be copied into the directory:

- `~/.arduino15/packages/arduino/hardware/samd/x.yy.zz/platform.txt`
 
#### For core version v1.8.9-

Supposing the Arduino SAMD version is 1.8.9. These files must be copied into the directory:

- `~/.arduino15/packages/arduino/hardware/samd/1.8.9/platform.txt`
- ***`~/.arduino15/packages/arduino/hardware/samd/1.8.9/cores/arduino/Arduino.h`***

Whenever a new version is installed, remember to copy these files into the new version directory. For example, new version is x.yy.z

These files must be copied into the directory:

- `~/.arduino15/packages/arduino/hardware/samd/x.yy.z/platform.txt`
- ***`~/.arduino15/packages/arduino/hardware/samd/x.yy.z/cores/arduino/Arduino.h`***
 
 This is mandatory to fix the ***notorious Arduino SAMD compiler error***. See [Improve Arduino compatibility with the STL (min and max macro)](https://github.com/arduino/ArduinoCore-samd/pull/399)
 
```
 ...\arm-none-eabi\include\c++\7.2.1\bits\stl_algobase.h:243:56: error: macro "min" passed 3 arguments, but takes just 2
     min(const _Tp& __a, const _Tp& __b, _Compare __comp)
```

Whenever the above-mentioned compiler error issue is fixed with the new Arduino SAMD release, you don't need to copy the `Arduino.h` file anymore.

#### 5. For Adafruit SAMD boards
 
 ***To be able to compile, run and automatically detect and display BOARD_NAME on Adafruit SAMD (Itsy-Bitsy M4, etc) boards***, you have to copy the whole [Adafruit SAMD Packages_Patches](Packages_Patches/adafruit/hardware/samd/1.7.9) directory into Adafruit samd directory (~/.arduino15/packages/adafruit/hardware/samd/1.7.9). 

Supposing the Adafruit SAMD core version is 1.7.9. This file must be copied into the directory:

- `~/.arduino15/packages/adafruit/hardware/samd/1.7.9/platform.txt`
- `~/.arduino15/packages/adafruit/hardware/samd/1.7.9/cores/arduino/Print.h`
- `~/.arduino15/packages/adafruit/hardware/samd/1.7.9/cores/arduino/Print.cpp`

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz
This file must be copied into the directory:

- `~/.arduino15/packages/adafruit/hardware/samd/x.yy.zz/platform.txt`
- `~/.arduino15/packages/adafruit/hardware/samd/x.yy.zz/cores/arduino/Print.h`
- `~/.arduino15/packages/adafruit/hardware/samd/x.yy.zz/cores/arduino/Print.cpp`

#### 6. For Seeeduino SAMD boards
 
 ***To be able to compile, run and automatically detect and display BOARD_NAME on Seeeduino SAMD (XIAO M0, Wio Terminal, etc) boards***, you have to copy the whole [Seeeduino SAMD Packages_Patches](Packages_Patches/Seeeduino/hardware/samd/1.8.2) directory into Seeeduino samd directory (~/.arduino15/packages/Seeeduino/hardware/samd/1.8.2). 

Supposing the Seeeduino SAMD core version is 1.8.2. This file must be copied into the directory:

- `~/.arduino15/packages/Seeeduino/hardware/samd/1.8.2/platform.txt`
- `~/.arduino15/packages/Seeeduino/hardware/samd/1.8.2/cores/arduino/Arduino.h`
- `~/.arduino15/packages/Seeeduino/hardware/samd/1.8.2/cores/arduino/Print.h`
- `~/.arduino15/packages/Seeeduino/hardware/samd/1.8.2/cores/arduino/Print.cpp`

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz
This file must be copied into the directory:

- `~/.arduino15/packages/Seeeduino/hardware/samd/x.yy.zz/platform.txt`
- `~/.arduino15/packages/Seeeduino/hardware/samd/x.yy.zz/cores/arduino/Arduino.h`
- `~/.arduino15/packages/Seeeduino/hardware/samd/x.yy.zz/cores/arduino/Print.h`
- `~/.arduino15/packages/Seeeduino/hardware/samd/x.yy.zz/cores/arduino/Print.cpp`

#### 7. For STM32 boards

#### 7.1 For STM32 boards to use LAN8720

To use LAN8720 on some STM32 boards 

- **Nucleo-144 (F429ZI, NUCLEO_F746NG, NUCLEO_F746ZG, NUCLEO_F756ZG)**
- **Discovery (DISCO_F746NG)**
- **STM32F4 boards (BLACK_F407VE, BLACK_F407VG, BLACK_F407ZE, BLACK_F407ZG, BLACK_F407VE_Mini, DIYMORE_F407VGT, FK407M1)**

you have to copy the files [stm32f4xx_hal_conf_default.h](Packages_Patches/STM32/hardware/stm32/2.2.0/system/STM32F4xx) and [stm32f7xx_hal_conf_default.h](Packages_Patches/STM32/hardware/stm32/2.2.0/system/STM32F7xx) into STM32 stm32 directory (~/.arduino15/packages/STM32/hardware/stm32/2.2.0/system) to overwrite the old files.

Supposing the STM32 stm32 core version is 2.2.0. These files must be copied into the directory:

- `~/.arduino15/packages/STM32/hardware/stm32/2.2.0/system/STM32F4xx/stm32f4xx_hal_conf_default.h` for STM32F4.
- `~/.arduino15/packages/STM32/hardware/stm32/2.2.0/system/STM32F7xx/stm32f7xx_hal_conf_default.h` for Nucleo-144 STM32F7.

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz,
theses files must be copied into the corresponding directory:

- `~/.arduino15/packages/STM32/hardware/stm32/x.yy.zz/system/STM32F4xx/stm32f4xx_hal_conf_default.h`
- `~/.arduino15/packages/STM32/hardware/stm32/x.yy.zz/system/STM32F7xx/stm32f7xx_hal_conf_default.h


#### 7.2 For STM32 boards to use Serial1

**To use Serial1 on some STM32 boards without Serial1 definition (Nucleo-144 NUCLEO_F767ZI, Nucleo-64 NUCLEO_L053R8, etc.) boards**, you have to copy the files [STM32 variant.h](Packages_Patches/STM32/hardware/stm32/2.2.0) into STM32 stm32 directory (~/.arduino15/packages/STM32/hardware/stm32/2.2.0). You have to modify the files corresponding to your boards, this is just an illustration how to do.

Supposing the STM32 stm32 core version is 2.2.0. These files must be copied into the directory:

- `~/.arduino15/packages/STM32/hardware/stm32/2.2.0/variants/STM32F7xx/F765Z(G-I)T_F767Z(G-I)T_F777ZIT/NUCLEO_F767ZI/variant.h` for Nucleo-144 NUCLEO_F767ZI.
- `~/.arduino15/packages/STM32/hardware/stm32/2.2.0/variants/STM32L0xx/L052R(6-8)T_L053R(6-8)T_L063R8T/NUCLEO_L053R8/variant.h` for Nucleo-64 NUCLEO_L053R8.

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz,
theses files must be copied into the corresponding directory:

- `~/.arduino15/packages/STM32/hardware/stm32/x.yy.zz/variants/STM32F7xx/F765Z(G-I)T_F767Z(G-I)T_F777ZIT/NUCLEO_F767ZI/variant.h`
- `~/.arduino15/packages/STM32/hardware/stm32/x.yy.zz/variants/STM32L0xx/L052R(6-8)T_L053R(6-8)T_L063R8T/NUCLEO_L053R8/variant.h`

#### 8. For RP2040-based boards using [Earle Philhower arduino-pico core](https://github.com/earlephilhower/arduino-pico)

#### 8.1 To use BOARD_NAME

 **To be able to automatically detect and display BOARD_NAME on RP2040-based boards (RASPBERRY_PI_PICO, ADAFRUIT_FEATHER_RP2040, GENERIC_RP2040, etc) boards**, you have to copy the file [RP2040 platform.txt](Packages_Patches/rp2040/hardware/rp2040/1.4.0) into rp2040 directory (~/.arduino15/packages/rp2040/hardware/rp2040/1.4.0). 

Supposing the rp2040 core version is 1.4.0. This file must be copied into the directory:

- `~/.arduino15/packages/rp2040/hardware/rp2040/1.4.0/platform.txt`

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz
This file must be copied into the directory:

- `~/.arduino15/packages/rp2040/hardware/rp2040/x.yy.zz/platform.txt`

With core after v1.5.0, this step is not necessary anymore thanks to the PR [Add -DBOARD_NAME="{build.board}" #136](https://github.com/earlephilhower/arduino-pico/pull/136).

#### 8.2 To avoid compile error relating to microsecondsToClockCycles

Some libraries, such as [Adafruit DHT-sensor-library](https://github.com/adafruit/DHT-sensor-library), require the definition of microsecondsToClockCycles(). **To be able to compile and run on RP2040-based boards**, you have to copy the files in [**RP2040 Arduino.h**](Packages_Patches/rp2040/hardware/rp2040/1.4.0/cores/rp2040/Arduino.h) into rp2040 directory (~/.arduino15/packages/rp2040/hardware/rp2040/1.4.0).

Supposing the rp2040 core version is 1.4.0. This file must be copied to replace:

- `~/.arduino15/packages/rp2040/hardware/rp2040/1.4.0/cores/rp2040/Arduino.h`

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz
This file must be copied to replace:

- `~/.arduino15/packages/rp2040/hardware/rp2040/x.yy.zz/cores/rp2040/Arduino.h`

With core after v1.5.0, this step is not necessary anymore thanks to the PR [Add defs for compatibility #142](https://github.com/earlephilhower/arduino-pico/pull/142).


#### 9. For Portenta_H7 boards using Arduino IDE in Linux

  **To be able to upload firmware to Portenta_H7 using Arduino IDE in Linux (Ubuntu, etc.)**, you have to copy the file [portenta_post_install.sh](Packages_Patches/arduino/hardware/mbed_portenta/2.7.2/portenta_post_install.sh) into mbed_portenta directory (~/.arduino15/packages/arduino/hardware/mbed_portenta/2.7.2/portenta_post_install.sh). 
  
  Then run the following command using `sudo`
  
```
$ cd ~/.arduino15/packages/arduino/hardware/mbed_portenta/2.7.2
$ chmod 755 portenta_post_install.sh
$ sudo ./portenta_post_install.sh
```

This will create the file `/etc/udev/rules.d/49-portenta_h7.rules` as follows:

```
# Portenta H7 bootloader mode UDEV rules

SUBSYSTEMS=="usb", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="035b", GROUP="plugdev", MODE="0666"
```

Supposing the ArduinoCore-mbed core version is 2.7.2. Now only one file must be copied into the directory:

- `~/.arduino15/packages/arduino/hardware/mbed_portenta/2.7.2/portenta_post_install.sh`

Whenever a new version is installed, remember to copy this files into the new version directory. For example, new version is x.yy.zz

This file must be copied into the directory:

- `~/.arduino15/packages/arduino/hardware/mbed_portenta/x.yy.zz/portenta_post_install.sh`


---
---

### Purpose

Detects a double reset so that an alternative start-up mode can be used. One example use is to allow re-configuration of a device's WiFi or Blynk Credentials.

### How It Works
  
   When the device starts up it checks the EEPROM, (Due)FlashStorage or LittleFS file `/drd.dat` for a flag to see if it has been recently reset within the configurable timeout seconds
   
---
---
   
### Examples: 

 1. [minimal](examples/minimal)
 2. [checkWaitingDRD](examples/checkWaitingDRD) **New**


#### Many other libraries are depending on this library's DRD feature

  All examples of these following libraries are using DRD feature of this [DoubleResetDetector_Generic Library](https://github.com/khoih-prog/DoubleResetDetector_Generic)
   
* [ 1. BlynkEthernet_WM](https://github.com/khoih-prog/BlynkEthernet_WM)
* [ 2. Blynk_Esp8266AT_WM](https://github.com/khoih-prog/Blynk_Esp8266AT_WM)
* [ 3. Blynk_WiFiNINA_WM](https://github.com/khoih-prog/Blynk_WiFiNINA_WM)
* [ 4. WiFiManager_NINA_Lite](https://github.com/khoih-prog/WiFiManager_NINA_Lite)
* [ 5. BlynkEthernet_STM32_WM](https://github.com/khoih-prog/BlynkEthernet_STM32_WM)
* [ 6. ESP_AT_WM_Lite](https://github.com/khoih-prog/ESP_AT_WM_Lite)
* [ 7. WIOTerminal_WiFiManager](https://github.com/khoih-prog/WIOTerminal_WiFiManager)
* [ 8. Ethernet_Manager](https://github.com/khoih-prog/Ethernet_Manager)
* [ 9. Ethernet_Manager_STM32](https://github.com/khoih-prog/Ethernet_Manager_STM32)
* [10. WiFiManager_Generic_Lite](https://github.com/khoih-prog/WiFiManager_Generic_Lite)
  
  
---
---
   
#### Example [checkWaitingDRD](examples/checkWaitingDRD)

https://github.com/khoih-prog/DoubleResetDetector_Generic/blob/5216729d2ebfb6d350dc39ebf8c4b8f3cab4ac39/examples/checkWaitingDRD/checkWaitingDRD.ino#L57-L164


---
---

### Debug Terminal Output Samples

### 1. SAMD_WiFiNINA on SAMD NANO_33_IOT using FlashStorage_SAMD

This is the terminal output for a Nano 33 IoT using [`WiFiManager_NINA_Lite`](https://github.com/khoih-prog/WiFiManager_NINA_Lite) when DRD is detected and not detected

#### 1.1 DRD Not Detected

```
Start SAMD_WiFiNINA on SAMD NANO_33_IOT
*NN: Hostname=SAMD-Master-Controller
Flag read = 0xd0d04321
No doubleResetDetected             <====== No DRD
SetFlag write = 0xd0d01234
*NN: bg: noConfigPortal = true
*NN: Connecting MultiWifi...
*NN: con2WF:spentMsec=0
WiFi-begin: return1 = 3
WiFi-begin: return2 = 3
*NN: con2WF:OK
*NN: SSID=HueNet1,RSSI=-34
*NN: IP=192.168.2.93
*NN: b:WOK
Stop doubleResetDetecting
ClearFlag write = 0xd0d04321
Your stored Credentials :
Blynk Server1 = new-account.duckdns.org
Token1 = new-token1
Blynk Server2 = new-account.ddns.net
Token2 = new-token2
Port = 8080
MQTT Server = new-mqtt.duckdns.org
```

#### 1.2 DRD detected and Config Portal opened

```
Start SAMD_WiFiNINA on SAMD NANO_33_IOT
*NN: Hostname=SAMD-Master-Controller
Flag read = 0xd0d01234
doubleResetDetected
ClearFlag write = 0xd0d04321
*NN: Double Reset Detected      <====== DRD detected
*NN: bg: noConfigPortal = false
*NN: b:OpenPortal               <====== Config Portal opened
*NN: SSID=WIFININA_51F485,PW=MyWIFININA_51F485
*NN: IP=192.168.4.1,CH=10

```

---

### 2. minimal on RASPBERRY_PI_PICO using LittleFS

This is the terminal output when running [minimal](examples/minimal) example on **RASPBERRY_PI_PICO** using LittleFS

#### 2.1 No DRD

```
DoubleResetDetector minimal Example Program on RASPBERRY_PI_PICO
DoubleResetDetector_Generic v1.8.1
-----------------------------------
LittleFS Flag read = 0xd0d04321
Flag read = 0xd0d04321
No doubleResetDetected
Saving DOUBLERESETDETECTOR_FLAG to DRD file : 0xd0d01234
Saving DRD file OK
SetFlag write = 0xd0d01234
No Double Reset Detected
Stop doubleResetDetecting
Saving to DRD file : 0xd0d04321
Saving DRD file OK
LittleFS Flag read = 0xd0d04321
ClearFlag write = 0xd0d04321
```


### 2.2 DRD


```
DoubleResetDetector minimal Example Program on RASPBERRY_PI_PICO
DoubleResetDetector_Generic v1.8.1
-----------------------------------
LittleFS Flag read = 0xd0d01234
Flag read = 0xd0d01234
doubleResetDetected
Saving to DRD file : 0xd0d04321
Saving DRD file OK
LittleFS Flag read = 0xd0d04321
ClearFlag write = 0xd0d04321
Double Reset Detected
```

---

### 3. minimal on MBED RASPBERRY_PI_PICO using LittleFS

This is the terminal output when running [minimal](examples/minimal) example on **MBED RASPBERRY_PI_PICO** using LittleFS

#### 3.1 No DRD

```
DoubleResetDetector minimal Example Program on RaspberryPi Pico
DoubleResetDetector_Generic v1.8.1
-----------------------------------
LittleFS size (KB) = 64
LittleFS Mount OK
LittleFS Flag read = 0xd0d04321
Flag read = 0xd0d04321
No doubleResetDetected
Saving DOUBLERESETDETECTOR_FLAG to DRD file : 0xd0d01234
Saving DRD file OK
SetFlag write = 0xd0d01234
No Double Reset Detected
Stop doubleResetDetecting
Saving to DRD file : 0xd0d04321
Saving DRD file OK
LittleFS Flag read = 0xd0d04321
ClearFlag write = 0xd0d04321
```


### 3.2 DRD


```
DoubleResetDetector minimal Example Program on RaspberryPi Pico
DoubleResetDetector_Generic v1.8.1
-----------------------------------
LittleFS size (KB) = 64
LittleFS Mount OK
LittleFS Flag read = 0xd0d01234
Flag read = 0xd0d01234
doubleResetDetected
Saving to DRD file : 0xd0d04321
Saving DRD file OK
LittleFS Flag read = 0xd0d04321
ClearFlag write = 0xd0d04321
Double Reset Detected
```

---

### 4. minimal on Nano RP2040 Connect using LittleFS

This is the terminal output when running [minimal](examples/minimal) example on **Nano RP2040 Connect** using LittleFS

#### 4.1 No DRD

```
DoubleResetDetector minimal Example Program on Nano RP2040 Connect
DoubleResetDetector_Generic v1.8.1
-----------------------------------
LittleFS size (KB) = 64
LittleFS Mount OK
LittleFS Flag read = 0xd0d04321
Flag read = 0xd0d04321
No doubleResetDetected
Saving DOUBLERESETDETECTOR_FLAG to DRD file : 0xd0d01234
Saving DRD file OK
SetFlag write = 0xd0d01234
No Double Reset Detected
Stop doubleResetDetecting
Saving to DRD file : 0xd0d04321
Saving DRD file OK
LittleFS Flag read = 0xd0d04321
ClearFlag write = 0xd0d04321
```


### 4.2 DRD


```
DoubleResetDetector minimal Example Program on Nano RP2040 Connect
DoubleResetDetector_Generic v1.8.1
-----------------------------------
LittleFS size (KB) = 64
LittleFS Mount OK
LittleFS Flag read = 0xd0d01234
Flag read = 0xd0d01234
doubleResetDetected
Saving to DRD file : 0xd0d04321
Saving DRD file OK
LittleFS Flag read = 0xd0d04321
ClearFlag write = 0xd0d04321
Double Reset Detected
```


---

### 5. minimal on Rtlduino RTL8720DN using FlashStorage

This is the terminal output when running [minimal](examples/minimal) example on **Rtlduino RTL8720DN** using FlashStorage

#### 5.1 No DRD

```
DoubleResetDetector minimal Example Program on Rtlduino RTL8720DN
DoubleResetDetector_Generic v1.8.1
-----------------------------------
Flag read = 0xbeefdeed
No doubleResetDetected
SetFlag write = 0xd0d01234
No Double Reset Detected
```


### 5.2 DRD


```
DoubleResetDetector minimal Example Program on Rtlduino RTL8720DN
DoubleResetDetector_Generic v1.8.1
-----------------------------------
Flag read = 0xd0d01234
doubleResetDetected
ClearFlag write = 0xd0d04321
Double Reset Detected
```

---


### 6. minimal on Nano 33 BLE using LittleFS

This is the terminal output when running [minimal](examples/minimal) example on **Rtlduino RTL8720DN** using FlashStorage

#### 6.1 No DRD

```
DoubleResetDetector minimal Example Program on Nano 33 BLE
DoubleResetDetector_Generic v1.8.1
-----------------------------------
LittleFS size (KB) = 64
LittleFS Mount OK
LittleFS Flag read = 0xd0d04321
Flag read = 0xd0d04321
No doubleResetDetected
Saving DOUBLERESETDETECTOR_FLAG to DRD file : 0xd0d01234
Saving DRD file OK
SetFlag write = 0xd0d01234
No Double Reset Detected
Stop doubleResetDetecting
Saving to DRD file : 0xd0d04321
Saving DRD file OK
LittleFS Flag read = 0xd0d04321
ClearFlag write = 0xd0d04321
```


### 6.2 DRD


```
DoubleResetDetector minimal Example Program on Nano 33 BLE
DoubleResetDetector_Generic v1.8.1
-----------------------------------
LittleFS size (KB) = 64
LittleFS Mount OK
LittleFS Flag read = 0xd0d01234
Flag read = 0xd0d01234
doubleResetDetected
Saving to DRD file : 0xd0d04321
Saving DRD file OK
LittleFS Flag read = 0xd0d04321
ClearFlag write = 0xd0d04321
Double Reset Detected
```

---

### 7. minimal on PORTENTA_H7_M7 using LittleFS

This is the terminal output when running [minimal](examples/minimal) example on **Rtlduino RTL8720DN** using FlashStorage

#### 7.1 No DRD

```
DoubleResetDetector minimal Example Program on PORTENTA_H7_M7
DoubleResetDetector_Generic v1.8.1
-----------------------------------
Flash Size: (KB) = 2048.00
FlashIAP Start Address: = 0x8080000
LittleFS size (KB) = 1536.00
LittleFS Mount OK
Loading DRD file failed
Flag read = 0x0
No doubleResetDetected
Saving DOUBLERESETDETECTOR_FLAG to DRD file : 0xd0d01234
Saving DRD file OK
SetFlag write = 0xd0d01234
No Double Reset Detected
```


### 7.2 DRD


```
DoubleResetDetector minimal Example Program on PORTENTA_H7_M7
DoubleResetDetector_Generic v1.8.1
-----------------------------------
Flash Size: (KB) = 2048.00
FlashIAP Start Address: = 0x8080000
LittleFS size (KB) = 1536.00
LittleFS Mount OK
LittleFS Flag read = 0xd0d01234
Flag read = 0xd0d01234
doubleResetDetected
Saving to DRD file : 0xd0d04321
Saving DRD file OK
LittleFS Flag read = 0xd0d04321
ClearFlag write = 0xd0d04321
Double Reset Detected
```

---
---

### Libraries using ESP_DoubleResetDetector or DoubleResetDetector_Generic library

You can also see how [`ESP_DoubleResetDetector`](https://github.com/khoih-prog/ESP_DoubleResetDetector) and [`DoubleResetDetector_Generic`](https://github.com/khoih-prog/DoubleResetDetector_Generic) are applied in many other libraries, such as:

 1. [Blynk_WM](https://github.com/khoih-prog/Blynk_WM)
 2. [BlynkEthernet_WM](https://github.com/khoih-prog/BlynkEthernet_WM)
 3. [WiFiManager_NINA_Lite](https://github.com/khoih-prog/WiFiManager_NINA_Lite)
 4. [BlynkESP32_BT_WF](https://github.com/khoih-prog/BlynkESP32_BT_WF), 
 5. [Blynk_GSM_Manager](https://github.com/khoih-prog/Blynk_GSM_Manager),
 6. [Blynk_Esp8266AT_WM](https://github.com/khoih-prog/Blynk_Esp8266AT_WM), 
 7. [Blynk_WiFiNINA_WM](https://github.com/khoih-prog/Blynk_WiFiNINA_WM), 
 8. [Blynk_Async_WM](https://github.com/khoih-prog/Blynk_Async_WM),
 9. [Blynk_Async_ESP32_BT_WF](https://github.com/khoih-prog/Blynk_Async_ESP32_BT_WF), 
10. [Blynk_Async_GSM_Manager](https://github.com/khoih-prog/Blynk_Async_GSM_Manager),
11. [ESP_WiFiManager](https://github.com/khoih-prog/ESP_WiFiManager)
12. [ESPAsync_WiFiManager](https://github.com/khoih-prog/ESPAsync_WiFiManager)
13. [WiFiManager_Generic_Lite](https://github.com/khoih-prog/WiFiManager_Generic_Lite)
14. [BlynkEthernet_STM32_WM](https://github.com/khoih-prog/BlynkEthernet_STM32_WM),
15. [ESP_AT_WM_Lite](https://github.com/khoih-prog/ESP_AT_WM_Lite)
16. [WIOTerminal_WiFiManager](https://github.com/khoih-prog/WIOTerminal_WiFiManager)
17. [Ethernet_Manager](https://github.com/khoih-prog/Ethernet_Manager)
18. [Ethernet_Manager_STM32](https://github.com/khoih-prog/Ethernet_Manager_STM32)
19. [ESP_WiFiManager_Lite](https://github.com/khoih-prog/ESP_WiFiManager_Lite)
20. [WiFiManager_Portenta_H7_Lite](https://github.com/khoih-prog/WiFiManager_Portenta_H7_Lite)

and the list is growing fast.

---
---

### Debug

Debug is disabled by default. To enable debug:

```cpp
// Use this to output debug msgs to Serial
#define DRD_GENERIC_DEBUG       true
```

---

## Troubleshooting

If you get compilation errors, more often than not, you may need to install a newer version of the board's core, applying Packages' Patches or this library latest version.


---
---

### Issues

Submit issues to: [DoubleResetDetector_Generic issues](https://github.com/khoih-prog/DoubleResetDetector_Generic/issues)

---

### TO DO

1. Search for bug and improvement.
2. More examples and more supported boards.


### DONE

 1. Add support to **Arduino SAMD21 (ZERO, MKR, NANO_33_IOT, etc.)** using FlashStorage_SAMD.
 2. Add support to **Adafruit SAMD21 (Itsy-Bitsy M0, Metro M0, Feather M0 Express, etc.)**  using FlashStorage_SAMD.
 3. Add support to **Adafruit SAMD51 (Itsy-Bitsy M4, Metro M4, Grand Central M4, Feather M4 Express, etc.)**  using FlashStorage_SAMD.
 4. Add support to **Adafruit nRF52 ( Feather nRF52832, nRF52840 Express, BlueFruit Sense, Itsy-Bitsy nRF52840 Express, Metro nRF52840 Express, NINA_B302_ublox, NINA_B112_ublox, etc.** using LittleFS.
 5. Add support to SAM DUE using DueFlashStorage.
 6. Add support to Seeeduino SAMD21/SAMD51: LoRaWAN, Zero, Femto M0, XIAO M0, Wio GPS Board, Wio Terminal, Grove UI Wireless  using FlashStorage_SAMD.
 7. Add support to STM32F/L/H/G/WB/MP1 boards using EEPROM.
 8. Add support to Teensy boards using EEPROM.
 9. Add support to AVR Mega, Nano, UNO, etc boards using EEPROM.
10. Add support to STM32F/L/H/G/WB/MP1 boards using FlashStorage_STM32.
11. Add support to STM32F/L/H/G/WB/MP1 boards using STM32 core v2.0.0
12. Add support to RP2040-based boards such as **RASPBERRY_PI_PICO, ADAFRUIT_FEATHER_RP2040 and GENERIC_RP2040**, using [Earle Philhower's arduino-pico core](https://github.com/earlephilhower/arduino-pico) and LittleFS.
13. Add support to RP2040-based boards, such as **Nano RP2040 Connect**, using [**Arduino mbed OS for Nano boards**](https://github.com/arduino/ArduinoCore-mbed) and LittleFS.
14. Add support to RP2040-based boards, such as **RASPBERRY_PI_PICO, ADAFRUIT_FEATHER_RP2040 and GENERIC_RP2040**, using [**Arduino-mbed RP2040** core](https://github.com/arduino/ArduinoCore-mbed) and LittleFS.
15. Add support to Realtek **RTL8720DN, RTL8722DM, RTL8722CSM, etc.**, using [**Arduino AmebaD core**](https://github.com/ambiot/ambd_arduino)
16. Add support to **MBED nRF52840-based boards such as Nano_33_BLE, Nano_33_BLE_Sense**, etc. using [**Arduino-mbed mbed_nano** core](https://github.com/arduino/ArduinoCore-mbed)
17. Add support to **Portenta_H7** using [**Arduino-mbed mbed_nano** core](https://github.com/arduino/ArduinoCore-mbed) and LittleFS.
18. Reduce **Portenta_H7** `LittleFS` size to 1024KB
19. Update to be compatible with new `FlashStorage_SAMD`, `FlashStorage_STM32` and `FlashStorage_STM32F1` libraries.
20. Use more efficient `FlashStorage_STM32` and `FlashStorage_STM32F1` libraries for STM32
21. Add support to more `SAMD` and `STM32` boards
22. Add support to `Sparkfun Pro nRF52840 Mini`
23. Add waitingForDRD() function to signal in DRD wating period

---
---

### Contributions and Thanks

1. Thanks to [Stephen Denne](https://github.com/datacute) for the [`DoubleResetDetector library`](https://github.com/datacute/DoubleResetDetector) this library is based upon.
2. Thanks to [Miguel Alexandre Wisintainer](https://github.com/tcpipchip) for help in debugging and testing. Without that, support to nRF52, especially **NINA_B302_ublox running as nRF52840**, has never been started and finished. See [u-blox nina b](https://github.com/khoih-prog/WiFiNINA_Generic/issues/1)

<table>
  <tr>
    <td align="center"><a href="https://github.com/datacute"><img src="https://github.com/datacute.png" width="100px;" alt="datacute"/><br /><sub><b>⭐️ Stephen Denne</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/tcpipchip"><img src="https://github.com/tcpipchip.png" width="100px;" alt="tcpipchip"/><br /><sub><b>Miguel Wisintainer</b></sub></a><br /></td>
  </tr> 
</table>

---

### Contributing

If you want to contribute to this project:
- Report bugs and errors
- Ask for enhancements
- Create issues and pull requests
- Tell other people about this library

---

### License

- The library is licensed under [MIT](https://github.com/khoih-prog/DoubleResetDetector_Generic/blob/master/LICENSE)

---

### Copyright

Copyright 2020- Khoi Hoang
