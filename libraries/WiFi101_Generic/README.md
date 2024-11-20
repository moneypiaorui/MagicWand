= {repository-name} library for the Arduino WiFi Shield 101 and MKR1000 board =

[![GitHub release](https://img.shields.io/github/release/khoih-prog/WiFi101_Generic.svg)](https://github.com/khoih-prog/WiFi101_Generic/releases)
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/khoih-prog/WiFi101_Generic/blob/master/LICENSE)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#Contributing)
[![GitHub issues](https://img.shields.io/github/issues/khoih-prog/WiFi101_Generic.svg)](http://github.com/khoih-prog/WiFi101_Generic/issues)


<a href="https://profile-counter.glitch.me/khoih-prog/count.svg" title="Total khoih-prog Visitor count"><img src="https://profile-counter.glitch.me/khoih-prog/count.svg" style="height: 30px;width: 200px;"></a>
<a href="https://profile-counter.glitch.me/khoih-prog-WiFi101_Generic/count.svg" title="WiFi101_Generic Visitor count"><img src="https://profile-counter.glitch.me/khoih-prog-WiFi101_Generic/count.svg" style="height: 30px;width: 200px;"></a>

---
---

## Table of Contents

* [Why do we need this WiFi101_Generic library](#why-do-we-need-this-WiFi101_Generic-library)
  * [Features](#features)
  * [Currently supported Boards](#currently-supported-boards)
* [Changelog](changelog.md) 
* [Prerequisites](#prerequisites)
* [Installation](#installation)
  * [Manual Install](#manual-install)
  * [VS Code & PlatformIO](#vs-code--platformio)
* [Packages' Patches](#packages-patches)
  * [1. For Arduino SAMD boards](#1-for-arduino-samd-boards)
      * [For core version v1.8.10+](#for-core-version-v1810)
      * [For core version v1.8.9-](#for-core-version-v189-)
  * [2. For Adafruit SAMD boards](#2-for-adafruit-samd-boards)
  * [3. For Seeeduino SAMD boards](#3-for-seeeduino-samd-boards)
* [Examples](#examples)
  * [ 1. AP_SimpleWebServer](examples/AP_SimpleWebServer)
  * [ 2. CheckWifi101FirmwareVersion](examples/CheckWifi101FirmwareVersion)
  * [ 3. ConnectNoEncryption](examples/ConnectNoEncryption)
  * [ 4. ConnectWithWEP](examples/ConnectWithWEP)
  * [ 5. ConnectWithWPA](examples/ConnectWithWPA)
  * [ 6. FirmwareUpdater](examples/FirmwareUpdater)
  * [ 7. MDNS_WiFiWebServer](examples/MDNS_WiFiWebServer)
  * [ 8. Provisioning_WiFiWebServer](examples/Provisioning_WiFiWebServer)
  * [ 9. ScanNetworks](examples/ScanNetworks)
  * [ 10. ScanNetworksAdvanced](examples/ScanNetworksAdvanced)
  * [ 11. SimpleWebServerWiFi](examples/SimpleWebServerWiFi)
  * [ 12. WiFiChatServer](examples/WiFiChatServer)
  * [ 13. WiFiPing](examples/WiFiPing)
  * [ 14. WiFiSSLClient](examples/WiFiSSLClient)
  * [ 15. WiFiUdpNtpClient](examples/WiFiUdpNtpClient)
  * [ 16. WiFiUdpSendReceiveString](examples/WiFiUdpSendReceiveString)
  * [ 17. WiFiWebClient](examples/WiFiWebClient)
  * [ 18. WiFiWebClientRepeating](examples/WiFiWebClientRepeating)
  * [ 19. WiFiWebServer](examples/WiFiWebServer)
* [Debug](#debug)
* [Troubleshooting](#troubleshooting)
* [Issues](#issues)
* [TO DO](#to-do)
* [DONE](#done)
* [Contributions and Thanks](#contributions-and-thanks)
* [Contributing](#contributing)
* [Copyright](#copyright)

---
---

### Why do we need this [WiFi101_Generic library](https://github.com/khoih-prog/WiFi101_Generic)

#### Features

This [**WiFi101_Generic library**](https://github.com/khoih-prog/WiFi101_Generic) is a `port and modification` to add, as many as possible, the features to the following library

1. [**WiFi101**](https://github.com/arduino-libraries/WiFi101)

for **Arduino SAMD21, etc.** boards using `ATMEL WINC1500 WiFi` module. 

---

#### Currently supported Boards

This [**WiFi101_Generic** library](https://github.com/khoih-prog/WiFi101_Generic) currently supports these following boards:

 1. **SAMD21**
  - Arduino SAMD21: MKRs, etc.
 
 2. **SAMD51**
  - Adafruit SAMD51 (M4): Metro M4, Grand Central M4, ItsyBitsy M4, Feather M4 Express, Trellis M4, Metro M4 AirLift Lite, MONSTER M4SK Express, Hallowing M4, etc.


---
---


## Prerequisites

 1. [`Arduino IDE 1.8.19+` for Arduino](https://github.com/arduino/Arduino). [![GitHub release](https://img.shields.io/github/release/arduino/Arduino.svg)](https://github.com/arduino/Arduino/releases/latest)
 2. [`Arduino SAMD core 1.8.13+`](https://github.com/arduino/ArduinoCore-samd) for SAMD ARM Cortex-M0+ boards. [![GitHub release](https://img.shields.io/github/release/arduino/ArduinoCore-samd.svg)](https://github.com/arduino/ArduinoCore-samd/releases/latest)
 3. [`Adafruit SAMD core 1.7.11+`](https://github.com/adafruit/ArduinoCore-samd) for SAMD ARM Cortex-M0+ and M4 boards (Nano 33 IoT, etc.). [![GitHub release](https://img.shields.io/github/release/adafruit/ArduinoCore-samd.svg)](https://github.com/adafruit/ArduinoCore-samd/releases/latest)
 4. [`Seeeduino SAMD core 1.8.3+`](https://github.com/Seeed-Studio/ArduinoCore-samd) for SAMD21/SAMD51 boards (XIAO M0, Wio Terminal, etc.). [![Latest release](https://img.shields.io/github/release/Seeed-Studio/ArduinoCore-samd.svg)](https://github.com/Seeed-Studio/ArduinoCore-samd/releases/latest/)


---

## Installation

### Manual Install

1. Navigate to [WiFi101_Generic](https://github.com/khoih-prog/WiFi101_Generic) page.
2. Download the latest release `WiFi101_Generic-main.zip`.
3. Extract the zip file to `WiFi101_Generic-main` directory 
4. Copy the whole `WiFi101_Generic-main` folder to Arduino libraries' directory such as `~/Arduino/libraries/`.

### VS Code & PlatformIO:

1. Install [VS Code](https://code.visualstudio.com/)
2. Install [PlatformIO](https://platformio.org/platformio-ide)
3. Install [**WiFi101_Generic** library](https://registry.platformio.org/libraries/khoih-prog/WiFi101_Generic) by using [Library Manager](https://registry.platformio.org/libraries/khoih-prog/WiFi101_Generic/installation). Search for WiFi101_Generic in [Platform.io Author's Libraries](https://platformio.org/lib/search?query=author:%22Khoi%20Hoang%22)
4. Use included [platformio.ini](platformio/platformio.ini) file from examples to ensure that all dependent libraries will installed automatically. Please visit documentation for the other options and examples at [Project Configuration File](https://docs.platformio.org/page/projectconf.html)

---
---

### Packages' Patches

#### 1. For Arduino SAMD boards
 
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

#### 2. For Adafruit SAMD boards
 
 ***To be able to compile, run and automatically detect and display BOARD_NAME on Adafruit SAMD (Itsy-Bitsy M4, etc) boards***, you have to copy the whole [Adafruit SAMD Packages_Patches](Packages_Patches/adafruit/hardware/samd/1.7.11) directory into Adafruit samd directory (~/.arduino15/packages/adafruit/hardware/samd/1.7.11). 

Supposing the Adafruit SAMD core version is 1.7.11. These files must be copied into the directory:

- `~/.arduino15/packages/adafruit/hardware/samd/1.7.11/platform.txt`
- `~/.arduino15/packages/adafruit/hardware/samd/1.7.11/cores/arduino/Print.h`
- `~/.arduino15/packages/adafruit/hardware/samd/1.7.11/cores/arduino/Print.cpp`

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz
These files must be copied into the directory:

- `~/.arduino15/packages/adafruit/hardware/samd/x.yy.zz/platform.txt`
- `~/.arduino15/packages/adafruit/hardware/samd/x.yy.zz/cores/arduino/Print.h`
- `~/.arduino15/packages/adafruit/hardware/samd/x.yy.zz/cores/arduino/Print.cpp`

#### 3. For Seeeduino SAMD boards
 
 ***To be able to compile, run and automatically detect and display BOARD_NAME on Seeeduino SAMD (XIAO M0, Wio Terminal, etc) boards***, you have to copy the whole [Seeeduino SAMD Packages_Patches](Packages_Patches/Seeeduino/hardware/samd/1.8.3) directory into Seeeduino samd directory (~/.arduino15/packages/Seeeduino/hardware/samd/1.8.3). 

Supposing the Seeeduino SAMD core version is 1.8.3. These files must be copied into the directory:

- `~/.arduino15/packages/Seeeduino/hardware/samd/1.8.3/platform.txt`
- `~/.arduino15/packages/Seeeduino/hardware/samd/1.8.3/cores/arduino/Arduino.h`
- `~/.arduino15/packages/Seeeduino/hardware/samd/1.8.3/cores/arduino/Print.h`
- `~/.arduino15/packages/Seeeduino/hardware/samd/1.8.3/cores/arduino/Print.cpp`

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz
These files must be copied into the directory:

- `~/.arduino15/packages/Seeeduino/hardware/samd/x.yy.zz/platform.txt`
- `~/.arduino15/packages/Seeeduino/hardware/samd/x.yy.zz/cores/arduino/Arduino.h`
- `~/.arduino15/packages/Seeeduino/hardware/samd/x.yy.zz/cores/arduino/Print.h`
- `~/.arduino15/packages/Seeeduino/hardware/samd/x.yy.zz/cores/arduino/Print.cpp`


---
---

### Examples:

   1. [AP_SimpleWebServer](examples/AP_SimpleWebServer)
   2. [CheckWifi101FirmwareVersion](examples/CheckWifi101FirmwareVersion)
   3. [ConnectNoEncryption](examples/ConnectNoEncryption)
   4. [ConnectWithWEP](examples/ConnectWithWEP)
   5. [ConnectWithWPA](examples/ConnectWithWPA)
   6. [FirmwareUpdater](examples/FirmwareUpdater)
   7. [MDNS_WiFiWebServer](examples/MDNS_WiFiWebServer)
   8. [Provisioning_WiFiWebServer](examples/Provisioning_WiFiWebServer)
   9. [ScanNetworks](examples/ScanNetworks)
  10. [ScanNetworksAdvanced](examples/ScanNetworksAdvanced)
  11. [SimpleWebServerWiFi](examples/SimpleWebServerWiFi)
  12. [WiFiChatServer](examples/WiFiChatServer)
  13. [WiFiPing](examples/WiFiPing)
  14. [WiFiSSLClient](examples/WiFiSSLClient)
  15. [WiFiUdpNtpClient](examples/WiFiUdpNtpClient)
  16. [WiFiUdpSendReceiveString](examples/WiFiUdpSendReceiveString)
  17. [WiFiWebClient](examples/WiFiWebClient)
  18. [WiFiWebClientRepeating](examples/WiFiWebClientRepeating)
  19. [WiFiWebServer](examples/WiFiWebServer)
 
---
---


### Debug

Debug is enabled by default on Serial. Debug Level from 0 to 4. To disable, change the _WIFININA_LOGLEVEL_ to 0

```cpp
// Use this to output debug msgs to Serial
#define DEBUG_WIFININA_PORT         Serial
// Use 0 to disable all output debug msgs
// Debug Level from 0 to 4
#define _WIFININA_LOGLEVEL_              2
```

---

## Troubleshooting

If you get compilation errors, more often than not, you may need to install a newer version of the board's core, applying Libraries' Patches, Packages' Patches or this library latest version.

---
---

### Issues ###

Submit issues to: [WiFi101_Generic issues](https://github.com/khoih-prog/WiFi101_Generic/issues)

---

### TO DO

1. Bug Searching and Killing
2. Support more types boards using ATMEL WINC1500 WiFi module


### DONE

 1. Add compatibility to [WebSockets2_Generic](https://github.com/khoih-prog/WebSockets2_Generic) library
 2. Fix severe limitation to permit sending much larger data than total 4K
 3. Use `allman astyle` and add `utils`

---
---

### Contributions and Thanks

Many thanks for everyone for bug reporting, new feature suggesting, testing and contributing to the development of this library.

1. Based on and modified from [**WiFi101**](https://github.com/arduino-libraries/WiFi101)

---

### Contributing

If you want to contribute to this project:
- Report bugs and errors
- Ask for enhancements
- Create issues and pull requests
- Tell other people about this library

---

## Copyright

Copyright (c) Arduino LLC. All right reserved.

Copyright 2022- Khoi Hoang



