# FlashStorage_STM32F1 Library

[![arduino-library-badge](https://www.ardu-badge.com/badge/FlashStorage_STM32F1.svg?)](https://www.ardu-badge.com/FlashStorage_STM32F1)
[![GitHub release](https://img.shields.io/github/release/khoih-prog/FlashStorage_STM32F1.svg)](https://github.com/khoih-prog/FlashStorage_STM32F1/releases)
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/khoih-prog/FlashStorage_STM32F1/blob/main/LICENSE)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#Contributing)
[![GitHub issues](https://img.shields.io/github/issues/khoih-prog/FlashStorage_STM32F1.svg)](http://github.com/khoih-prog/FlashStorage_STM32F1/issues)

---
---

## Table of Contents

* [Important Change from v1.1.0](#Important-Change-from-v110)
* [Why do we need this FlashStorage_STM32F1 library](#why-do-we-need-this-flashstorage_stm32f1-library)
  * [Features](#features)
  * [Currently supported Boards](#currently-supported-boards)
    * [STM32F1xx](#stm32f1xx)
    * [STM32F3xx](#stm32f3xx)
* [Changelog](changelog.md)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
  * [Use Arduino Library Manager](#use-arduino-library-manager)
  * [Manual Install](#manual-install)
  * [VS Code & PlatformIO](#vs-code--platformio)
* [Packages' Patches](#packages-patches)
  * [1. For STM32 boards to use LAN8720](#1-for-stm32-boards-to-use-lan8720)
  * [2. For STM32 boards to use Serial1](#2-for-stm32-boards-to-use-serial1)
* [HOWTO Fix `Multiple Definitions` Linker Error](#howto-fix-multiple-definitions-linker-error)
* [Limited number of writes](#limited-number-of-writes)
* [Usage](#usage)
  * [Using the alternative EEPROM-like API](#using-the-alternative-eeprom-like-api)
* [Examples](#examples)
  * [  1. EEPROM_Clear](examples/EEPROM_Clear)
  * [  2. EEPROM_CRC](examples/EEPROM_CRC)
  * [  3. EEPROM_get](examples/EEPROM_get)
  * [  4. EEPROM_iteration](examples/EEPROM_iteration)
  * [  5. EEPROM_put](examples/EEPROM_put)
  * [  6. EEPROM_read](examples/EEPROM_read)
  * [  7. EEPROM_update](examples/EEPROM_update)
  * [  8. EEPROM_write](examples/EEPROM_write)
  * [  9. **EmulateEEPROM**](examples/EmulateEEPROM)
  * [ 10. **FlashStoreAndRetrieve**](examples/FlashStoreAndRetrieve)
  * [ 11. **multiFileProject**](examples/multiFileProject) **New** 
* [Example FlashStoreAndRetrieve](#example-flashstoreandretrieve)
* [Debug Terminal Output Samples](#debug-terminal-output-samples)
  * [1. EEPROM_get on BLUEPILL_F103C8 with 64KB Flash](#1-eeprom_get-on-bluepill_f103c8-with-64kb-flash)
    * [1.1. First Start](#11-first-start)
    * [1.2. After reset](#12-after-reset)
  * [2. FlashStoreAndRetrieve on BLUEPILL_F103C8 with 64KB Flash](#2-flashstoreandretrieve-on-bluepill_f103c8-with-64kb-flash)
    * [2.1. First Start](#21-first-start)
    * [2.2. After reset](#22-after-reset)
  * [3. EEPROM_write on BLUEPILL_F103C8 with 128KB Flash](#3-eeprom_write-on-bluepill_f103c8-with-128kb-flash)
  * [4. EmulatedEEPROM on BLUEPILL_F103C8 with 128KB Flash](#4-emulatedeeprom-on-bluepill_f103c8-with-128kb-flash)
    * [4.1. First Start](#41-first-start)
    * [4.2. After reset](#42-after-reset)
  * [5. FlashStoreAndRetrieve on GENERIC_F103RCTX with 256KB Flash](#5-flashstoreandretrieve-on-generic_f103rctx-with-256kb-flash)
    * [5.1. First Start](#51-first-start)
    * [5.2. After reset](#52-after-reset)
* [FAQ](#faq)
  * [Can I use a single object to store more stuff?](#can-i-use-a-single-object-to-store-more-stuff)
  * [The content of the FlashStorage is erased each time a new sketch is uploaded?](#the-content-of-the-flashstorage-is-erased-each-time-a-new-sketch-is-uploaded)
  * [Do you recommend to use FLASH instead of EEPROM?](#do-you-recommend-to-use-flash-instead-of-eeprom)
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

### Important Change from v1.1.0

Please have a look at [HOWTO Fix `Multiple Definitions` Linker Error](#howto-fix-multiple-definitions-linker-error)

---
---

### Why do we need this [FlashStorage_STM32F1 library](https://github.com/khoih-prog/FlashStorage_STM32F1)

### Features

The FlashStorage_STM32F1 library, inspired by [Cristian Maglie's FlashStorage](https://github.com/cmaglie/FlashStorage), provides a convenient way to store and retrieve user's data using emulated-EEPROM, from the non-volatile flash memory of STM32F1/F3, including non-genuine CH32F103xx, CS32F103xx, etc. boards.

The flash memory, generally used to store the firmware code, can also be used to store / retrieve more user's data and faster than from EEPROM. Thanks to the **buffered data writing and reading**, the flash access time is greatly reduced to **increase the life of the flash**.

Currently, the library supports both new [**STM32 core v2.0.0**](https://github.com/stm32duino/Arduino_Core_STM32/releases/tag/2.0.0) and previous [**STM32 core v1.9.0**](https://github.com/stm32duino/Arduino_Core_STM32/releases/tag/1.9.0)

---

### Currently supported Boards

1. **STM32F1/F3 boards with / without integrated EEPROM**

#### STM32F1xx

- NUCLEO_F103RB, DISCO_F100RB
- BLUEPILL_F103C6, BLUEPILL_F103C8, BLUEPILL_F103CB
- BLACKPILL_F103C8, BLACKPILL_F103CB
- Generic STM32F1, STM32F3
- VCCGND_F103ZET6_MINI, VCCGND_F103ZET6, 
- HY_TINYSTM103TB, MAPLEMINI_F103CB
- BLUEBUTTON_F103R8T, BLUEBUTTON_F103RBT, BLUEBUTTON_F103RCT, BLUEBUTTON_F103RET
- GENERIC_F100C4TX, GENERIC_F100C8TX, GENERIC_F100CBTX
- GENERIC_F103C4TX, GENERIC_F103C6TX, GENERIC_F103C6UX, GENERIC_F103C8TX, GENERIC_F103CBTX, GENERIC_F103CBUX
- GENERIC_F100R8TX, GENERIC_F100RBTX
- GENERIC_F103R4HX, GENERIC_F103R6HX, GENERIC_F103R4TX, GENERIC_F103R6TX, GENERIC_F103R8HX, GENERIC_F103RBHX, GENERIC_F103R8TX
- GENERIC_F103RBTX, GENERIC_F103RCTX, GENERIC_F103RDTX, GENERIC_F103RETX, GENERIC_F103RCYX, GENERIC_F103RDYX, GENERIC_F103REYX
- GENERIC_F103RFTX, GENERIC_F103RGTX
- GENERIC_F103T4UX, GENERIC_F103T6UX, GENERIC_F103T8UX, GENERIC_F103TBUX
- GENERIC_F103V8HX, GENERIC_F103VBHX, GENERIC_F103V8TX, GENERIC_F103VBTX, GENERIC_F103VBIX, GENERIC_F103VCHX, GENERIC_F103VDHX
- GENERIC_F103VEHX, GENERIC_F103VCTX, GENERIC_F103VDTX, GENERIC_F103VETX, GENERIC_F103VFTX, GENERIC_F103VGTX, 
- GENERIC_F103ZCHX, GENERIC_F103ZDHX, GENERIC_F103ZEHX, GENERIC_F103ZCTX, GENERIC_F103ZDTX, GENERIC_F103ZETX, GENERIC_F103ZFHX,
- GENERIC_F103ZGHX, GENERIC_F103ZFTX, GENERIC_F103ZGTX
- MALYANM200_F103CB, AFROFLIGHT_F103CB

#### STM32F3xx

- NUCLEO_F302R8, NUCLEO_F303RE, NUCLEO_F303K8, DISCO_F303VC
- BLACKPILL_F303CC, OLIMEXINO_STM32F3
- GENERIC_F302R6TX, GENERIC_F302R8TX, GENERIC_F303RBTX, GENERIC_F303RCTX, GENERIC_F303RDTX, GENERIC_F303RETX
- GENERIC_F303CBTX, GENERIC_F303CCTX
- GENERIC_F303K6TX, GENERIC_F303K8TX
- GENERIC_F303VBTX, GENERIC_F303VCTX
- GENERIC_F334K4TX, GENERIC_F334K6TX, GENERIC_F334K8TX
- SPARKY_F303CC

---
---

## Prerequisites

 1. [`Arduino IDE 1.8.19+` for Arduino](https://github.com/arduino/Arduino). [![GitHub release](https://img.shields.io/github/release/arduino/Arduino.svg)](https://github.com/arduino/Arduino/releases/latest)
 2. [`Arduino Core for STM32 v2.2.0+`](https://github.com/stm32duino/Arduino_Core_STM32) for STM32 boards. [![GitHub release](https://img.shields.io/github/release/stm32duino/Arduino_Core_STM32.svg)](https://github.com/stm32duino/Arduino_Core_STM32/releases/latest)

---

## Installation

### Use Arduino Library Manager

The best and easiest way is to use `Arduino Library Manager`. Search for [**FlashStorage_STM32F1**](https://github.com/khoih-prog/FlashStorage_STM32F1), then select / install the latest version.
You can also use this link [![arduino-library-badge](https://www.ardu-badge.com/badge/FlashStorage_STM32F1.svg?)](https://www.ardu-badge.com/FlashStorage_STM32F1) for more detailed instructions.

### Manual Install

Another way to install is to:

1. Navigate to [**FlashStorage_STM32F1**](https://github.com/khoih-prog/FlashStorage_STM32F1) page.
2. Download the latest release `FlashStorage_STM32F1-main.zip`.
3. Extract the zip file to `FlashStorage_STM32F1-main` directory 
4. Copy whole `FlashStorage_STM32F1-main` folder to Arduino libraries' directory such as `~/Arduino/libraries/`.

### VS Code & PlatformIO

1. Install [VS Code](https://code.visualstudio.com/)
2. Install [PlatformIO](https://platformio.org/platformio-ide)
3. Install [**FlashStorage_STM32F1** library](https://platformio.org/lib/show/12688/FlashStorage_STM32F1) by using [Library Manager](https://platformio.org/lib/show/12688/FlashStorage_STM32F1/installation). Search for **FlashStorage_STM32F1** in [Platform.io Author's Libraries](https://platformio.org/lib/search?query=author:%22Khoi%20Hoang%22)
4. Use included [platformio.ini](platformio/platformio.ini) file from examples to ensure that all dependent libraries will installed automatically. Please visit documentation for the other options and examples at [Project Configuration File](https://docs.platformio.org/page/projectconf.html)


---
---

### Packages' Patches

#### 1. For STM32 boards to use LAN8720

Already updated and tested with latest **STM32 core v2.2.0**

To use LAN8720 on some STM32 boards 

- **Nucleo-144 (F429ZI, NUCLEO_F746NG, NUCLEO_F746ZG, NUCLEO_F756ZG)**
- **Discovery (DISCO_F746NG)**
- **STM32F4 boards (BLACK_F407VE, BLACK_F407VG, BLACK_F407ZE, BLACK_F407ZG, BLACK_F407VE_Mini, DIYMORE_F407VGT, FK407M1)**

you have to copy the files [stm32f4xx_hal_conf_default.h](Packages_Patches/STM32/hardware/stm32/2.2.0/system/STM32F4xx) and [stm32f7xx_hal_conf_default.h](Packages_Patches/STM32/hardware/stm32/2.2.0/system/STM32F7xx) into STM32 stm32 directory (~/.arduino15/packages/STM32/hardware/stm32/x.yy.zz/system) to overwrite the old files.

Supposing the STM32 stm32 core version is 2.2.0. These files must be copied into the directory:

- `~/.arduino15/packages/STM32/hardware/stm32/2.2.0/system/STM32F4xx/stm32f4xx_hal_conf_default.h` for STM32F4.
- `~/.arduino15/packages/STM32/hardware/stm32/2.2.0/system/STM32F7xx/stm32f7xx_hal_conf_default.h` for Nucleo-144 STM32F7.

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz,
theses files must be copied into the corresponding directory:

- `~/.arduino15/packages/STM32/hardware/stm32/x.yy.zz/system/STM32F4xx/stm32f4xx_hal_conf_default.h`
- `~/.arduino15/packages/STM32/hardware/stm32/x.yy.zz/system/STM32F7xx/stm32f7xx_hal_conf_default.h


#### 2. For STM32 boards to use Serial1

**To use Serial1 on some STM32 boards without Serial1 definition (Nucleo-144 NUCLEO_F767ZI, Nucleo-64 NUCLEO_L053R8, etc.) boards**, you have to copy the files [STM32 variant.h](Packages_Patches/STM32/hardware/stm32/2.2.0) into STM32 stm32 directory (~/.arduino15/packages/STM32/hardware/stm32/2.2.0). You have to modify the files corresponding to your boards, this is just an illustration how to do.

Supposing the STM32 stm32 core version is 2.2.0. These files must be copied into the directory:

- `~/.arduino15/packages/STM32/hardware/stm32/2.2.0/variants/NUCLEO_F767ZI/variant.h` for Nucleo-144 NUCLEO_F767ZI.
- `~/.arduino15/packages/STM32/hardware/stm32/2.2.0/variants/NUCLEO_L053R8/variant.h` for Nucleo-64 NUCLEO_L053R8.

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz,
theses files must be copied into the corresponding directory:

- `~/.arduino15/packages/STM32/hardware/stm32/x.yy.zz/variants/NUCLEO_F767ZI/variant.h`
- `~/.arduino15/packages/STM32/hardware/stm32/x.yy.zz/variants/NUCLEO_L053R8/variant.h`


---
---

### HOWTO Fix `Multiple Definitions` Linker Error

The current library implementation, using `xyz-Impl.h` instead of standard `xyz.cpp`, possibly creates certain `Multiple Definitions` Linker error in certain use cases.

You can include this `.hpp` file

```
// Can be included as many times as necessary, without `Multiple Definitions` Linker Error
#include "FlashStorage_STM32F1.hpp"     //https://github.com/khoih-prog/FlashStorage_STM32F1
```

in many files. But be sure to use the following `.h` file **in just 1 `.h`, `.cpp` or `.ino` file**, which must **not be included in any other file**, to avoid `Multiple Definitions` Linker Error

```
// To be included only in main(), .ino with setup() to avoid `Multiple Definitions` Linker Error
#include "FlashStorage_STM32F1.h"           //https://github.com/khoih-prog/FlashStorage_STM32F1
```

Check the new [**multiFileProject** example](examples/multiFileProject) for a `HOWTO` demo.

Have a look at the discussion in [Different behaviour using the src_cpp or src_h lib #80](https://github.com/khoih-prog/ESPAsync_WiFiManager/discussions/80)


---
---

## Limited number of writes

The flash memory has a limited amount of write cycles. Typical flash memories can perform about 10000 writes cycles to the same flash block before starting to "wear out" and begin to lose the ability to retain data.

So **BEWARE: IMPROPER USE OF THIS LIBRARY CAN QUICKLY AND PERMANENTLY DESTROY THE FLASH MEMORY OF YOUR MICRO**, in particular you should avoid to call the `put()` or`commit()` functions too often and make sure that in the entire life of the micro the number of calls to `put()` or`commit()` stay well below the above limit of 10000 (it's a good rule-of-thumb to keep that number in mind even if the manufacturer of the micro guarantees a bigger number of cycles).


---
---

## Usage

### Using the alternative EEPROM-like API

Include `FlashStorage_STM32F1.h` to get an EEPROM emulation with the internal flash memory.

See [EmulateEEPROM](examples/EmulateEEPROM) sketch for an example.

The API is very similar to the well known Arduino EEPROM.h API but with 4 additional functions:

* `bool isValid()` returns `true` if data in the emulated-EEPROM is valid (the data written to flash at least once by `EEPROM.commit()` or `EEPROM.put()`). Otherwise emulated-EEPROM data is "undefined" and the function returns `false`.
* `void commit()` store the EEPROM data in flash. Use this with care: Every call writes the complete emulated-EEPROM data to flash. This will reduce the remaining flash-write-cycles. Don't call this method in a loop or [you will kill your flash soon](#limited-number-of-writes).
* `void setCommitASAP(bool value = true)` to set or clear the `_commitASAP` private variable (default is `true` to be safe). If _commitASAP is false, the call to `EEPROM.put()` won't force the `EEPROM.commit()` to extend the flash life. You'll have to remember to call `EEPROM.commit()` manually to save the emulated-EEPROM data into flash or data will be lost.
* `bool getCommitASAP()` to return the current value of `_commitASAP`.

---
---

### Examples

 1. [EEPROM_Clear](examples/EEPROM_Clear)
 2. [EEPROM_CRC](examples/EEPROM_CRC)
 3. [EEPROM_get](examples/EEPROM_get)
 4. [EEPROM_iteration](examples/EEPROM_iteration)
 5. [EEPROM_put](examples/EEPROM_put)
 6. [EEPROM_read](examples/EEPROM_read)
 7. [EEPROM_update](examples/EEPROM_update)
 8. [EEPROM_write](examples/EEPROM_write)
 9. [EmulateEEPROM](examples/EmulateEEPROM)
10. [FlashStoreAndRetrieve](examples/FlashStoreAndRetrieve)
11. [**multiFileProject**](examples/multiFileProject) **New** 

---
---

### Example [FlashStoreAndRetrieve](examples/FlashStoreAndRetrieve)


```cpp
// Use 0-2. Larger for more debugging messages
#define FLASH_DEBUG       2

// You can select another sector. Be careful not larger than (REGISTERED_NUMBER_FLASH_SECTORS - 1) and large enough not to overwrite your program
// Default is (REGISTERED_NUMBER_FLASH_SECTORS - 1) if you don't specify here
#define USING_FLASH_SECTOR_NUMBER           (REGISTERED_NUMBER_FLASH_SECTORS - 2)

// To be included only in main(), .ino with setup() to avoid `Multiple Definitions` Linker Error
#include <FlashStorage_STM32F1.h>

// Note: the area of flash memory reserved for the variable is
// lost every time the sketch is uploaded on the board.

void setup()
{
  Serial.begin(115200);
  while (!Serial);

  delay(200);

  Serial.print(F("\nStart FlashStoreAndRetrieve on ")); Serial.println(BOARD_NAME);
  Serial.println(FLASH_STORAGE_STM32F1_VERSION);

  Serial.print("EEPROM length: ");
  Serial.println(EEPROM.length());

  EEPROM.init();

  uint16_t address = 0;
  int number;

  // Read the content of emulated-EEPROM
  EEPROM.get(address, number);

  // Print the current number on the serial monitor
  Serial.print("Number = 0x"); Serial.println(number, HEX);

  // Save into emulated-EEPROM the number increased by 1 for the next run of the sketch
  EEPROM.put(address, (int) (number + 1));

  if (!EEPROM.getCommitASAP())
  {
    Serial.println("CommitASAP not set. Need commit()");
    EEPROM.commit();
  }

  Serial.println("Done writing to emulated EEPROM. You can reset now");
}

void loop()
{
  // Do nothing...
}
```

---
---

### Debug Terminal Output Samples

### 1. EEPROM_get on BLUEPILL_F103C8 with 64KB Flash

The following is the sample terminal output when running example [EEPROM_get](examples/EEPROM_get) on STM32F1 BLUEPILL_F103C8 with 64KB Flash

#### 1.1 First Start

```
Start EEPROM_get on BLUEPILL_F103C8
FlashStorage_STM32F1 v1.1.0
EEPROM length: 1019
Start Flash Address: 0x800F800
[FLASH] REGISTERED_NUMBER_FLASH_SECTORS (KB) = 64
[FLASH] USING_FLASH_SECTOR_NUMBER =  62
EEPROM doesn't store valid data, writing WRITTEN_SIGNATURE and some example data
Float written to EEPROM: 123.456
Done writing custom object to EEPROM: 
===============
Field1: 3.14159
Field2: 65
Name: Working!
===============
Reset to see how you can retrieve the values by using EEPROM_get!
```

#### 1.2 After reset

```
Start EEPROM_get on BLUEPILL_F103C8
FlashStorage_STM32F1 v1.1.0
EEPROM length: 1019
Start Flash Address: 0x800F800
[FLASH] REGISTERED_NUMBER_FLASH_SECTORS (KB) = 64
[FLASH] USING_FLASH_SECTOR_NUMBER =  62
EEPROM has valid data with WRITTEN_SIGNATURE. Now read some example data
Read float from EEPROM: 123.456
Read custom object from EEPROM: 
===============
Field1: 3.14159
Field2: 65
Name: Working!
===============
```

---

### 2. FlashStoreAndRetrieve on BLUEPILL_F103C8 with 64KB Flash

The following is the sample terminal output when running example [FlashStoreAndRetrieve](examples/FlashStoreAndRetrieve) on STM32F1 BLUEPILL_F103C8 with 64KB Flash

#### 2.1 First Start

```
Start FlashStoreAndRetrieve on BLUEPILL_F103C8
FlashStorage_STM32F1 v1.1.0
EEPROM length: 1019
Start Flash Address: 0x800FC00
[FLASH] REGISTERED_NUMBER_FLASH_SECTORS (KB) = 64
[FLASH] USING_FLASH_SECTOR_NUMBER =  63
Number = 0x0
Done writing to emulated EEPROM. You can reset now
```

#### 2.2 After reset

```

Start FlashStoreAndRetrieve on BLUEPILL_F103C8
FlashStorage_STM32F1 v1.1.0
EEPROM length: 1019
Start Flash Address: 0x800FC00
[FLASH] REGISTERED_NUMBER_FLASH_SECTORS (KB) = 64
[FLASH] USING_FLASH_SECTOR_NUMBER =  63
Number = 0x1
Done writing to emulated EEPROM. You can reset now
```

---

### 3. EEPROM_write on BLUEPILL_F103C8 with 128KB Flash

The following is the sample terminal output when running example [EEPROM_write](examples/EEPROM_write) on STM32F1 BLUEPILL_F103C8 with 128KB Flash


```
Start EEPROM_write on BLUEPILL_F103C8
FlashStorage_STM32F1 v1.1.0
EEPROM length: 1019
Start Flash Address: 0x801F800
[FLASH] REGISTERED_NUMBER_FLASH_SECTORS (KB) = 128
[FLASH] USING_FLASH_SECTOR_NUMBER =  126
Done writing emulated EEPROM. Time spent (ms) = 29
Done writing emulated EEPROM. Time spent (ms) = 0
Done writing emulated EEPROM. Time spent (ms) = 0
Done writing emulated EEPROM. Time spent (ms) = 0
Done writing emulated EEPROM. Time spent (ms) = 0
Done writing emulated EEPROM. Time spent (ms) = 0
```

---

### 4. EmulatedEEPROM on BLUEPILL_F103C8 with 128KB Flash

The following is the sample terminal output when running example [EmulatedEEPROM](examples/EmulatedEEPROM) on STM32F1 BLUEPILL_F103C8 with 128KB Flash

#### 4.1 First Start

```
Start EmulatedEEPROM on BLUEPILL_F103C8
FlashStorage_STM32F1 v1.1.0
EEPROM length: 1019
Start Flash Address: 0x801FC00
[FLASH] REGISTERED_NUMBER_FLASH_SECTORS (KB) = 128
[FLASH] USING_FLASH_SECTOR_NUMBER =  127
EEPROM is empty, writing WRITTEN_SIGNATURE and some example data:
-> 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119
Done writing to emulated EEPROM. You can reset now to test
```

#### 4.2 After reset

```
Start EmulatedEEPROM on BLUEPILL_F103C8
FlashStorage_STM32F1 v1.1.0
EEPROM length: 1019
Start Flash Address: 0x801FC00
[FLASH] REGISTERED_NUMBER_FLASH_SECTORS (KB) = 128
[FLASH] USING_FLASH_SECTOR_NUMBER =  127
EEPROM has been written.Signature = 0xBEEFDEED
Here is the content of the next 16 bytes:
-> 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 11913
Clearing WRITTEN_SIGNATURE for next try
Done clearing signature in emulated EEPROM. You can reset now
```

---

### 5. FlashStoreAndRetrieve on GENERIC_F103RCTX with 256KB Flash

The following is the sample terminal output when running example [FlashStoreAndRetrieve](examples/FlashStoreAndRetrieve) on STM32F1 GENERIC_F103RCTX with 2564KB Flash


<p align="center">
    <img src="https://github.com/khoih-prog/FlashStorage_STM32F1/blob/main/pics/STM32F103RCT6.jpg">
</p>


|STLink|<--->|GENERIC_F103RCTX|
|:-:|:-:|:-:|
|SWCLK|<--->|SWCLK / PA14|
|SWDIO|<--->|SWDIO / PA13|
|GND|<--->|GND|
|3.3v|<--->|3.3V|


#### 5.1 First Start

```
Start FlashStoreAndRetrieve on GENERIC_F103RCTX
FlashStorage_STM32F1 v1.1.0
EEPROM length: 1019
Start Flash Address: 0x803F800
[FLASH] REGISTERED_NUMBER_FLASH_SECTORS (KB) = 256
[FLASH] USING_FLASH_SECTOR_NUMBER =  254
Number = 0xFFFFFFFF
Done writing to emulated EEPROM. You can reset now
```

#### 5.2 After reset

```
Start FlashStoreAndRetrieve on GENERIC_F103RCTX
FlashStorage_STM32F1 v1.1.0
EEPROM length: 1019
Start Flash Address: 0x803F800
[FLASH] REGISTERED_NUMBER_FLASH_SECTORS (KB) = 256
[FLASH] USING_FLASH_SECTOR_NUMBER =  254
Number = 0x0
Done writing to emulated EEPROM. You can reset now
```

---
---

## FAQ

### Can I use a single object to store more stuff?

Yes, you can declare a `struct` with more fields and call a `EEPROM.put()` to store the entire structure. See the [StoreNameAndSurname](examples/StoreNameAndSurname) for how to do it.

### The content of the FlashStorage is erased each time a new sketch is uploaded?

Not with STM32F1/F3.

### Do you recommend to use FLASH instead of EEPROM?

No. If your board provides an integrated-EEPROM, it's advisable to use that because EEPROM has longer lifetime, number of write cycles, etc.).

In the absence of an integrated-EEPROM or its size is too small for your use-case, you can use this library to use a small portion flash memory as emulated-EEPROM, provided that you keep in mind the limits as in [Limited number of writes](#limited-number-of-writes)

---
---

### Troubleshooting

If you get compilation errors, more often than not, you may need to install a newer version of the core for Arduino boards.

Sometimes, the library will only work if you update the board core to the latest version because I am using newly added functions.

---
---


### Issues

Submit issues to: [FlashStorage_STM32F1 issues](https://github.com/khoih-prog/FlashStorage_STM32F1/issues)

---

### TO DO

1. Search for bug and improvement.
2. Similar features for remaining Arduino STM32 boards

---

### DONE

1. Basic emulated-EEPROM for STM32F1/F3.
2. Add support to new [**STM32 core v1.9.0**](https://github.com/stm32duino/Arduino_Core_STM32/releases/tag/1.9.0)
3. Similar features for remaining Arduino boards such as SAMD21, SAMD51, etc.
4. Add Table of Contents
5. Add support to new [**STM32 core v2.2.0+**](https://github.com/stm32duino/Arduino_Core_STM32/releases/tag/2.2.0)
6. Add `EEPROM.put()` and `EEPROM.get()` functions to read/write the whole struct in emulated-EEPROM
7. Fix `multiple-definitions` linker error. 
8. Clean-up by reducing the number of library files

---
---

### Contributions and Thanks

Many thanks for everyone for bug reporting, new feature suggesting, testing and contributing to the development of this library.

1. Inspired by [Cristian Maglie's FlashStorage](https://github.com/cmaglie/FlashStorage).

<table>
  <tr>
    <td align="center"><a href="https://github.com/cmaglie"><img src="https://github.com/cmaglie.png" width="100px;" alt="cmaglie"/><br /><sub><b>⭐️ Cristian Maglie</b></sub></a><br /></td>
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

- The library is licensed under [LGPLv3](https://github.com/khoih-prog/FlashStorage_STM32F1/blob/main/LICENSE)

---

### Copyright

Copyright 2021- Khoi Hoang

