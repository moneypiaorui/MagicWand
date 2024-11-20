# FlashStorage_RTL8720 Library

[![arduino-library-badge](https://www.ardu-badge.com/badge/FlashStorage_RTL8720.svg?)](https://www.ardu-badge.com/FlashStorage_RTL8720)
[![GitHub release](https://img.shields.io/github/release/khoih-prog/FlashStorage_RTL8720.svg)](https://github.com/khoih-prog/FlashStorage_RTL8720/releases)
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/khoih-prog/FlashStorage_RTL8720/blob/main/LICENSE)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#Contributing)
[![GitHub issues](https://img.shields.io/github/issues/khoih-prog/FlashStorage_RTL8720.svg)](http://github.com/khoih-prog/FlashStorage_RTL8720/issues)

---
---

## Table of Contents

* [Why do we need this FlashStorage_RTL8720 library](#why-do-we-need-this-flashstorage_rtl8720-library)
  * [Features](#features)
  * [Currently supported Boards](#currently-supported-boards)
* [Changelog](changelog.md)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
  * [Use Arduino Library Manager](#use-arduino-library-manager)
  * [Manual Install](#manual-install)
  * [VS Code & PlatformIO](#vs-code--platformio)
* [Limited number of writes](#limited-number-of-writes)
* [Usage](#usage)
  * [Using the alternative FlashStorage API](#using-the-alternative-flashstorage-api)
* [Examples](#examples)
  * [  1. FlashStorage_Clear](examples/FlashStorage_Clear)
  * [  2. FlashStorage_CRC](examples/FlashStorage_CRC)
  * [  3. FlashStorage_get](examples/FlashStorage_get)
  * [  4. FlashStorage_iteration](examples/FlashStorage_iteration)
  * [  5. FlashStorage_put](examples/FlashStorage_put)
  * [  6. FlashStorage_read](examples/FlashStorage_read)
  * [  7. FlashStorage_update](examples/FlashStorage_update)
  * [  8. FlashStorage_write](examples/FlashStorage_write)
  * [  9. **FlashStoreAndRetrieve**](examples/FlashStoreAndRetrieve)
  * [ 10. **StoreNameAndSurname**](examples/StoreNameAndSurname)
  * [ 11. **multiFileProject**](examples/multiFileProject) **New** 
* [Example StoreNameAndSurname](#example-storenameandsurname)
* [Debug Terminal Output Samples](#debug-terminal-output-samples)
  * [1. StoreNameAndSurname on Rtlduino RTL8720DN](#1-storenameandsurname-on-rtlduino-rtl8720dn)
    * [1.1. First Start](#11-first-start)
    * [1.2. After Restart](#12-after-restart)
  * [2. FlashStorage_read on Rtlduino RTL8720DN](#2-flashstorage_read-on-rtlduino-rtl8720dn)
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

### Why do we need this [FlashStorage_RTL8720 library](https://github.com/khoih-prog/FlashStorage_RTL8720)

### Features

The FlashStorage_RTL8720 library, inspired by [Cristian Maglie's FlashStorage](https://github.com/cmaglie/FlashStorage), provides a convenient way to store and retrieve user's data using FlashStorage, from the non-volatile flash memory of RTL8720.

The flash memory, generally used to store the firmware code, can also be used to store / retrieve more user's data and faster than from EEPROM. Thanks to the **buffered data writing and reading**, the flash access time is greatly reduced to **increase the life of the flash**.

---

### Currently supported Boards

1. **RTL8720DN, RTL8722DM, RTM8722CSM, etc. boards**


---
---

## Prerequisites

 1. [`Arduino IDE 1.8.19+` for Arduino](https://github.com/arduino/Arduino). [![GitHub release](https://img.shields.io/github/release/arduino/Arduino.svg)](https://github.com/arduino/Arduino/releases/latest)
 2. [`Arduino AmebaD core 3.1.1+`](https://github.com/ambiot/ambd_arduino) for Realtek RTL8720DN, RTL8722DM and RTM8722CSM. [![GitHub release](https://img.shields.io/github/release/ambiot/ambd_arduino.svg)](https://github.com/ambiot/ambd_arduino/releases/latest)

---

## Installation

### Use Arduino Library Manager

The best and easiest way is to use `Arduino Library Manager`. Search for [**FlashStorage_RTL8720**](https://github.com/khoih-prog/FlashStorage_RTL8720), then select / install the latest version.
You can also use this link [![arduino-library-badge](https://www.ardu-badge.com/badge/FlashStorage_RTL8720.svg?)](https://www.ardu-badge.com/FlashStorage_RTL8720) for more detailed instructions.

### Manual Install

Another way to install is to:

1. Navigate to [**FlashStorage_RTL8720**](https://github.com/khoih-prog/FlashStorage_RTL8720) page.
2. Download the latest release `FlashStorage_RTL8720-main.zip`.
3. Extract the zip file to `FlashStorage_RTL8720-main` directory 
4. Copy whole `FlashStorage_RTL8720-main` folder to Arduino libraries' directory such as `~/Arduino/libraries/`.

### VS Code & PlatformIO

1. Install [VS Code](https://code.visualstudio.com/)
2. Install [PlatformIO](https://platformio.org/platformio-ide)
3. Install [**FlashStorage_RTL8720** library](https://platformio.org/lib/show/12658/FlashStorage_RTL8720) by using [Library Manager](https://platformio.org/lib/show/12658/FlashStorage_RTL8720/installation). Search for **FlashStorage_RTL8720** in [Platform.io Author's Libraries](https://platformio.org/lib/search?query=author:%22Khoi%20Hoang%22)
4. Please visit documentation for the other options and examples at [Project Configuration File](https://docs.platformio.org/page/projectconf.html)

---
---

### Limited number of writes

The flash memory has a limited amount of write cycles. Typical flash memories can perform about 10000 writes cycles to the same flash block before starting to "wear out" and begin to lose the ability to retain data.

So **BEWARE: IMPROPER USE OF THIS LIBRARY CAN QUICKLY AND PERMANENTLY DESTROY THE FLASH MEMORY OF YOUR MICRO**, in particular you should avoid to call the `FlashStorage.writeByte(), FlashStorage.writeWord()` functions, without calling `FlashStorage.setCommitASAP(false)`, too often. Also make sure that in the entire life of the number of calls to `write` or `commit()` stay well below the above limit of 10000 (it's a good rule-of-thumb to keep that number in mind even if the manufacturer of the micro guarantees a bigger number of cycles).

The same caution must be taken if you're using the FlashStorage API (see below) with the `FlashStorage.commit()` function.

---
---


## Usage

### Using the alternative FlashStorage API

Include `FlashStorage_RTL8720.h` to get a FlashStorage with the internal flash memory.

See [FlashStoreAndRetrieve](examples/FlashStoreAndRetrieve) sketch for an example.

The API is very similar to the [FlashStorage_SAMD.h](https://github.com/khoih-prog/FlashStorage_SAMD) API, with these functions:

* `void writeByte(uint32_t offset, uint8_t data)`
* `uint8_t readByte(uint32_t offset)`
* `void updateByte(uint32_t offset, uint8_t value)`
* `void writeWord(uint32_t offset, uint32_t data)`
* `uint32_t readWord(uint32_t offset)`
* `void updateWord(uint32_t offset, uint32_t value)`
* `template< typename T > T &get( uint32_t offset, T &t )`
* `template< typename T > const T &put( uint32_t offset, const T &t )`
* `void readFlashToBuffer()`
* `bool isValid()`
* `uint16_t length()`
* `void setCommitASAP(bool value = true)`
* `bool getCommitASAP()`
* `void eraseFlashSector()`

---

* `bool isValid()` returns `true` if data in the FlashStorage is valid (the data written to flash at least once by `FlashStorage.commit()` or `FlashStorage.put()`). Otherwise FlashStorage data is "dirty" and the function returns `false`.

* `void commit()` store the FlashStorage buffer data in FlashStorage. Use this with care: Every call writes the complete FlashStorage buffer data to flash. This will reduce the remaining flash-write-cycles. Don't call this method in a loop or [you will kill your flash soon](#limited-number-of-writes).

* `void setCommitASAP(bool value = true)` to set or clear the `_commitASAP` private variable (default is `true` to be safe). If `_commitASAP` is false, the call to `FlashStorage.put(), etc` won't force the `FlashStorage.commit()` to extend the flash life. You'll have to remember to call `FlashStorage.commit()` manually to save the FlashStorage buffered data into FlashStorage or data will be lost.

* `bool getCommitASAP()` to return the current value of `_commitASAP`.

---
---

### Examples

 1. [FlashStorage_Clear](examples/FlashStorage_Clear)
 2. [FlashStorage_CRC](examples/FlashStorage_CRC)
 3. [FlashStorage_get](examples/FlashStorage_get)
 4. [FlashStorage_iteration](examples/FlashStorage_iteration)
 5. [FlashStorage_put](examples/FlashStorage_put)
 6. [FlashStorage_read](examples/FlashStorage_read)
 7. [FlashStorage_update](examples/FlashStorage_update)
 8. [FlashStorage_write](examples/FlashStorage_write)
 9. [FlashStoreAndRetrieve](examples/FlashStoreAndRetrieve)
10. [StoreNameAndSurname](examples/StoreNameAndSurname)
11. [**multiFileProject**](examples/multiFileProject) **New**


---
---

### Example [StoreNameAndSurname](examples/StoreNameAndSurname)


```cpp
#define FLASH_DEBUG               1

// Can be included as many times as necessary, without `Multiple Definitions` Linker Error
#include <FlashStorage_RTL8720.h>

const int WRITTEN_SIGNATURE = 0xBEEFDEED;

// Create a structure that is big enough to contain a name
// and a surname. The "valid" variable is set to "true" once
// the structure is filled with actual data for the first time.
typedef struct
{
  char name[100];
  char surname[100];
} Person;

void setup()
{
  Serial.begin(115200);
  while (!Serial);

  delay(200);

  Serial.print(F("\nStart StoreNameAndSurname on ")); Serial.println(BOARD_NAME);
  Serial.println(FLASH_STORAGE_RTL8720_VERSION);

  Serial.print("FlashStorage length: ");
  Serial.println(FlashStorage.length());

  // Check signature at address 0
  int signature;

  // Create a "Person" variable and call it "owner"
  uint16_t storedAddress = 0;
  Person owner;

  FlashStorage.get(storedAddress, signature);

  // If the FlashStorage is empty then no WRITTEN_SIGNATURE
  if (signature == WRITTEN_SIGNATURE)
  {
    FlashStorage.get(storedAddress + sizeof(signature), owner);

    // Say hello to the returning user!
    Serial.print("Hi "); Serial.print(owner.name); Serial.print(" "); Serial.print(owner.surname);
    Serial.println(", nice to see you again :-)");

    Serial.println("Clearing WRITTEN_SIGNATURE for next try");

    FlashStorage.put(0, 0);
    Serial.println("Done clearing signature in FlashStorage. You can reset now");
  }
  else
  {
    Serial.println("EEPROM is empty, writing WRITTEN_SIGNATURE and some example data:");

    FlashStorage.put(storedAddress, WRITTEN_SIGNATURE);

    // ...in this case we ask for user data.
    Serial.setTimeout(30000);
    Serial.print("Insert your name : ");
    String name = Serial.readStringUntil('\n');
    Serial.println(name);
    Serial.print("Insert your surname : ");
    String surname = Serial.readStringUntil('\n');
    Serial.println(surname);

    // Fill the "owner" structure with the data entered by the user...
    name.toCharArray(owner.name, 100);
    surname.toCharArray(owner.surname, 100);

    // ...and finally save everything into FlashStorage
    FlashStorage.put(storedAddress + sizeof(signature), owner);

    // Print a confirmation of the data inserted.
    Serial.print("<< Your name: "); Serial.print(owner.name);
    Serial.print(". Your surname: "); Serial.print(owner.surname);
    Serial.println(" >> have been saved. Thank you!");
  }
}

void loop()
{
  // Do nothing...
}
```

---
---

### Debug Terminal Output Samples


#### 1. StoreNameAndSurname on Rtlduino RTL8720DN

#### 1.1 First Start

```
Start StoreNameAndSurname on Rtlduino RTL8720DN
FlashStorage_RTL8720 v1.1.0
FlashStorage length: 4096
EEPROM is empty, writing WRITTEN_SIGNATURE and some example data:
Insert your name : First_Name
Insert your surname : Last_Name

<< Your name: First_Name. Your surname: Last_Name >> have been saved. Thank you!
```

#### 1.2 After Restart

```
Start StoreNameAndSurname on Rtlduino RTL8720DN
FlashStorage_RTL8720 v1.1.0
FlashStorage length: 4096
Hi First_Name Last_Name, nice to see you again :-)
Clearing WRITTEN_SIGNATURE for next try
Done clearing signature in FlashStorage. You can reset now
```

---

### 2. FlashStorage_read on Rtlduino RTL8720DN

The following is the sample terminal output when running example [FlashStorage_read](examples/FlashStorage_read) on Rtlduino RTL8720DN


```
Start FlashStorage_read on Rtlduino RTL8720DN
FlashStorage_RTL8720 v1.1.0
FlashStorage length: 4096
0	100
1	101
2	102
3	103
4	104
5	105
...
```

---
---

## FAQ

### Can I use a single object to store more stuff?

Yes, you can declare a complex `struct` with more fields and call a `FlashStorage.put()` to store the entire structure. See the [StoreNameAndSurname](examples/StoreNameAndSurname) for how to do it.

### The content of the FlashStorage is erased each time a new sketch is uploaded?

Not with RTL8720DN, RTL8722DM, RTM8722CSM, etc.

### Do you recommend to use FLASH instead of EEPROM?

No. If your board provides an integrated-EEPROM, it's advisable to use that because EEPROM has longer lifetime, number of write cycles, etc.

In the absence of an integrated-EEPROM or its size is too small for your use-case, you can use this library to use a small portion flash memory as FlashStorage, provided that you keep in mind the limits as in [Limited number of writes](#limited-number-of-writes)

---
---

### Troubleshooting

If you get compilation errors, more often than not, you may need to install a newer version of the core for Arduino boards.

Sometimes, the library will only work if you update the board core to the latest version because I am using newly added functions.

---
---

### Issues

Submit issues to: [FlashStorage_RTL8720 issues](https://github.com/khoih-prog/FlashStorage_RTL8720/issues)

---

### TO DO

1. Search for bug and improvement.
2. Similar features for remaining Arduino boards

---

### DONE

1. Basic FlashStorage for RTL8720.
2. Add Table of Contents
3. Fix `multiple-definitions` linker error. 


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

- The library is licensed under [MIT](https://github.com/khoih-prog/FlashStorage_RTL8720/blob/main/LICENSE)

---

### Copyright

Copyright 2021- Khoi Hoang

