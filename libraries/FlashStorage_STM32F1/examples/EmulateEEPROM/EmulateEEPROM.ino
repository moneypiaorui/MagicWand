/******************************************************************************************************************************************
  EmulatedEEPROM.ino
  For STM32F1 using Flash emulated-EEPROM

  The FlashStorage_STM32F1 library aims to provide a convenient way to store and retrieve user's data using the non-volatile flash memory
  of STM32F1. It now supports writing and reading the whole object, not just byte-and-byte.

  Based on and modified from Cristian Maglie's FlashStorage (https://github.com/cmaglie/FlashStorage)

  Built by Khoi Hoang https://github.com/khoih-prog/FlashStorage_STM32F1
  Licensed under LGPLv3 license

  Orginally written by A. Christian

  Copyright (c) 2015-2016 Arduino LLC.  All right reserved.
  Copyright (c) 2021 Khoi Hoang.

  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
  as published bythe Free Software Foundation, either version 3 of the License, or (at your option) any later version.
  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
  You should have received a copy of the GNU Lesser General Public License along with this library.
  If not, see (https://www.gnu.org/licenses/)
 ******************************************************************************************************************************************/

// Demonstrate how to use FlashAsEEPROM_STM32F1 with an API that is similar to the EEPROM library.

// Use 0-2. Larger for more debugging messages
#define FLASH_DEBUG       2

// You can select another sector. Be careful not larger than (REGISTERED_NUMBER_FLASH_SECTORS - 1) and large enough not to overwrite your program
// Default is (REGISTERED_NUMBER_FLASH_SECTORS - 1) if you don't specify here
#define USING_FLASH_SECTOR_NUMBER           (REGISTERED_NUMBER_FLASH_SECTORS - 2)

// To be included only in main(), .ino with setup() to avoid `Multiple Definitions` Linker Error
#include <FlashStorage_STM32F1.h>

const int WRITTEN_SIGNATURE = 0xBEEFDEED;

void setup()
{
  Serial.begin(115200);
  while (!Serial);

  delay(200);

  Serial.print(F("\nStart EmulatedEEPROM on ")); Serial.println(BOARD_NAME);
  Serial.println(FLASH_STORAGE_STM32F1_VERSION);

  Serial.print("EEPROM length: ");
  Serial.println(EEPROM.length());

  EEPROM.init();

  // Check signature at address 0
  int signature;

  EEPROM.get(0, signature);

  // If the EEPROM is empty then no WRITTEN_SIGNATURE
  if (signature != WRITTEN_SIGNATURE)
  {
    Serial.println("EEPROM is empty, writing WRITTEN_SIGNATURE and some example data:");

    EEPROM.put(0, WRITTEN_SIGNATURE);

    Serial.print("->");

    for (int i = sizeof(WRITTEN_SIGNATURE); i < 20; i++)
    {
      EEPROM.write(i, 100 + i);
      Serial.print(" ");
      Serial.print(100 + i);
    }

    // commit() saves all the changes to EEPROM, it must be called
    // every time the content of virtual EEPROM is changed to make
    // the change permanent.
    // This operation burns Flash write cycles and should not be
    // done too often. See readme for details:
    // https://github.com/khoih-prog/FlashStorage_SAMD#limited-number-of-writes

    EEPROM.commit();
    Serial.println("\nDone writing to emulated EEPROM. You can reset now to test");
  }
  else
  {
    EEPROM.get(0, signature);

    Serial.print("EEPROM has been written.Signature = 0x"); Serial.println(signature, HEX);

    Serial.println("Here is the content of the next 16 bytes:");

    Serial.print("->");

    for (int i = sizeof(WRITTEN_SIGNATURE); i < 20; i++)
    {
      Serial.print(" ");
      Serial.print(EEPROM.read(i));
    }

    Serial.println("\nClearing WRITTEN_SIGNATURE for next try");

    EEPROM.put(0, 0);

    if (!EEPROM.getCommitASAP())
    {
      Serial.println("CommitASAP not set. Need commit()");
      EEPROM.commit();
    }
    
    Serial.println("Done clearing signature in emulated EEPROM. You can reset now");
  }
}

void loop()
{
}
