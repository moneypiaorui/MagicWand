/******************************************************************************************************************************************
  EEPROM_write.ino
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
/*
   EEPROM Write

   Stores values read from analog input 0 into the EEPROM. These values will stay in the EEPROM
   when the board is turned off and may be retrieved later by another sketch.
*/

// Use 0-2. Larger for more debugging messages
#define FLASH_DEBUG       2

// You can select another sector. Be careful not larger than (REGISTERED_NUMBER_FLASH_SECTORS - 1) and large enough not to overwrite your program
// Default is (REGISTERED_NUMBER_FLASH_SECTORS - 1) if you don't specify here
#define USING_FLASH_SECTOR_NUMBER           (REGISTERED_NUMBER_FLASH_SECTORS - 2)

// To be included only in main(), .ino with setup() to avoid `Multiple Definitions` Linker Error
#include <FlashStorage_STM32F1.h>

// start reading from the first byte (address 0) of the EEPROM
int address = 0;

void setup()
{
  Serial.begin(115200);
  while (!Serial);

  delay(200);

  Serial.print(F("\nStart EEPROM_write on ")); Serial.println(BOARD_NAME);
  Serial.println(FLASH_STORAGE_STM32F1_VERSION);

  Serial.print("EEPROM length: ");
  Serial.println(EEPROM.length());

  EEPROM.init();
}

void loop()
{
  unsigned long startMillis = millis();

  for (int i = 0 ; i < EEPROM.length() ; i++)
  {
    /***
      The function EEPROM.update(address, val) is equivalent to the following:

      if( EEPROM.read(address) != val )
      {
        EEPROM.write(address, val);
      }
    ***/

    //Serial.print("Writing address = "); Serial.println(i);
    EEPROM.write(i, (uint8_t) ( i + 1));
  }

  EEPROM.commit();

  Serial.print("Done writing emulated EEPROM. Time spent (ms) = "); Serial.println(millis() - startMillis);

  delay(60000);
}
