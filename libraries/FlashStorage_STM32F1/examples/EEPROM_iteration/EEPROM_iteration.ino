/******************************************************************************************************************************************
  EEPROM_iteration.ino
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
/***
    eeprom_iteration example.

    A set of example snippets highlighting the simplest methods for traversing the EEPROM.

    Running this sketch is not necessary, this is simply highlighting certain programming methods.

    Written by Christopher Andrews 2015
    Released under MIT licence.
***/

// Use 0-2. Larger for more debugging messages
#define FLASH_DEBUG       2

// You can select another sector. Be careful not larger than (REGISTERED_NUMBER_FLASH_SECTORS - 1) and large enough not to overwrite your program
// Default is (REGISTERED_NUMBER_FLASH_SECTORS - 1) if you don't specify here
#define USING_FLASH_SECTOR_NUMBER           (REGISTERED_NUMBER_FLASH_SECTORS - 2)

// To be included only in main(), .ino with setup() to avoid `Multiple Definitions` Linker Error
#include <FlashStorage_STM32F1.h>

void setup()
{
  Serial.begin(115200);
  while (!Serial);

  delay(200);

  Serial.print(F("\nStart EEPROM_iteration on ")); Serial.println(BOARD_NAME);
  Serial.println(FLASH_STORAGE_STM32F1_VERSION);

  Serial.print("EEPROM length: ");
  Serial.println(EEPROM.length());

  EEPROM.init();

  /***
    Iterate the EEPROM using a for loop.
  ***/

  for (int index = 0 ; index < EEPROM.length() ; index++) 
  {
    // Add one to each cell in the EEPROM
    EEPROM.write(index, EEPROM.read(index) + 1);
  }

  EEPROM.commit();

  Serial.println("Done for loop");

  /***
    Iterate the EEPROM using a while loop.
  ***/

  int index = 0;

  while (index < EEPROM.length()) 
  {
    // Add one to each cell in the EEPROM
    EEPROM.write(index, EEPROM.read(index) + 1);
    index++;
  }

  EEPROM.commit();

  Serial.println("Done while loop");

  /***
    Iterate the EEPROM using a do-while loop.
  ***/

  int idx = 0;  //Used 'idx' to avoid name conflict with 'index' above.

  do 
  {
    // Add one to each cell in the EEPROM
    EEPROM.write(index, EEPROM.read(index) + 1);
    idx++;
  } while (idx < EEPROM.length());

  EEPROM.commit();

  Serial.println("Done do-while loop");
  
} //End of setup function.

void loop() {}
