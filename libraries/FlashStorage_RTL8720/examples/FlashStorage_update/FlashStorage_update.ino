/******************************************************************************************************************************************
  FlashStorage_update.ino
  For Realtek RTL8720DN, etc. using Flash

  The FlashStorage_RTL8720DN library aims to provide a convenient way to store and retrieve user's data using the non-volatile flash memory
  of RTL8720DN, etc.. It's using the buffered read and write to minimize the access to Flash.
  It now supports writing and reading the whole object, not just byte-and-byte.

  Inspired by Cristian Maglie's FlashStorage (https://github.com/cmaglie/FlashStorage)

  Built by Khoi Hoang https://github.com/khoih-prog/FlashStorage_RTL8720DN
  Licensed under MIT license
 ******************************************************************************************************************************************/
/***
   FlashStorage Update method

   Stores values read from analog input 0 into the FlashStorage.
   These values will stay in the FlashStorage when the board is
   turned off and may be retrieved later by another sketch.

   If a value has not changed in the FlashStorage, it is not overwritten
   which would reduce the life span of the FlashStorage unnecessarily.

   Released using MIT licence.
 ***/

#define FLASH_DEBUG               1

// Can be included as many times as necessary, without `Multiple Definitions` Linker Error
#include <FlashStorage_RTL8720.h>

/** the current address in the FlashStorage (i.e. which byte we're going to write to next) **/
int address = 0;

void setup()
{
  Serial.begin(115200);
  while (!Serial);

  delay(200);

  Serial.print(F("\nStart FlashStorage_read on ")); Serial.println(BOARD_NAME);
  Serial.println(FLASH_STORAGE_RTL8720_VERSION);

  Serial.print("FlashStorage length: ");
  Serial.println(FlashStorage.length());

  FlashStorage.setCommitASAP(false);
}

uint8_t startNumber = 0;

void loop()
{
  unsigned long startMillis = millis();
 
  for (int i = 0 ; i < FlashStorage.length() ; i++) 
  {
    /***
      The function FlashStorage.updateByte(address, val) is equivalent to the following:
  
      if( FlashStorage.readByte(address) != val )
      {
        FlashStorage.writeByte(address, val);
      }
    ***/
    FlashStorage.updateByte(i, startNumber++);
  }
  
  FlashStorage.commit();

  Serial.print("Done updating FlashStorage. Time spent (ms) = "); Serial.println(millis() - startMillis);

  delay(60000);
}
