/******************************************************************************************************************************************
  FlashStorage_write.ino
  For Realtek RTL8720DN, etc. using Flash

  The FlashStorage_RTL8720DN library aims to provide a convenient way to store and retrieve user's data using the non-volatile flash memory
  of RTL8720DN, etc.. It's using the buffered read and write to minimize the access to Flash.
  It now supports writing and reading the whole object, not just byte-and-byte.

  Inspired by Cristian Maglie's FlashStorage (https://github.com/cmaglie/FlashStorage)

  Built by Khoi Hoang https://github.com/khoih-prog/FlashStorage_RTL8720DN
  Licensed under MIT license
 ******************************************************************************************************************************************/
/*
   FlashStorage Write

   Stores values read from analog input 0 into the FlashStorage. These values will stay in the FlashStorage 
   when the board is turned off and may be retrieved later by another sketch.
*/

#define FLASH_DEBUG               1

// Can be included as many times as necessary, without `Multiple Definitions` Linker Error
#include <FlashStorage_RTL8720.h>

// start reading from the first byte (address 0) of the FlashStorage
int address = 0;

void setup()
{
  Serial.begin(115200);
  while (!Serial);

  delay(200);

  Serial.print(F("\nStart FlashStorage_write on ")); Serial.println(BOARD_NAME);
  Serial.println(FLASH_STORAGE_RTL8720_VERSION);

  Serial.print("FlashStorage length: ");
  Serial.println(FlashStorage.length());

  FlashStorage.setCommitASAP(false);
}

uint8_t startNumber = 100;

void loop()
{
  unsigned long startMillis = millis();
  
  for (int i = 0 ; i < FlashStorage.length() ; i++) 
  {
    /***
      The function FlashStorage.update(address, val) is equivalent to the following:
  
      if( FlashStorage.read(address) != val )
      {
        FlashStorage.write(address, val);
      }
    ***/
    FlashStorage.writeByte(i, startNumber++);
  }
  
  FlashStorage.commit();

  Serial.print("Done writing emulated FlashStorage. Time spent (ms) = "); Serial.println(millis() - startMillis);

  delay(60000);
}
