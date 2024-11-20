/******************************************************************************************************************************************
  FlashStorage_Clear.ino
  For Realtek RTL8720DN, etc. using Flash

  The FlashStorage_RTL8720DN library aims to provide a convenient way to store and retrieve user's data using the non-volatile flash memory
  of RTL8720DN, etc.. It's using the buffered read and write to minimize the access to Flash.
  It now supports writing and reading the whole object, not just byte-and-byte.

  Inspired by Cristian Maglie's FlashStorage (https://github.com/cmaglie/FlashStorage)

  Built by Khoi Hoang https://github.com/khoih-prog/FlashStorage_RTL8720DN
  Licensed under MIT license
 ******************************************************************************************************************************************/

#define FLASH_DEBUG               1

// Can be included as many times as necessary, without `Multiple Definitions` Linker Error
#include <FlashStorage_RTL8720.h>

void setup() 
{
  Serial.begin(115200);
  while (!Serial);

  delay(200);

  Serial.print(F("\nStart FlashStorage_Clear on ")); Serial.println(BOARD_NAME);
  Serial.println(FLASH_STORAGE_RTL8720_VERSION);
  
  Serial.print("FlashStorage length: ");
  Serial.println(FlashStorage.length());

  unsigned long startMillis = millis();
  
  FlashStorage.eraseFlashSector();

  // The time spent can be very short (4-5ms) if the FlashStorage is not dirty.
  Serial.print("Done clearing FlashStorage. Time spent (ms) = "); Serial.println(millis() - startMillis);
}

void loop() 
{
  /** Empty loop. **/
}
