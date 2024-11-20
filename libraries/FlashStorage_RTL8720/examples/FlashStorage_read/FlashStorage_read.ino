/******************************************************************************************************************************************
  FlashStorage_read.ino
  For Realtek RTL8720DN, etc. using Flash

  The FlashStorage_RTL8720DN library aims to provide a convenient way to store and retrieve user's data using the non-volatile flash memory
  of RTL8720DN, etc.. It's using the buffered read and write to minimize the access to Flash.
  It now supports writing and reading the whole object, not just byte-and-byte.

  Inspired by Cristian Maglie's FlashStorage (https://github.com/cmaglie/FlashStorage)

  Built by Khoi Hoang https://github.com/khoih-prog/FlashStorage_RTL8720DN
  Licensed under MIT license
 ******************************************************************************************************************************************/
/*
   FlashStorage Read

   Reads the value of each byte of the FlashStorage and prints it to the computer.
   This example code is in the public domain.
*/

#define FLASH_DEBUG               1

// Can be included as many times as necessary, without `Multiple Definitions` Linker Error
#include <FlashStorage_RTL8720.h>

// start reading from the first byte (address 0) of the FlashStorage
int address = 0;
byte value;

void setup()
{
  Serial.begin(115200);
  while (!Serial);

  delay(200);

  Serial.print(F("\nStart FlashStorage_read on ")); Serial.println(BOARD_NAME);
  Serial.println(FLASH_STORAGE_RTL8720_VERSION);

  Serial.print("FlashStorage length: ");
  Serial.println(FlashStorage.length());
}

void loop() 
{
  // read a byte from the current address of the FlashStorage
  value = FlashStorage.readByte(address);

  Serial.print(address);
  Serial.print("\t");
  Serial.print(value, DEC);
  Serial.println();
 
  if (++address == FlashStorage.length()) 
  {
    address = 0;
  }

  /***
    As the FlashStorage sizes are powers of two, wrapping (preventing overflow) of an
    FlashStorage address is also doable by a bitwise and of the length - 1.

    ++address &= FlashStorage.length() - 1;
  ***/

  delay(500);
}
