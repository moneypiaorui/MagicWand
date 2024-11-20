/******************************************************************************************************************************************
  FlashStorage_put.ino
  For Realtek RTL8720DN, etc. using Flash

  The FlashStorage_RTL8720DN library aims to provide a convenient way to store and retrieve user's data using the non-volatile flash memory
  of RTL8720DN, etc.. It's using the buffered read and write to minimize the access to Flash.
  It now supports writing and reading the whole object, not just byte-and-byte.

  Inspired by Cristian Maglie's FlashStorage (https://github.com/cmaglie/FlashStorage)

  Built by Khoi Hoang https://github.com/khoih-prog/FlashStorage_RTL8720DN
  Licensed under MIT license
 ******************************************************************************************************************************************/
/***
    FlashStorage_put example.

    This shows how to use the FlashStorage.put() method. Also, this sketch will pre-set the FlashStorage data
    for the example sketch FlashStorage_get.

    Note, unlike the single byte version FlashStorage.write(), the put method will use update semantics. As in a byte
    will only be written to the FlashStorage if the data is actually different.

    Written by Christopher Andrews 2015
    Released under MIT licence.
***/

#define FLASH_DEBUG               1

// Can be included as many times as necessary, without `Multiple Definitions` Linker Error
#include <FlashStorage_RTL8720.h>

struct MyObject
{
  float field1;
  byte field2;
  char name[10];
};

void setup()
{
  Serial.begin(115200);
  while (!Serial);

  delay(200);

  Serial.print(F("\nStart FlashStorage_put on ")); Serial.println(BOARD_NAME);
  Serial.println(FLASH_STORAGE_RTL8720_VERSION);

  Serial.print("FlashStorage length: ");
  Serial.println(FlashStorage.length());

  float f = 123.456f;  //Variable to store in FlashStorage.
  int FlashAddress = 0;   //Location we want the data to be put.


  //One simple call, with the address first and the object second.
  FlashStorage.put(FlashAddress, f);

  Serial.println("Written float data type!");

  /** Put is designed for use with custom structures also. **/

  //Data to store.
  MyObject customVar =
  {
    3.14159f,
    65,
    "Working!"
  };

  FlashAddress += sizeof(float); //Move address to the next byte after float 'f'.

  FlashStorage.put(FlashAddress, customVar);
  Serial.print("Written custom data type! \n\nView the example sketch FlashStorage_get to see how you can retrieve the values!");
}

void loop()
{
  /* Empty loop */
}
