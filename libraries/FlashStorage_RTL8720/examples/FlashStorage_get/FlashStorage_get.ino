/******************************************************************************************************************************************
  FlashStorage_get.ino
  For Realtek RTL8720DN, etc. using Flash

  The FlashStorage_RTL8720DN library aims to provide a convenient way to store and retrieve user's data using the non-volatile flash memory
  of RTL8720DN, etc.. It's using the buffered read and write to minimize the access to Flash.
  It now supports writing and reading the whole object, not just byte-and-byte.

  Inspired by Cristian Maglie's FlashStorage (https://github.com/cmaglie/FlashStorage)

  Built by Khoi Hoang https://github.com/khoih-prog/FlashStorage_RTL8720DN
  Licensed under MIT license
 ******************************************************************************************************************************************/

/***
    FlashStorage_get example.

    This shows how to use the FlashStorage.get() method.

    To pre-set the FlashStorage data, run the example sketch FlashStorage_put. This sketch will run without it.
    However, the values shown will be shown from what ever is already on the FlashStorage.

    This may cause the serial object to print out a large string of garbage if there is no null character
    inside one of the strings loaded.

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

void secondTest()
{
  int FlashAddress = sizeof(float); //Move address to the next byte after float 'f'.

  MyObject customVar; //Variable to store custom object read from FlashStorage.
  FlashStorage.get(FlashAddress, customVar);

  Serial.println("Read custom object from FlashStorage: ");
  Serial.println(customVar.field1, 5);
  Serial.println(customVar.field2);
  Serial.println(customVar.name);
}

void setup()
{
  Serial.begin(115200);
  while (!Serial);

  delay(200);

  Serial.print(F("\nStart FlashStorage_get on ")); Serial.println(BOARD_NAME);
  Serial.println(FLASH_STORAGE_RTL8720_VERSION);

  Serial.print("FlashStorage length: ");
  Serial.println(FlashStorage.length());

  float f = 0.00f;   //Variable to store data read from FlashStorage.
  int FlashAddress = 0; //FlashStorage address to start reading from

  Serial.print("Read float from FlashStorage: ");

  //Get the float data from the FlashStorage at position 'FlashAddress'
  FlashStorage.get(FlashAddress, f);
  Serial.println(f, 3);    //This may print 'ovf, nan' if the data inside the FlashStorage is not a valid float.

  /***
    As get also returns a reference to 'f', you can use it inline.
    E.g: Serial.print( FlashStorage.get( FlashAddress, f ) );
  ***/

  /***
    Get can be used with custom structures too.
    I have separated this into an extra function.
  ***/

  secondTest(); //Run the next test.
}

void loop()
{
  /* Empty loop */
}
