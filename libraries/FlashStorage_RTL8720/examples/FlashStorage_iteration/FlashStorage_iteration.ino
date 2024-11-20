/******************************************************************************************************************************************
  FlashStorage_iteration.ino
  For Realtek RTL8720DN, etc. using Flash

  The FlashStorage_RTL8720DN library aims to provide a convenient way to store and retrieve user's data using the non-volatile flash memory
  of RTL8720DN, etc.. It's using the buffered read and write to minimize the access to Flash.
  It now supports writing and reading the whole object, not just byte-and-byte.

  Inspired by Cristian Maglie's FlashStorage (https://github.com/cmaglie/FlashStorage)

  Built by Khoi Hoang https://github.com/khoih-prog/FlashStorage_RTL8720DN
  Licensed under MIT license
 ******************************************************************************************************************************************/
/***
    FlashStorage_iteration example.

    A set of example snippets highlighting the simplest methods for traversing the FlashStorage.

    Running this sketch is not necessary, this is simply highlighting certain programming methods.

    Written by Christopher Andrews 2015
    Released under MIT licence.
***/

#define FLASH_DEBUG               1

// Can be included as many times as necessary, without `Multiple Definitions` Linker Error
#include <FlashStorage_RTL8720.h>

void setup()
{
  Serial.begin(115200);
  while (!Serial);

  delay(200);

  Serial.print(F("\nStart FlashStorage_iteration on ")); Serial.println(BOARD_NAME);
  Serial.println(FLASH_STORAGE_RTL8720_VERSION);

  Serial.print("FlashStorage length: ");
  Serial.println(FlashStorage.length());

  /***
    Iterate the FlashStorage using a for loop.
  ***/

  FlashStorage.setCommitASAP(false);
  
  for (int index = 0 ; index < FlashStorage.length() ; index++) 
  {
    // Add one to each cell in the FlashStorage
    FlashStorage.writeByte(index, FlashStorage.readByte(index) + 1);
  }

  if (FlashStorage.commit())
  {
    Serial.println("Done for loop");
  }
  else
  {
    Serial.println("Error committing for loop");
  }

  /***
    Iterate the FlashStorage using a while loop.
  ***/

  int index = 0;

  while (index < FlashStorage.length()) 
  {
    //Add one to each cell in the FlashStorage
    // Add one to each cell in the FlashStorage
    FlashStorage.writeByte(index, FlashStorage.readByte(index) + 1);
    index++;
  }

  if (FlashStorage.commit())
  {
    Serial.println("Done while loop");
  }
  else
  {
    Serial.println("Error committing while loop");
  }

  /***
    Iterate the FlashStorage using a do-while loop.
  ***/

  index = 0;  //Used 'idx' to avoid name conflict with 'index' above.

  do 
  {
    //Add one to each cell in the FlashStorage
    // Add one to each cell in the FlashStorage
    FlashStorage.writeByte(index, FlashStorage.readByte(index) + 1);
    index++;
  } while (index < FlashStorage.length());

  if (FlashStorage.commit())
  {
    Serial.println("Done do-while loop");
  }
  else
  {
    Serial.println("Error committing do-while loop");
  }

  // Necessary only for code below
  FlashStorage.setCommitASAP(true);
  
} //End of setup function.

void loop() {}
