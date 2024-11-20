/****************************************************************************************************************************
  multiFileProject.ino
  
  For Realtek RTL8720DN, etc. using Flash

  The FlashStorage_RTL8720DN library aims to provide a convenient way to store and retrieve user's data using the non-volatile flash memory
  of RTL8720DN, etc.. It's using the buffered read and write to minimize the access to Flash.
  It now supports writing and reading the whole object, not just byte-and-byte.

  Inspired by Cristian Maglie's FlashStorage (https://github.com/cmaglie/FlashStorage)

  Built by Khoi Hoang https://github.com/khoih-prog/FlashStorage_RTL8720DN
  Licensed under MIT license
*****************************************************************************************************************************/

// To demo how to include files in multi-file Projects

#if !defined(CONFIG_PLATFORM_8721D)
  #error Only for Ameba Realtek RTL8720DN, RTL8722DM and RTM8722CSM platform.
#endif

#define FLASH_STORAGE_RTL8720_VERSION_MIN_TARGET      "FlashStorage_STM32F1 v1.1.0"
#define FLASH_STORAGE_RTL8720_VERSION_MIN             1001000

// Can be included as many times as necessary, without `Multiple Definitions` Linker Error
#include "FlashStorage_RTL8720.h"

#include "multiFileProject.h"

void setup() 
{
  Serial.begin(115200);
  while (!Serial);
  
  Serial.println("\nStart multiFileProject");
  Serial.println(FLASH_STORAGE_RTL8720_VERSION);

#if defined(FLASH_STORAGE_RTL8720_VERSION_MIN)
  if (FLASH_STORAGE_RTL8720_VERSION_INT < FLASH_STORAGE_RTL8720_VERSION_MIN)
  {
    Serial.print("Warning. Must use this example on Version equal or later than : ");
    Serial.println(FLASH_STORAGE_RTL8720_VERSION_MIN_TARGET);
  }
#endif

  testFlashStorage();
}

void loop() 
{
  // put your main code here, to run repeatedly:
}
