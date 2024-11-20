/****************************************************************************************************************************
  multiFileProject.ino
  
  For STM32F1 using Flash emulated-EEPROM

  The FlashStorage_STM32F1 library aims to provide a convenient way to store and retrieve user's data using the non-volatile flash memory
  of STM32F1. It now supports writing and reading the whole object, not just byte-and-byte.

  Based on and modified from Cristian Maglie's FlashStorage (https://github.com/cmaglie/FlashStorage)

  Built by Khoi Hoang https://github.com/khoih-prog/FlashStorage_STM32F1
  Licensed under LGPLv3 license
*****************************************************************************************************************************/

// To demo how to include files in multi-file Projects

// The .hpp contains only definitions, and can be included as many times as necessary, without `Multiple Definitions` Linker Error
// The .h contains implementations, and can be included only in main(), .ino with setup() to avoid `Multiple Definitions` Linker Error

#if !( defined(STM32F1xx) || defined(STM32F3xx) )
  #error This code is intended to run on STM32F1/F3 platform! Please check your Tools->Board setting.  
#endif

#define FLASH_STORAGE_STM32F1_VERSION_MIN_TARGET      "FlashStorage_STM32F1 v1.1.0"
#define FLASH_STORAGE_STM32F1_VERSION_MIN             1001000

// To be included only in main(), .ino with setup() to avoid `Multiple Definitions` Linker Error
#include "FlashStorage_STM32F1.h"

#include "multiFileProject.h"

void setup() 
{
  Serial.begin(115200);
  while (!Serial);
  
  Serial.println("\nStart multiFileProject");
  Serial.println(FLASH_STORAGE_STM32F1_VERSION);

#if defined(FLASH_STORAGE_STM32F1_VERSION_MIN)
  if (FLASH_STORAGE_STM32F1_VERSION_INT < FLASH_STORAGE_STM32F1_VERSION_MIN)
  {
    Serial.print("Warning. Must use this example on Version equal or later than : ");
    Serial.println(FLASH_STORAGE_STM32F1_VERSION_MIN_TARGET);
  }
#endif

  testEEPROM();
}

void loop() 
{
  // put your main code here, to run repeatedly:
}
