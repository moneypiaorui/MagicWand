/******************************************************************************************************************************************
  FlashStorage_STM32F1_Impl.h
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
  
  Version: 1.1.0

  Version Modified By   Date        Comments
  ------- -----------  ----------   -----------
  1.0.0   K Hoang      16/08/2021  Initial coding to support to STM32F1
  1.0.1   K Hoang      17/08/2021  Fix bug
  1.1.0   K Hoang      25/01/2022  Fix `multiple-definitions` linker error
 ******************************************************************************************************************************************/

#pragma once


#ifndef FlashStorage_STM32F1_Impl_hpp
#define FlashStorage_STM32F1_Impl_hpp


FlashClass::FlashClass(const void *flash_addr, const uint32_t& size) :
  PAGE_SIZE(1024),
  PAGES(1),
  MAX_FLASH(PAGE_SIZE * PAGES),
  ROW_SIZE(MAX_FLASH / 64),
  flash_address((volatile void *)flash_addr),
  flash_size(size)
{
}

//////////////////////////////////////////////////////////

void FlashClass::write(const volatile void *flash_ptr, const void *data)
{
  uint16_t *AddressPtr;
  uint16_t *valuePtr;
  
  AddressPtr  = (uint16_t *)flash_ptr;
  valuePtr    = (uint16_t *)data;
  
  // we're using 16 bit words here while flash_size in bytes 
  uint32_t size = flash_size / 2;
    
  while (size)
  {
    // unlock the flash
    // Key 1 : 0x45670123
    // Key 2 : 0xCDEF89AB
    FLASH->KEYR = FLASH_KEY1;
    FLASH->KEYR = FLASH_KEY2;
    FLASH->CR &= ~(1 << 1); // ensure PER is low
    FLASH->CR |= (1 << 0);  // set the PG bit
    
    *(AddressPtr) = *(valuePtr);

    while (FLASH->SR & (1 << 0)); // wait while busy

    if (FLASH->SR & (1 << 2))
    {
      FLASH_LOGDEBUG(F("Flash not erased"));
      return;
    }

    if (FLASH->SR & (1 << 4))
    {
      FLASH_LOGDEBUG(F("Write protect error"));
      return;
    }

    AddressPtr++;
    valuePtr++;
    size--;
  }
}

//////////////////////////////////////////////////////////

void FlashClass::erase(const volatile void *flash_ptr)
{
  // unlock the flash
  // Key 1 : 0x45670123
  // Key 2 : 0xCDEF89AB
  FLASH->KEYR = FLASH_KEY1;
  FLASH->KEYR = FLASH_KEY2;
  FLASH->CR &= ~(1 << 0); // Ensure PG bit is low
  FLASH->CR |= (1 << 1);  // set the PER bit
  FLASH->AR = (uint32_t) flash_ptr;
  FLASH->CR |= (1 << 6);  // set the start bit

  while (FLASH->SR & (1 << 0)); // wait while busy
}

//////////////////////////////////////////////////////////

void FlashClass::read(const volatile void *flash_ptr, void *data)
{
  uint16_t *AddressPtr;
  uint16_t *valuePtr;
  
  AddressPtr  = (uint16_t *) flash_ptr;
  valuePtr    = (uint16_t *) data;
   
  // we're using 16 bit words here while flash_size in bytes 
  uint32_t size = flash_size / 2;
 
  while (size)
  {
    *((uint16_t *)valuePtr) = *((uint16_t *)AddressPtr);

    valuePtr++;
    AddressPtr++;
    size--;
  } 
}

#endif      //#ifndef FlashStorage_STM32F1_Impl_hpp

