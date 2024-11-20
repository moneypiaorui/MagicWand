/******************************************************************************************************************************************
  FlashStorage_STM32F1.hpp
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

#ifndef FlashStorage_STM32F1_hpp
#define FlashStorage_STM32F1_hpp

#if !( defined(STM32F1xx) || defined(STM32F3xx) )
  #error This code is intended to run on STM32F1/F3 platform! Please check your Tools->Board setting.  
#endif

#ifndef FLASH_STORAGE_STM32F1_VERSION
  #define FLASH_STORAGE_STM32F1_VERSION             "FlashStorage_STM32F1 v1.1.0"

  #define FLASH_STORAGE_STM32F1_VERSION_MAJOR       1
  #define FLASH_STORAGE_STM32F1_VERSION_MINOR       1
  #define FLASH_STORAGE_STM32F1_VERSION_PATCH       0

#define FLASH_STORAGE_STM32F1_VERSION_INT           1001000

#endif

#include <Arduino.h>

/////////////////////////////////////////////////////

#ifndef FLASH_DEBUG
  #define FLASH_DEBUG               0
#endif

#if !defined(FLASH_DEBUG_OUTPUT)
  #define FLASH_DEBUG_OUTPUT    Serial
#endif

const char FLASH_MARK[]  = "[FLASH] ";
const char FLASH_SP[]    = " ";

#define FLASH_PRINT          FLASH_DEBUG_OUTPUT.print
#define FLASH_PRINTLN        FLASH_DEBUG_OUTPUT.println
#define FLASH_FLUSH          FLASH_DEBUG_OUTPUT.flush

#define FLASH_PRINT_MARK     FLASH_PRINT(FLASH_MARK)
#define FLASH_PRINT_SP       FLASH_PRINT(FLASH_SP)

/////////////////////////////////////////////////////

#define FLASH_LOGERROR(x)         if(FLASH_DEBUG>0) { FLASH_PRINT("[FLASH] "); FLASH_PRINTLN(x); }
#define FLASH_LOGERROR0(x)        if(FLASH_DEBUG>0) { FLASH_PRINT(x); }
#define FLASH_HEXLOGERROR0(x)     if(FLASH_DEBUG>0) { FLASH_PRINTLN(x, HEX); }
#define FLASH_LOGERROR1(x,y)      if(FLASH_DEBUG>0) { FLASH_PRINT("[FLASH] "); FLASH_PRINT(x); FLASH_PRINT_SP; FLASH_PRINTLN(y); }
#define FLASH_LOGERROR2(x,y,z)    if(FLASH_DEBUG>0) { FLASH_PRINT("[FLASH] "); FLASH_PRINT(x); FLASH_PRINT_SP; FLASH_PRINT(y); FLASH_PRINT_SP; FLASH_PRINTLN(z); }
#define FLASH_LOGERROR3(x,y,z,w)  if(FLASH_DEBUG>0) { FLASH_PRINT("[FLASH] "); FLASH_PRINT(x); FLASH_PRINT_SP; FLASH_PRINT(y); FLASH_PRINT_SP; FLASH_PRINT(z); FLASH_PRINT_SP; FLASH_PRINTLN(w); }

/////////////////////////////////////////////////////

#define FLASH_LOGDEBUG(x)         if(FLASH_DEBUG>1) { FLASH_PRINT("[FLASH] "); FLASH_PRINTLN(x); }
#define FLASH_LOGDEBUG0(x)        if(FLASH_DEBUG>1) { FLASH_PRINT(x); }
#define FLASH_HEXLOGDEBUG0(x)     if(FLASH_DEBUG>1) { FLASH_PRINTLN(x, HEX); }
#define FLASH_LOGDEBUG1(x,y)      if(FLASH_DEBUG>1) { FLASH_PRINT("[FLASH] "); FLASH_PRINT(x); FLASH_PRINT_SP; FLASH_PRINTLN(y); }
#define FLASH_LOGDEBUG2(x,y,z)    if(FLASH_DEBUG>1) { FLASH_PRINT("[FLASH] "); FLASH_PRINT(x); FLASH_PRINT_SP; FLASH_PRINT(y); FLASH_PRINT_SP; FLASH_PRINTLN(z); }
#define FLASH_LOGDEBUG3(x,y,z,w)  if(FLASH_DEBUG>1) { FLASH_PRINT("[FLASH] "); FLASH_PRINT(x); FLASH_PRINT_SP; FLASH_PRINT(y); FLASH_PRINT_SP; FLASH_PRINT(z); FLASH_PRINT_SP; FLASH_PRINTLN(w); }


//////////////////////////////////////////////////////////

class FlashClass 
{
public:
  FlashClass(const void *flash_addr = NULL, const uint32_t& size = 0);

  void write(const void *data) { write(flash_address, data); }
  void erase()                 { erase(flash_address);       }
  void read(void *data)        { read(flash_address, data);  }

  void write(const volatile void *flash_ptr, const void *data);
  
  void read(const volatile void *flash_ptr, void *data);

private:
  void erase(const volatile void *flash_ptr);

  const uint32_t PAGE_SIZE, PAGES, MAX_FLASH, ROW_SIZE;
  const volatile void *flash_address;
  const uint32_t flash_size;
};

//////////////////////////////////////////////////////////

template<class T>
class FlashStorageClass
{
public:
  FlashStorageClass(const void *flash_addr) : flash(flash_addr, sizeof(T)) { };

  // Write data into flash memory
  inline void write(T &data)  { flash.erase(); flash.write(&data); }

  // Read data from flash memory.
  inline void read(T &data)  { flash.read(&data); }

private:
  FlashClass flash;
};

///////////////////////////////////////////////////////////////////////////////////////

#ifndef EEPROM_EMULATION_SIZE
  #define EEPROM_EMULATION_SIZE ( 1024 - sizeof(bool) - sizeof(uint32_t) )
#endif

const uint32_t STM32F1_EEPROM_EMULATION_SIGNATURE = 0xFEEDDEED;

typedef struct 
{
  byte data[EEPROM_EMULATION_SIZE];
  bool valid;
  uint32_t signature;
} EEPROM_EMULATION;


//////////////////////////////////////////////////////////

#define REGISTERED_NUMBER_FLASH_SECTORS       (*((uint16_t *) FLASH_SIZE_DATA_REGISTER))

#if !defined(USING_FLASH_SECTOR_NUMBER)
  #define USING_FLASH_SECTOR_NUMBER           (REGISTERED_NUMBER_FLASH_SECTORS - 1)
#endif

#define START_FLASH_ADDRESS                   0x8000000

#define START_FLASH_STORAGE_ADDRESS           (START_FLASH_ADDRESS + USING_FLASH_SECTOR_NUMBER * 1024) 

static FlashStorageClass<EEPROM_EMULATION> eeprom_storage( (const void *) (START_FLASH_STORAGE_ADDRESS) );

//////////////////////////////////////////////////////////

class EEPROMClass 
{
  public:
   
    //////////////////////////////////////////////////////////

    EEPROMClass() : _initialized(false), _dirty(false), _commitASAP(true)  
    {
      // Empty
    }

    //////////////////////////////////////////////////////////

    uint8_t read(const int& address)
    {
      if (!_initialized) 
        init();
        
      return _eeprom.data[address];
    }

    //////////////////////////////////////////////////////////

    void update(const int& address, const uint8_t& value)
    {
      if (!_initialized) 
        init();
        
      if (_eeprom.data[address] != value) 
      {
        _dirty = true;
        _eeprom.data[address] = value;
      }
    }

    //////////////////////////////////////////////////////////

    void write(const int& address, const uint8_t& value)
    {
      update(address, value);
    }

    //////////////////////////////////////////////////////////

    void init()
    {
      FLASH_LOGDEBUG0(F("Start Flash Address: 0x")); FLASH_HEXLOGDEBUG0((uint32_t) START_FLASH_STORAGE_ADDRESS);
      FLASH_LOGDEBUG1(F("REGISTERED_NUMBER_FLASH_SECTORS (KB) ="), REGISTERED_NUMBER_FLASH_SECTORS);
      FLASH_LOGDEBUG1(F("USING_FLASH_SECTOR_NUMBER = "), USING_FLASH_SECTOR_NUMBER);

      // Use reference
      eeprom_storage.read(_eeprom);
      
      if (_eeprom.signature != STM32F1_EEPROM_EMULATION_SIGNATURE)
      {
        memset(_eeprom.data, 0xFF, EEPROM_EMULATION_SIZE);
        _eeprom.signature = STM32F1_EEPROM_EMULATION_SIGNATURE;
      }
      
      _eeprom.valid = true;
       
      _initialized = true;
    }

    //////////////////////////////////////////////////////////
     
    /**
     * Read from eeprom cells to an object
    * @param index
    * @param value
    */
    //Functionality to 'get' data to objects to from EEPROM.
    template< typename T > T& get( const int& idx, T &t )
    {       
      // Copy the data from the flash to the buffer if not yet
      if (!_initialized) 
        init();
        
      uint16_t offset = idx;
      uint8_t* _pointer = (uint8_t *) &t;
      
      for ( uint16_t count = sizeof(T) ; count ; --count, ++offset )
      {  
        *_pointer++ = _eeprom.data[offset];
      }
        
      return t;
    }

    //////////////////////////////////////////////////////////

    /**
    * Read from eeprom cells to an object
    * @param index
    * @param value
    */
    //Functionality to 'get' data to objects to from EEPROM.
    template< typename T > const T& put( const int& idx, const T &t )
    {            
      // Copy the data from the flash to the buffer if not yet
      if (!_initialized) 
        init();
      
      uint16_t offset = idx;
         
      const uint8_t* _pointer = (const uint8_t *) &t;
      
      for ( uint16_t count = sizeof(T) ; count ; --count, ++offset )
      {              
        _eeprom.data[offset] = *_pointer++;
      }

      if (_commitASAP)
      {
        _dirty = false;
        _eeprom.valid = true;
        // Save the data from the buffer
        eeprom_storage.write(_eeprom);
      }
      else  
      {
        // Delay saving the data from the buffer to the flash. Just flag and wait for commit() later
        _dirty = true;    
      }
           
      return t;
    }

    //////////////////////////////////////////////////////////

    bool isValid()
    {
      if (!_initialized) 
        init();
        
      return _eeprom.valid;
    }

    //////////////////////////////////////////////////////////

    void commit()
    {
      if (!_initialized) 
        init();
        
      if (_dirty) 
      {
        _dirty = false;
        _eeprom.valid = true;
        // Save the data from the buffer
        eeprom_storage.write(_eeprom);
      }
    }

    //////////////////////////////////////////////////////////
  
    uint16_t length() { return EEPROM_EMULATION_SIZE; }
    
    //////////////////////////////////////////////////////////
    
    void setCommitASAP(bool value = true) { _commitASAP = value; }
    
    //////////////////////////////////////////////////////////
    
    bool getCommitASAP() { return _commitASAP; }

    //////////////////////////////////////////////////////////
    
  private:
    bool _initialized;
    EEPROM_EMULATION _eeprom;
    bool _dirty;
    bool _commitASAP;
};

///////////////////////////////////////////////////////////////////////////////////////

static EEPROMClass EEPROM;
     
#endif    //#ifndef FlashStorage_STM32F1_hpp
