/******************************************************************************************************************************************
  FlashStorage_RTL8720.h
  For Realtek RTL8720DN, etc. using Flash

  The FlashStorage_RTL8720DN library aims to provide a convenient way to store and retrieve user's data using the non-volatile flash memory
  of RTL8720DN, etc.. It's using the buffered read and write to minimize the access to Flash.
  It now supports writing and reading the whole object, not just byte-and-byte.

  Inspired by Cristian Maglie's FlashStorage (https://github.com/cmaglie/FlashStorage)

  Built by Khoi Hoang https://github.com/khoih-prog/FlashStorage_RTL8720DN
  Licensed under MIT license
  Version: 1.1.0

  Version Modified By   Date        Comments
  ------- -----------  ----------   -----------
  1.0.0   K Hoang      06/08/2021  Initial coding to support RTL8720DN using emulated-FlashStorage
  1.1.0   K Hoang      25/01/2022  Fix `multiple-definitions` linker error
  ******************************************************************************************************************************************/

#pragma once

#ifndef FlashStorage_RTL8720_h
#define FlashStorage_RTL8720_h

#if !defined(CONFIG_PLATFORM_8721D)
  #error Only for Ameba Realtek RTL8720DN, RTL8722DM and RTM8722CSM platform.
#endif

#define LATEST_RTL8720_FIRMWARE           "1.0.0"

#ifndef FLASH_STORAGE_RTL8720_VERSION
  #define FLASH_STORAGE_RTL8720_VERSION             "FlashStorage_RTL8720 v1.1.0"

  #define FLASH_STORAGE_RTL8720_VERSION_MAJOR       1
  #define FLASH_STORAGE_RTL8720_VERSION_MINOR       1
  #define FLASH_STORAGE_RTL8720_VERSION_PATCH       0

#define FLASH_STORAGE_RTL8720_VERSION_INT           1001000

#endif

#define FLASH_MEMORY_APP_BASE             0x00100000

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
#define FLASH_LOGERROR1(x,y)      if(FLASH_DEBUG>0) { FLASH_PRINT("[FLASH] "); FLASH_PRINT(x); FLASH_PRINT_SP; FLASH_PRINTLN(y); }
#define FLASH_LOGERROR2(x,y,z)    if(FLASH_DEBUG>0) { FLASH_PRINT("[FLASH] "); FLASH_PRINT(x); FLASH_PRINT_SP; FLASH_PRINT(y); FLASH_PRINT_SP; FLASH_PRINTLN(z); }
#define FLASH_LOGERROR3(x,y,z,w)  if(FLASH_DEBUG>0) { FLASH_PRINT("[FLASH] "); FLASH_PRINT(x); FLASH_PRINT_SP; FLASH_PRINT(y); FLASH_PRINT_SP; FLASH_PRINT(z); FLASH_PRINT_SP; FLASH_PRINTLN(w); }

/////////////////////////////////////////////////////

#define FLASH_LOGDEBUG(x)         if(FLASH_DEBUG>1) { FLASH_PRINT("[FLASH] "); FLASH_PRINTLN(x); }
#define FLASH_LOGDEBUG0(x)        if(FLASH_DEBUG>1) { FLASH_PRINT(x); }
#define FLASH_LOGDEBUG1(x,y)      if(FLASH_DEBUG>1) { FLASH_PRINT("[FLASH] "); FLASH_PRINT(x); FLASH_PRINT_SP; FLASH_PRINTLN(y); }
#define FLASH_LOGDEBUG2(x,y,z)    if(FLASH_DEBUG>1) { FLASH_PRINT("[FLASH] "); FLASH_PRINT(x); FLASH_PRINT_SP; FLASH_PRINT(y); FLASH_PRINT_SP; FLASH_PRINTLN(z); }
#define FLASH_LOGDEBUG3(x,y,z,w)  if(FLASH_DEBUG>1) { FLASH_PRINT("[FLASH] "); FLASH_PRINT(x); FLASH_PRINT_SP; FLASH_PRINT(y); FLASH_PRINT_SP; FLASH_PRINT(z); FLASH_PRINT_SP; FLASH_PRINTLN(w); }

//////////////////////////////////////////


#ifdef __cplusplus
extern "C" 
{
#endif

#include "flash_api.h"
#include "cmsis_os.h"

#ifdef __cplusplus
}
#endif

#define BOARD_TYPE      "Rtlduino RTL8720DN"

#ifndef BOARD_NAME
  #if defined(ARDUINO_BOARD)
    #define BOARD_NAME    ARDUINO_BOARD
  #elif defined(BOARD_TYPE)
    #define BOARD_NAME    BOARD_TYPE
  #else
    #define BOARD_NAME    "Unknown Board"
  #endif  
#endif

static flash_t flash_obj;

class FlashStorageClass_RTL8720
{
  public:
  
    FlashStorageClass_RTL8720() : _initialized(false), _dirtyBuffer(false), _commitASAP(true) , _validFlashStorage(true) 
    {
      buf = (unsigned char *) malloc ( FLASH_SECTOR_SIZE );
      memset(buf, 0, sizeof(buf));
    }

    ~FlashStorageClass_RTL8720() 
    {
      if (buf != NULL) 
      {
        free(buf);
        buf = NULL;
      }
    }
   
    /////////////////////////////////////////////////////
 
    // Read whole sector from Flash to local buf
    void readFlashToBuffer() 
    {       
      if (buf)
      {
        flash_stream_read(_pflash_obj, _base_address, FLASH_SECTOR_SIZE, buf);
      }
    }
    
    /////////////////////////////////////////////////////
    
    // Write whole sector from local buf to Flash
    // Return true if commit OK
    bool commit() 
    {
      if (!_initialized)
        init();
        
      if (buf)
      {       
        unsigned int sector;
      
        sector = ((_base_address) / FLASH_SECTOR_SIZE ) * FLASH_SECTOR_SIZE;;    //((_base_address + offset) / FLASH_SECTOR_SIZE ) * FLASH_SECTOR_SIZE;
        flash_erase_sector(_pflash_obj, sector);

        flash_stream_write(_pflash_obj, sector, FLASH_SECTOR_SIZE, buf);
        
        _dirtyBuffer = false;
        _validFlashStorage = true;
      }
      
      return !(_dirtyBuffer);
    }
    
    /////////////////////////////////////////////////////

    /**
     * Read a uint8_t byte at FlashStorage address
     * @param index
     * @return value
     */
    uint8_t readByte(const uint32_t& offset) 
    {     
      if (!_initialized)
        init();
      
      return * (uint8_t*) (buf + offset);
    }
    
    /////////////////////////////////////////////////////
        
    void writeByte(const uint32_t& offset, const uint8_t& data) 
    {     
      if (!_initialized) 
        init();
     
      * (uint8_t*) (buf + offset) = data;
      
      if (_commitASAP)
      {
        // Save the data from the buffer to the flash right away
        commit();
        
        _dirtyBuffer = false;
        _validFlashStorage = true;
      }
      else  
      {
        // Delay saving the data from the buffer to the flash. Just flag and wait for commit() later
        _dirtyBuffer = true;    
      }      
    }
    
    /////////////////////////////////////////////////////
    
    /**
     * Update a uint8_t byte at FlashStorage address
     * @param index
     * @param value
     */
    void updateByte(const uint32_t& offset, const uint8_t& value)
    {
      if (!_initialized) 
        init();
        
      if (readByte(offset) != value)
      {
        _dirtyBuffer = true;
        writeByte(offset, value);
      }
    }
         
    /////////////////////////////////////////////////////

    /**
     * Read a uint32_t word at FlashStorage address
     * @param index
     * @return value
     */
    uint32_t readWord(const uint32_t& offset) 
    {     
      if (!_initialized)
        init();
      
      return * (uint32_t*) (buf + offset);
    }
    
    /////////////////////////////////////////////////////
        
    void writeWord(const uint32_t& offset, const uint32_t& data) 
    {     
      if (!_initialized) 
        init();
     
      * (uint32_t*) (buf + offset) = data;
      
      if (_commitASAP)
      {
        // Save the data from the buffer to the flash right away
        commit();
        
        _dirtyBuffer = false;
        _validFlashStorage = true;
      }
      else  
      {
        // Delay saving the data from the buffer to the flash. Just flag and wait for commit() later
        _dirtyBuffer = true;    
      }      
    }
    
    /////////////////////////////////////////////////////
    
    /**
     * Update a uint8_t byte at FlashStorage address
     * @param index
     * @param value
     */
    void updateWord(const uint32_t& offset, const uint32_t& value)
    {
      if (!_initialized) 
        init();
        
      if (readWord(offset) != value)
      {
        _dirtyBuffer = true;
        writeWord(offset, value);
      }
    }
   
    /////////////////////////////////////////////////////
   
    /**
     * Update FlashStorage cells from an object
     * @param index
     * @param value
     */
    //Functionality to 'get' and 'put' objects to and from FlashStorage.
    template< typename T > T &get( const uint32_t& offset, T &t )
    {       
      // Copy the data from the flash to the buffer if not yet
      if (!_initialized)
        init();
                   
      memcpy((uint8_t *) &t, (uint8_t *) (buf + offset), sizeof(T));
      
      return t;
    }
    
    /////////////////////////////////////////////////////
    
    template< typename T > const T &put( const uint32_t& offset, const T &t )
    {            
      // Copy the data from the flash to the buffer if not yet
      if (!_initialized) 
        init();
        
      memcpy((uint8_t *) (buf + offset), (uint8_t *) &t, sizeof(T));
      
      if (_commitASAP)
      {
        // Save the data from the buffer to the flash right away
        commit();
        
        _dirtyBuffer = false;
        _validFlashStorage = true;
      }
      else  
      {
        // Delay saving the data from the buffer to the flash. Just flag and wait for commit() later
        _dirtyBuffer = true;    
      }
                        
      return t;
    }
    
    /////////////////////////////////////////////////////

    /**
     * Check whether the FlashStorage data is valid
     * @return true, if FlashStorage data is valid (has been written at least once), false if not
     */
    bool isValid()
    {         
      return _initialized;
    }
    
    /////////////////////////////////////////////////////

    uint16_t length() 
    { 
      return _flashSize; 
    }

    /////////////////////////////////////////////////////
    
    void setCommitASAP(bool value = true) 
    { 
      _commitASAP = value; 
    }
    
    /////////////////////////////////////////////////////
    
    bool getCommitASAP() 
    { 
      return _commitASAP; 
    }
    
    void eraseFlashSector()
    {
      // All data to 0xFFFFFFFF
      
      flash_erase_sector(_pflash_obj, _base_address);
    }
    
    /////////////////////////////////////////////////////
    
  private:
  
    void init(uint32_t flashSize = FLASH_SECTOR_SIZE)
    {
      if (buf == NULL) 
      {
        FLASH_LOGDEBUG("init() error, buf = NULL");
        
        return;
      }
            
      FLASH_LOGDEBUG1("init(), flashSize =", _flashSize);
      
      _flashSize = flashSize;
            
      readFlashToBuffer();
      
      _initialized = true;
    }
    
    /////////////////////////////////////////////////////

    flash_t*  _pflash_obj   = &flash_obj;
    uint32_t  _flashSize    = FLASH_SECTOR_SIZE;          // sector size = 4KB = 0x1000
    uint32_t  _base_address = FLASH_MEMORY_APP_BASE;      // 0x00100000
    
    bool      _initialized;     
    bool      _dirtyBuffer;
    bool      _commitASAP;
    bool      _validFlashStorage;
    
    /**
     * @brief The buf size. (It can be regarded as work size)
     */
    //uint32_t buf_size;

    /**
     * @brief The buf to be operated.
     * @note Modify buf won't change the content of buf. It needs update to write back to flash memory.
     */
    unsigned char *buf;
};

static FlashStorageClass_RTL8720 FlashStorage;


#endif    //#ifndef FlashStorage_RTL8720_h
