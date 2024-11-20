#ifndef ARDUTFLITE_H
#define ARDUTFLITE_H

#include <Arduino.h>
#include <Chirale_TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

bool modelInit(const unsigned char* model, byte* tensorArena, int tensorArenaSize);
bool modelSetInput(float inputValue, int index);
bool modelRunInference();
float modelGetOutput(int index);

#endif
