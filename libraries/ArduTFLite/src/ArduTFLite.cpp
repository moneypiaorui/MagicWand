#include "ArduTFLite.h"

tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;


bool modelInit(const unsigned char* model, byte* tensorArena, int tensorArenaSize){
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch!");
    return false;
  }
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
  tflInterpreter->AllocateTensors();
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
  return true;
}

bool modelSetInput(float inputValue, int index){
    if (tflInputTensor == nullptr || index >= tflInputTensor->bytes / sizeof(float)) {
        Serial.println("Input tensor index out of range!");
        return false;
    }
    tflInputTensor->data.f[index] = inputValue;

    return true;
}

bool modelRunInference(){
    TfLiteStatus invokeStatus = tflInterpreter->Invoke();
        if (invokeStatus != kTfLiteOk) {
          return false;
        }
    return true;
}

float modelGetOutput(int index) {
    if (tflOutputTensor == nullptr || index >= tflOutputTensor->bytes / sizeof(float)) {
        Serial.println("Output tensor index out of range!");
        return -1;
    }

    return tflOutputTensor->data.f[index];
}