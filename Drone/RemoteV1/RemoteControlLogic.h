#ifndef RMT_H
#define RMT_H

#include "Arduino.h"

#define EmergencyButtonPin 9
/*
#define ToggleButtonPin 4
#define ToggleButtonLEDPin 3
*/
#define X_PIN 16
#define Y_PIN 17

#define X2_PIN 14
#define Y2_PIN 15

#define JOYSTICK_BUTTON 4

void initControlLogic();
char lowByteNonZ(uint16_t val);
char highByteNonZ(uint16_t val);
bool isOn();
float rawValToAxis(float rawVal);
int xAxis();
int yAxis();
int xAxis2();
int yAxis2();
String GetControlString(int movementX, int movementY);

#endif