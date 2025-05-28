#ifndef IMU_H
#define IMU_H

#include "Arduino.h"
#include "Utils.h"
#include <Wire.h>

void initIMU();
void setIMURegisters();
void readIMU();
void updateRotation();
void IMUPrintDebug();
void reinitIMU();

float getAnglePitch();
float getAngleRoll();
float getAngleRoll();
float getDRollLP();
float getDPitchLP();
float getDYawLP();
float getRollAcc();
float getPitchAcc();
void setIMURegister(unsigned int targetRegister, unsigned int value);
int getIMUByte();
void requestIMUData();
void readIMUData();
void updateLowPassFilteredVals();
void referentialUpdate();
void updateAccAngleEstimate();
void shiftTowardAccestimate();


#endif