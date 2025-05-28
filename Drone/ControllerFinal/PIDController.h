#ifndef PID_H
#define PID_H

#include "IMUManager.h"
#include "Arduino.h"
#include "RXManager.h"
#include "Utils.h"


void calculatePid();
void updateESCTarget(FlightStatus status);
void reinitPID();
void PIDPrintDebug();
void calculateTargetAngles(float* angleRoll, float* anglePitch);
void calculateTargetAngVel(float angleRoll, float anglePitch, float* targetDRoll, float* targetDPitch, float* targetDYaw);
float calculatePidOutput(float currentRate, float targetRate, float pGain, float iGain, float dGain,
                         float maxOutput,float maxInt, float* iTerm, float* prevError);
void calculateRollPid(float gyroDRollLp, float targetDRoll);
void calculatePitchPid(float gyroDPitchLp, float targetDPitch);
void calculateYawPid(float gyroDYawLp, float targetDYaw);

float getESC1TargetSpeed();
float getESC2TargetSpeed();
float getESC3TargetSpeed();
float getESC4TargetSpeed();

#endif