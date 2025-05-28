#include "PIDController.h"
// Ziegler Nichols

#define K_U_ROLL 500   // Ultimate gain for roll (fixed duplicate definition)
#define T_U_ROLL 0.5   // Ultimate period for roll (seconds)
#define K_U_PITCH 450  // Ultimate gain for pitch
#define T_U_PITCH 0.7  // Ultimate period for pitch (seconds)

// Ziegler-Nichols PID tuning coefficients (classic method)
// https://en.wikipedia.org/wiki/Ziegler-Nichols_method
#define ZN_KP_COEFF 0.6    // Kp = 0.6 * Ku
#define ZN_KI_COEFF 1.2    // Ki = 1.2 * Ku / Tu
#define ZN_KD_COEFF 0.075  // Kd = 0.075 * Ku * Tu

// Roll PID gains using Ziegler-Nichols method
float pGainRoll = ZN_KP_COEFF * K_U_ROLL;
float iGainRoll = (ZN_KI_COEFF * K_U_ROLL) / T_U_ROLL;
float dGainRoll = ZN_KD_COEFF * K_U_ROLL * T_U_ROLL;
float maxOutputRoll = 400;
float maxIntTermRoll = 400;

// Pitch PID gains using Ziegler-Nichols method
float pGainPitch = ZN_KP_COEFF * K_U_PITCH;
float iGainPitch = (ZN_KI_COEFF * K_U_PITCH) / T_U_PITCH;
float dGainPitch = ZN_KD_COEFF * K_U_PITCH * T_U_PITCH;
float maxOutputPitch = 400;
float maxIntTermPitch = 400;

float pGainYaw = 4.0;
float iGainYaw = 0.0;
float dGainYaw = 0.0;
float maxOutputYaw = 400;
float maxIntTermYaw = 400;

float iTermRoll, targetDRoll, outputRoll, prevErrorDRoll;
float iTermPitch, targetDPitch, outputPitch, prevErrorDPitch;
float iTermYaw, targetDYaw, outputYaw, prevErrorDYaw;

float targetEsc1, targetEsc2, targetEsc3, targetEsc4;

void calculateTargetAngles(float* angleRoll, float* anglePitch) {
  float joystickSensitivityDivider = 50;
  float xTarget = getLORAY2() / joystickSensitivityDivider / 1.4;
  float yTarget = getLORAX2() / joystickSensitivityDivider * 1.4;
  *angleRoll -= xTarget;
  *anglePitch -= yTarget;
}

float calculatePidOutput(float currentRate, float targetRate, float pGain, float iGain, float dGain,
                         float maxOutput, float maxInt, float* iTerm, float* prevError) {
  float pidErrorTemp = currentRate - targetRate;
  *iTerm += iGain * pidErrorTemp;

  if (abs(*iTerm) > maxInt) {
    *iTerm = (*iTerm > 0) ? maxOutput : -maxOutput;
  }

  float output = pGain * pidErrorTemp + *iTerm + dGain * (pidErrorTemp - *prevError);

  if (abs(output) > maxOutput) {
    output = (output > 0) ? maxOutput : -maxOutput;
  }
  *prevError = pidErrorTemp;
  return output;
}

void calculateRollPid(float gyroDRollLp, float targetDRoll) {
  outputRoll = calculatePidOutput(gyroDRollLp, targetDRoll, pGainRoll, iGainRoll, dGainRoll,
                                  maxOutputRoll, maxIntTermRoll, &iTermRoll, &prevErrorDRoll);
}

void calculatePitchPid(float gyroDPitchLp, float targetDPitch) {
  outputPitch = calculatePidOutput(gyroDPitchLp, targetDPitch, pGainPitch, iGainPitch, dGainPitch,
                                   maxOutputPitch, maxIntTermPitch, &iTermPitch, &prevErrorDPitch);

  if (outputPitch > maxOutputPitch) outputPitch = maxOutputPitch;
  else if (outputPitch < maxOutputPitch * -1) outputPitch = maxOutputPitch * -1;
}

void calculateYawPid(float gyroDYawLp, float targetDYaw) {
  outputYaw = calculatePidOutput(gyroDYawLp, targetDYaw, pGainYaw, iGainYaw, dGainYaw,
                                 maxOutputYaw, maxIntTermYaw, &iTermYaw, &prevErrorDYaw);
}


void calculateTargetAngVel(float angleRoll, float anglePitch, float* targetDRoll, float* targetDPitch, float* targetDYaw) {
  *targetDRoll = -5 * angleRoll;
  *targetDPitch = -5 * anglePitch;
  *targetDYaw = 0;  // No need to turn back towards 0, simply ensure the quadcopter isn't turning around the yaw.
}

void calculatePid() {
  float anglePitch = getAnglePitch();
  float angleRoll = getAngleRoll();
  float gyroDRollLp = getDRollLP();
  float gyroDPitchLp = getDPitchLP();
  float gyroDYawLp = getDYawLP();

  calculateTargetAngles(&angleRoll, &anglePitch);
  float targetDRoll, targetDPitch, targetDYaw;
  calculateTargetAngVel(angleRoll, anglePitch, &targetDRoll, &targetDPitch, &targetDYaw);

  calculateRollPid(gyroDRollLp, targetDRoll);
  calculatePitchPid(gyroDPitchLp, targetDPitch);
  calculateYawPid(gyroDYawLp, targetDYaw);
}

void PIDPrintDebug() {
  Serial.println("targetEsc1:" + String(targetEsc1));
  Serial.println("targetEsc2:" + String(targetEsc2));
  Serial.println("targetEsc3:" + String(targetEsc3));
  Serial.println("targetEsc4:" + String(targetEsc4));
}

void reinitPID() {
  iTermRoll = 0;
  iTermYaw = 0;
  iTermPitch = 0;
  prevErrorDRoll = 0;
  prevErrorDPitch = 0;
  prevErrorDYaw = 0;
}

void updateESCTarget(FlightStatus status) {
  float throttleLORAInput = getLORAThrottle();
  float throttle = throttleLORAInput * 2.5 + 500;

  float JoystickSensitivityDivider = 50;

  float xTarget = getLORAY2() / JoystickSensitivityDivider / 1.0;
  float yTarget = getLORAX2() / JoystickSensitivityDivider * 1.4;
  float absSum = abs(xTarget) + abs(yTarget);

  if (status == FlightStatus::ON) {
    if (throttle > 2800) throttle = 2800;  // Cap throttle
    targetEsc1 = throttle - outputPitch + outputRoll - outputYaw;
    targetEsc2 = throttle + outputPitch + outputRoll + outputYaw;
    targetEsc3 = throttle + outputPitch - outputRoll - outputYaw;
    targetEsc4 = throttle - outputPitch - outputRoll + outputYaw;
  } else {
    targetEsc1 = 0;
    targetEsc2 = 0;
    targetEsc3 = 0;
    targetEsc4 = 0;
  }
}

float getESC1TargetSpeed() {
  return targetEsc1;
}

float getESC2TargetSpeed() {
  return targetEsc2;
}

float getESC3TargetSpeed() {
  return targetEsc3;
}

float getESC4TargetSpeed() {
  return targetEsc4;
}