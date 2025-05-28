#include "IMUManager.h"

#define GYRO_ADDR 0x68
#define CAL_ITER 200
#define LSB_SENSITIVITY_GYRO 65.5  // 65.5 LSB/°/s = LSB sensitivy when the range is at 500 °/s
// By default, the value is in degree.
#define ANGLE_INTEGRATION_MULT_DEG 1 / (UPDATE_FREQ * LSB_SENSITIVITY_GYRO)
#define ANGLE_INTEGRATION_MULT_RAD DEG_TO_RAD* ANGLE_INTEGRATION_MULT_DEG

int accAxis[3], gyroAxis[3];
double gyroAxisCal[3];

float gyroDRollLP, gyroDPitchLP, gyroDYawLP;
double gyroDPitch, gyroDRoll, gyroDYaw;
float anglePitch, angleRoll;

long accX, accY, accZ, accTot;
float rollAcc, pitchAcc;


void initIMU() {
  setIMURegisters();
  Serial.print("Calibrating...");
  gyroAxisCal[0] = 0;
  gyroAxisCal[1] = 0;
  gyroAxisCal[2] = 0;

  // Use a buffer to ensure the calibration values are not used while calibrating
  double gyroAxisCalTmp[3];
  gyroAxisCalTmp[0] = 0;
  gyroAxisCalTmp[1] = 0;
  gyroAxisCalTmp[2] = 0;

  for (int calInt = 0; calInt < CAL_ITER; calInt++) {
    readIMU();
    gyroAxisCalTmp[0] += gyroAxis[0];
    gyroAxisCalTmp[1] += gyroAxis[1];
    gyroAxisCalTmp[2] += gyroAxis[2];
    delay(3);  // Give some delay, otherwise the same measurment is used multiple times.
  }

  Serial.println(" - Done");
  gyroAxisCal[0] = gyroAxisCalTmp[0] / CAL_ITER;
  gyroAxisCal[1] = gyroAxisCalTmp[1] / CAL_ITER;
  gyroAxisCal[2] = gyroAxisCalTmp[2] / CAL_ITER;
}


void reinitIMU() {
  //Set the angle to the accelerometer's estimation.
  float rollAcc = getRollAcc();
  float pitchAcc = getPitchAcc();
  anglePitch = pitchAcc;
  angleRoll = rollAcc;
}

void setIMURegister(unsigned int targetRegister, unsigned int value) {
  Wire.beginTransmission(GYRO_ADDR);
  Wire.write(targetRegister);
  Wire.write(value);
  Wire.endTransmission();
  delay(1);
  // Check that the value is set.
  Wire.beginTransmission(GYRO_ADDR);
  Wire.write(targetRegister);
  Wire.endTransmission();

  Wire.requestFrom(GYRO_ADDR, 1);
  waitWire(1, 1, 0, 30, 0);
  if (Wire.read() != value) {
    enterErrorMode(0, 30, 0);  // Green error code (G_reen as in G_yro)
  }
}

void setIMURegisters() {
  // Datasheet : https://www.alldatasheet.com/datasheet-pdf/view/1132807/TDK/MPU-6050.html
  setIMURegister(0x6B, 0x00);  // Select the internal 8 MHz oscillator
  setIMURegister(0x1C, 0x10);  // Set the accelerometer's range to  8g
  setIMURegister(0x1B, 0x08);  // Set the gyro's range to 500 °/s
  setIMURegister(0x1A, 0x03);  // Enable the low pass filter, 44 Hz for acc & 42 for gyro.
}

int getIMUByte() {
  return Wire.read() << 8 | Wire.read();
}

void requestIMUData() {
  Wire.beginTransmission(GYRO_ADDR);
  Wire.write(0x3B);
  Wire.endTransmission();
  Wire.requestFrom(GYRO_ADDR, 14);
}

void readIMUData() {
  accAxis[0] = getIMUByte();
  accAxis[1] = getIMUByte();
  accAxis[2] = getIMUByte();
  int temperature = getIMUByte();
  gyroAxis[0] = getIMUByte();
  gyroAxis[1] = getIMUByte();
  gyroAxis[2] = getIMUByte();

  gyroDRoll = gyroAxis[0] - gyroAxisCal[0];
  gyroDPitch = gyroAxis[1] - gyroAxisCal[1];
  gyroDYaw = gyroAxis[2] - gyroAxisCal[2];

  accX = accAxis[1];
  accY = accAxis[0];
  accZ = accAxis[2];

  // Some correction as the IMU is turned.
  gyroDPitch *= -1;
  gyroDYaw *= -1;

  accX *= -1;
  accZ *= -1;
}

void readIMU() {
  requestIMUData();
  waitWire(14, 1, 0, 30, 0);
  readIMUData();
}

void updateLowPassFilteredVals() {
  // Convert the IMU's output to degrees, and then add them to the previous values to make a low pass filter
  float alpha = 0.65;  // For the low pass filter
  gyroDRollLP = gyroDRollLP * alpha + (gyroDRoll / LSB_SENSITIVITY_GYRO) * (1 - alpha);
  gyroDPitchLP = gyroDPitchLP * alpha + (gyroDPitch / LSB_SENSITIVITY_GYRO) * (1 - alpha);
  gyroDYawLP = gyroDYawLP * alpha + (gyroDYaw / LSB_SENSITIVITY_GYRO) * (1 - alpha);
}

void referentialUpdate() {
  // Tait–Bryan angles -> a change in yaw changes the roll & pitch, but a change in roll only affects the roll, same for the pitch.
  // Change of referential, with an angle change equal to dYaw * dt
  /*
  [ pitch_new ]     [ cos(θ)  -sin(θ) ]   [ pitch ]
  [ roll_new  ]  =  [ sin(θ)   cos(θ) ] * [ roll  ]
  */

  double referentialAngleChangeRad = gyroDYaw * ANGLE_INTEGRATION_MULT_RAD;
  float cosVal = cos(referentialAngleChangeRad);
  float sinVal = sin(referentialAngleChangeRad);

  float anglePitchN = anglePitch * cosVal - angleRoll * sinVal;  //If the IMU has yawed transfer the roll angle to the pitch angle.
  angleRoll = anglePitch * sinVal + cosVal * angleRoll;          //If the IMU has yawed transfer the pitch angle to the roll angle.
  anglePitch = anglePitchN;
}

void updateAccAngleEstimate() {
  accTot = sqrt((accX * accX) + (accY * accY) + (accZ * accZ));

  if (abs(accY) < accTot) {  // To prevent NaN with asin
    pitchAcc = asin((float)accY / accTot) * RAD_TO_DEG;
  }
  if (abs(accX) < accTot) {
    rollAcc = asin((float)accX / accTot) * -1.0 * RAD_TO_DEG;
  }
  //Offset removeal
  pitchAcc -= -5.5;
  rollAcc -= -0.0;
}

void shiftTowardAccestimate() {
  // Sensor fusion to correct the gyro's drift.
  float alpha = 0.0005;
  anglePitch = anglePitch * (1 - alpha) + pitchAcc * alpha;
  angleRoll = angleRoll * (1 - alpha) + rollAcc * alpha;
}

void updateRotation() {
  updateLowPassFilteredVals();
  // Integration of the angular velocity
  anglePitch += gyroDPitch * ANGLE_INTEGRATION_MULT_DEG;
  angleRoll += gyroDRoll * ANGLE_INTEGRATION_MULT_DEG;
  referentialUpdate();
  updateAccAngleEstimate();
  shiftTowardAccestimate();
}

float getAnglePitch() {
  return anglePitch;
}

float getAngleRoll() {
  return angleRoll;
}
float getDRollLP() {
  return gyroDRollLP;
}
float getDPitchLP() {
  return gyroDPitchLP;
}
float getDYawLP() {
  return gyroDYawLP;
}
float getRollAcc() {
  return rollAcc;
}

float getPitchAcc() {
  return rollAcc;
}

void IMUPrintDebug() {
  Serial.println("pitch:" + String(anglePitch));
  Serial.println("roll:" + String(angleRoll));
  Serial.println("pitchAcc: " + String(pitchAcc));
  Serial.println("rollAcc: " + String(rollAcc));
}
