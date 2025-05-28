#include "RemoteControlLogic.h"

void initControlLogic() {
  pinMode(EmergencyButtonPin, INPUT);
  pinMode(JOYSTICK_BUTTON, INPUT);
}

bool isOn() {
  return digitalRead(EmergencyButtonPin);
}

float rawValToAxis(float rawVal) {
  if (rawVal > 400 && rawVal < 460) {
    return 0;
  }
  if (rawVal <= 400) {
    return (rawVal - 400.0);
  }
  return rawVal - 460;
}

int xAxis() {
  return analogRead(X_PIN) + 20;
}

int yAxis() {
  return analogRead(Y_PIN);
  // return 550;
}

int xAxis2() {
  return analogRead(X2_PIN);
}

int yAxis2() {
  return analogRead(Y2_PIN);
}

// Wrap the 10 bits of the ADC in two bythes with a 1 on the left and on the right, 
// to ensure the string isn't terminated.
char lowByteNonZ(uint16_t val) {
  return (char)(((val << 1) & 0xFF) | 0x1);  // Ensure it is never 0
}

char highByteNonZ(uint16_t val) {
  return (char)(((val >> 7) & 0xFF) | 0x80);
}


String GetControlString(int movementX, int movementY) {
  uint16_t x_axis = (uint16_t)(xAxis());
  uint16_t y_axis = (uint16_t)(yAxis());

  uint16_t x_axis2 = (uint16_t)(xAxis2() + movementX);
  uint16_t y_axis2 = (uint16_t)(yAxis2() + movementY);

  Serial.println("Vals : ");
  Serial.print(x_axis);
  Serial.print(" ; ");
  Serial.print(y_axis);
  Serial.print(" ; ");
  Serial.print(x_axis2);
  Serial.print(" ; ");
  Serial.print(y_axis2);
  Serial.println(" ");

  // Create a String from the binary data
  String result;
  result.reserve(9);
  result = "F";  // Add the 'F' character

  // Append each byte from the integers
  result += highByteNonZ(x_axis);
  result += lowByteNonZ(x_axis);

  result += highByteNonZ(y_axis);
  result += lowByteNonZ(y_axis);

  result += highByteNonZ(x_axis2);
  result += lowByteNonZ(x_axis2);

  result += highByteNonZ(y_axis2);
  result += lowByteNonZ(y_axis2);

  return result;
}
