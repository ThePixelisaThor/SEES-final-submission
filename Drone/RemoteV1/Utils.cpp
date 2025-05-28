#include "Utils.h"

String inString = "";  // string to hold input
int val = 0;
unsigned long timeLastChar;
void initSerial() {
  Serial.begin(115200);
  timeLastChar = 0;
  while (!Serial)
    ;
}

String readSerial() {
  String receivedData = "";
  timeLastChar = millis() - 4;
  while (millis() - timeLastChar < 5) {  // Check if there is data available to read
    if (Serial.available() == 0) {
      return "";
    }
    char c = Serial.read();  // Read a character from the serial buffer
    if (c == '\n') {         // If newline character is encountered, break the loop
      break;
    }
    receivedData += c;       // Append the character to receivedData
    delayMicroseconds(100);  // Optional delay
    timeLastChar = millis();
  }
  return receivedData;
}

void prinIncomingPckt() {
  int resLength;
  char *res;
  res = (char *)malloc(MAX_MSG_SIZE * sizeof(char));
  ReadRadio(10, res, &resLength, 100);
  if (resLength == 0) {
    free(res);
    return;
  }
  Serial.println("Raw packet : " + String(res));
  char *ans = unpackPacket(res, resLength);
  res[resLength] = '\0';
  if (resLength > 0) {
    Serial.println("Unpacked : " + String(ans));
  }
  free(res);
  free(ans);
  Serial.print("' with RSSI ");
  Serial.println(LoRa.packetRssi());
}

void sendSerialMsg() {
  String receivedData = readSerial();

  if (receivedData != "") {
    Serial.println("Sending : " + receivedData);
    sendDataByRadio(receivedData);
  }
}


void printCharArray(char *ptr, int msgLength) {
  for (int i = 0; i < min(MAX_MSG_SIZE, msgLength); i++) {
    if (ptr[i] == '\0') {
      Serial.println("Found \\0 at i = " + String(i) + ".");
      break;
    }
    Serial.print(ptr[i]);
  }
  Serial.println("");
}

int indexInCharArray(char *array, char targt) {
  int result = -1;
  for (int i = 0; i < MAX_MSG_SIZE; i++) {
    if (array[i] == '\0')
      break;
    if (array[i] == targt) {
      result = i;
      break;
    }
  }
  return result;
}



int indexInCharArrayFromEnd(char *ptr, char toSearch, int length) {
  int result = -1;
  for (int i = length - 1; i >= 0; i--) {
    /*if (array[i] == '\0')
      break;*/
    if (ptr[i] == toSearch) {
      result = i;
      break;
    }
  }
  return result;
}



void printRadioData(char *result, int *resLength)  // A debug function, to delete
{
  if (*resLength != 0) {
    Serial.println("--------Raw data from radio : ");
    for (int i = 0; i < *resLength; i++) {
      Serial.print(result[i]);
    }
    Serial.println("");
    for (int i = 0; i < *resLength; i++) {
      unsigned short val = (unsigned char)result[i];
      Serial.print(val);
      Serial.print("#");
    }
    Serial.println("--------- length : " + String(*resLength));
  }
}
