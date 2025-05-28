#include "Radio.h"

void initLora() {
  if (!LoRa.begin(433E6)) {
    Serial.println("Starting LoRa failed!");
    while (1)
      ;
  }
  LoRa.setSyncWord(0xF3);
  LoRa.setSignalBandwidth(500E3);
  LoRa.setSpreadingFactor(8);
}

void waitForPckt(unsigned long timeStart) {
  while (LoRa.parsePacket() == 0 && timeStart > millis()) {  // Wait till a packet or time runs out
    delay(1);
  }
}

bool hasCorrectChecksum(char *res, int resLength) {
  char *nMsg = unpackPacket(res, resLength);
  bool result = (nMsg != NULL);
  free(nMsg);
  return result;
}

void readRadio(int maxDelay, char *result, int *resLength, int delay2) {
  *resLength = 0;
  unsigned long timeStart = millis() + maxDelay;
  unsigned long timeToPrint = millis();
  waitForPckt(timeStart);

  while ((LoRa.available() || timeStart > millis()) && *resLength < 20) {
    if (LoRa.available()) {
      result[*resLength] = (char)LoRa.read();
      (*resLength)++;
      timeStart = millis() + delay2;
      if (hasCorrectChecksum(result, *resLength)) {
        break;
      }
    } else {
      delay(1);
    }
  }
}


void PrintRadioData(char *result, int *resLength)  // A debug function, to delete
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
  } else {
    Serial.print("n");
  }
}

unsigned short calculateChecksum(char *data, int length) {
  unsigned short checksum = 0;
  for (int i = 0; i < length; i++) {
    checksum += (unsigned char)data[i];
  }
  return checksum & 0xFFFF; 
}

char *extractPayload(char *pckt, int delimPos) {
  char *unpacked = (char *)malloc((delimPos + 1) * sizeof(char));
  for (int i = 0; i < delimPos; i++) {
    unpacked[i] = pckt[i];
  }
  unpacked[delimPos] = '\0';
  return unpacked;
}
String makePacket(String data) {
  // Calculate checksum (2-byte)
  unsigned short checksum = 0;
  for (int i = 0; i < data.length(); i++) {
    checksum += (unsigned char)data[i];
  }
  checksum = checksum & 0xFFFF;  // Ensure it's a 2-byte checksum
  String packet = data + "|";
  packet += (char)((checksum >> 8) & 0xFF);  // High byte
  packet += (char)(checksum & 0xFF);         // Low byte
  return packet;
}

void sendDataByRadio(String toSend) {
  while (LoRa.beginPacket() == 0) {
    Serial.print("waiting for radio ... ");
    delay(10);
  }
  
  LoRa.beginPacket();
  LoRa.print(makePacket(toSend));
  LoRa.endPacket();  // Can use true as argument for async / non-blocking mode
}



char *unpackPacket(char *pckt, int length) {
  int delimPos = length - 3;
  if (pckt[delimPos] != '|')
    return NULL;

  char *unpacked = extractPayload(pckt, delimPos);

  unpacked[delimPos] = '\0';
  char highByte = pckt[delimPos + 1];
  char lowByte = pckt[delimPos + 2];
  unsigned short receivedChecksum = (highByte << 8) | (lowByte & 0xFF);
  unsigned short calculatedChecksum = calculateChecksum(pckt, delimPos);
  if (calculatedChecksum != receivedChecksum) {
    free(unpacked);
    return NULL;
  }
  
  return unpacked;
}
