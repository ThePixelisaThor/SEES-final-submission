#include "Radio.h"

void initLora()
{
  if (!LoRa.begin(433E6)) {  // or 915E6
    // Serial.println("Starting LoRa failed!");
    // while (1)
      // ;
  }
  LoRa.setSyncWord(0xF3);
  // LoRa.enableCrc();
  LoRa.setSignalBandwidth(500E3);
  LoRa.setSpreadingFactor(8); // In theory between 6 and 12, but for some reason can't work at 6. Higher means slower but steadier
}

void WaitForPckt(unsigned long timeStart) {
  while (LoRa.parsePacket() == 0 && timeStart > millis()) {  // Wait till a packet or time runs out
    delay(2);
  }
}

bool hasCorrectChecksum(char *res, int resLength) {
  char *nMsg = unpackPacket(res, resLength);
  bool result = (nMsg != NULL);
  free(nMsg);
  return result;
}
void ReadRadio(int maxDelay, char *result, int *resLength, int delay2) {
  *resLength = 0;
  unsigned long timeStart = millis() + maxDelay;
  unsigned long timeToPrint = millis();
  WaitForPckt(timeStart);
 

  while (LoRa.available() || timeStart > millis()) {
    if (LoRa.available()) {
      int inChar = LoRa.read();
      result[*resLength] = (char)inChar;
      (*resLength)++;
      timeStart = millis() + delay2;
      if (hasCorrectChecksum(result, *resLength)) {
        break;
      }
    } else {
      delay(1);
    }
  }
  //PrintRadioData(result, resLength);  // Debug fct
}

unsigned short calculateChecksum(char *data, int length) {
  unsigned short checksum = 0;
  for (int i = 0; i < length; i++) {
    checksum += (unsigned char)data[i];
  }
  return checksum & 0xFFFF;  // Ensure it's a 2-byte checksum
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
  // Create packet with data and checksum separated by '|'
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
  //Serial.println("sending : " + makePacket(toSend) + "...");
  LoRa.beginPacket();
  LoRa.print(makePacket(toSend));
  LoRa.endPacket();  // true = async / non-blocking mode
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
    //Serial.println("Checksum mismatch!");
    free(unpacked);
    return NULL;
  }
  // Serial.printnl("Checkum matches!");
  return unpacked;
}
