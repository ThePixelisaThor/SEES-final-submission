#ifndef UTILS_H
#define UTILS_H
#define MAX_MSG_SIZE 400
#include "Radio.h"

void initSerial();
String readSerial();

void prinIncomingPckt();
void printRadioData(char *result, int *resLength);
void sendSerialMsg();
void printCharArray(char *ptr, int msgLength);
int indexInCharArray(char *array, char targt);
int indexInCharArrayFromEnd(char *ptr, char toSearch, int length);

#endif