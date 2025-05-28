import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import re

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

classes = ['01_fist', '02_up', '03_down', '04_left', '05_right', '06_forward', '07_backward', '08_nothing']

IMG_WIDTH, IMG_HEIGHT = 80, 45

class GesturePredictor(nn.Module):
    def __init__(self):
        super(GesturePredictor, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, stride=2, padding=2)  # More filters & stride
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.4)  # Increased dropout

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.4)  # Increased dropout

        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.4)  # Increased dropout

        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)  # Additional layer
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.4)  # Increased dropout

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)  # Additional layer
        self.bn5 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.4)  # Increased dropout

        self.dropout = nn.Dropout(0.4)  # Increased dropout
        self.fc1 = nn.Linear(58880, 256)  # Larger fully connected layers
        self.fc2 = nn.Linear(256, 8)

        self.activation = nn.LeakyReLU(0.1)  # Improved activation

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x_res = x  # Residual connection
        x = self.activation(self.bn3(self.conv3(x)))
        x += x_res  # Residual connection
        x = self.activation(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x

model = GesturePredictor()
model.load_state_dict(torch.load('handrecognition_model3.pth', weights_only=True))
model.eval()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    red_channel = frame[:, :, 2]  # Extract red channel

    red_channel = cv2.resize(red_channel, (80, 45))
    red_channel = np.array(red_channel, dtype=np.float32)
    
    rgb_image = np.stack([red_channel, red_channel, red_channel], axis=-1)
    rgb_image = cv2.resize(rgb_image, (0, 0), fx=12, fy=12)
    
    # model
    red_channel = red_channel.reshape(-1, 1, IMG_WIDTH, IMG_HEIGHT)
    output = model(torch.tensor(red_channel))
    _, predicted = torch.max(output, 1)
    # print(_)

    cv2.putText(rgb_image, f'Gesture: {classes[predicted]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Capture', rgb_image.astype(np.uint8))
    
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        break


cap.release()
cv2.destroyAllWindows()