import numpy as np
import cv2
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import onnx
import onnxsim
import cv2
import re

print(cv2.getBuildInformation())

class GesturePredictor(nn.Module):
    def __init__(self):
        super(GesturePredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 3, 128)  # Adjusted based on expected feature map size
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x

# Read the red channel byte data from the file
with open('red_channel.txt', 'r') as file:
    red_values = file.read().strip().split()

# Convert the list of red values to a numpy array
width, height = 80, 45  # Specify the width and height of the image
red_values = np.array([int(value) for value in red_values], dtype=np.uint8)

# Reshape the values to the correct shape for the image (height, width)
red_channel_image = red_values.reshape((height, width))

# Since OpenCV expects an image to have 3 channels, we will create a 3-channel image
# by stacking the red channel into all three channels (red, green, blue)
# to visualize it as an RGB image.

# Create a 3-channel image (just to visualize using OpenCV)
# red_channel_image = np.array(red_channel_image, dtype=np.uint8).reshape((width, height))

rgb_image = np.stack([np.zeros_like(red_channel_image), np.zeros_like(red_channel_image), red_channel_image], axis=-1)
rgb_image = cv2.resize(rgb_image, (0, 0), fx=10, fy=10)
# Display the red channel image using OpenCV
cv2.imshow('Red Channel Visualization', rgb_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

red_channel_image = np.array(red_channel_image, dtype=np.float32)
X = red_channel_image.reshape(-1, 1, 80, 45)

image = torch.tensor(X);

model = GesturePredictor()
model.load_state_dict(torch.load('handrecognition_model2.pth', weights_only=True))
model.eval()

output = model(image)

# quantize
output = output.detach().numpy()
quantized = np.clip(np.round(output / 2), -127, 128)

print(output)
print(quantized)