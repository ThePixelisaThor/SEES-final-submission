import cv2
import numpy as np
import os
import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

augmentations = A.Compose([
    # A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0.05, p=0.1)
])


class GesturePredictor(nn.Module):
    def __init__(self):
        super(GesturePredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(576, 128)  # Adjusted output size after convolutions
        self.dropout = nn.Dropout(0.20)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def add_noise(image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB);
    image = augmentations(image=image)["image"]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);
    return image

def get_next_filename(gesture_dir):
    existing_files = [int(f.split('.')[0]) for f in os.listdir(gesture_dir) if f.endswith('.png')]
    return max(existing_files, default=-1) + 1

# def capture_gestures(output_dir='my_dataset', gestures=['01_fist', '02_up', '03_down', '04_left', '05_right', '06_forward', '07_backward'], samples_per_gesture=200):
def capture_gestures(output_dir='my_dataset', gestures=['04_left', '05_right', '06_forward', '07_backward', '08_nothing'], samples_per_gesture=800):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
    
    model = GesturePredictor()
    model.load_state_dict(torch.load('handrecognition_model3.pth', weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    classes = ['01_fist', '02_up', '03_down', '04_left', '05_right', '06_forward', '07_backward', '08_nothing']

    for gesture in gestures:
        gesture_dir = os.path.join(output_dir, gesture)
        os.makedirs(gesture_dir, exist_ok=True)
        print(f'Capturing {gesture}... Press SPACE to start, ESC to exit.')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            red_channel = frame[:, :, 2]  # Extract red channel
            red_channel = cv2.resize(red_channel, (80, 45))
            red_channel = np.array(red_channel, dtype=np.float32)
            rgb_image = np.stack([red_channel, red_channel, red_channel], axis=-1)
            rgb_image = cv2.resize(rgb_image, (0, 0), fx=12, fy=12)
            cv2.putText(rgb_image, f'Gesture asked: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # model
            img = red_channel.astype("uint8")

            cv2.imshow('Capture', img.astype("uint8"))

            start_x = (80 - 45) // 2
            img = img[:, start_x:start_x + 45]
            



            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Apply Otsu's Thresholding
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a mask
            mask = np.zeros_like(img)

            # Draw the largest contour (assuming the hand is the largest object)
            if True:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

            # Apply the mask
            segmented = cv2.bitwise_and(binary, mask)

            final = segmented.astype("float32") / 255.0

            _, binary_image = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            final = binary_image.astype("float32") / 255.0

            output = model(torch.tensor(np.reshape(final, (-1, 1, 45, 45))))
            _, predicted = torch.max(output, 1)
            # print(_)

            # cv2.putText(rgb_image, f'Gesture predicted: {classes[predicted]}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            final = cv2.resize(final, (0, 0), fx=12, fy=12) * 255.0
            final = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)

            cv2.putText(final, f'Gesture predicted: {classes[predicted]}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Capture', final.astype("uint8"))
            # cv2.imshow('Capture', rgb_image.astype(np.uint8))
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == 32:  # Space to start capturing
                break

        next_index = get_next_filename(gesture_dir)

        for i in range(samples_per_gesture):
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                red_channel = frame[:, :, 2]  # Extract red channel
                red_channel = cv2.resize(red_channel, (80, 45))
                noisy = add_noise(red_channel)  # Add noise
                
                red_channel = np.array(red_channel, dtype=np.float32)
                red_channel = red_channel.reshape(-1, 1, 80, 45)
                output = model(torch.tensor(red_channel))
                _, predicted = torch.max(output, 1)
                if classes[predicted] != gesture:
                    break

            filename = os.path.join(gesture_dir, f'{next_index + i}.png')
            cv2.imwrite(filename, noisy)
            print(f'Saved {filename}')
            
            noisy = cv2.resize(noisy, (0, 0), fx=12, fy=12)
            cv2.imshow('Capture', cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR))
            cv2.waitKey(25)  # Brief pause to let user adjust hand position
    
    cap.release()
    cv2.destroyAllWindows()
    print('Dataset collection complete!')

if __name__ == "__main__":
    capture_gestures()
