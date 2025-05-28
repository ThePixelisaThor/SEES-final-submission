# Code used to quantize a PyTorch ML model into ESPDL (proprietary format of espressif)
# This code is based on the esp-dl examples

from torch.utils.data import DataLoader, TensorDataset
import torch
from ppq.api import espdl_quantize_torch
from ppq.executor.torch import TorchExecutor
import os
import cv2
import numpy as np
import re
import torch.nn as nn
import torch.nn.functional as F

from typing import Iterable, List, Tuple
from ppq import QuantizationSettingFactory, QuantizationSetting
from ppq.api import espdl_quantize_torch, get_target_platform

IMG_WIDTH, IMG_HEIGHT = 45, 45


def collate_fn(batch):
    batch = batch[0].to(DEVICE)
    return batch


def generate_data():
    path = "D:\\Onedrive\hdd portable\\vaulthor\VaulThor\\2 Projects\\thesis\\esp_dl_1\\esp-dl\\examples\\tutorial\\how_to_quantize_model\\quantize_my_model"
    data = []
    labels = []

    # Load and preprocess images
    base_path = os.path.join(path, 'my_dataset_blackwhite')
    for folder in sorted(os.listdir(base_path)):
        print(folder)
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
                if img is not None:
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    data.append(img)
                    match = re.match(r'(\d{2})_', folder) # The file format is specific
                    label = int(match.group(1))
                    labels.append(label - 1)

    # Convert to numpy arrays
    X = np.array(data, dtype=np.float32)  # Normalize pixel values
    Y = np.array(labels, dtype=np.int64)  # Labels should be integers

    # Reshape X to fit model input (Add channel dimension)
    X = X.reshape(-1, 1, IMG_WIDTH, IMG_HEIGHT) / 255.0

    return X, Y

# Definition of the model
class GesturePredictor(nn.Module):
    def __init__(self):
        super(GesturePredictor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)
        self.fc1 = nn.Linear(576, 128)
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


if __name__ == "__main__":

    # Quantization for ESP
    quant_setting = QuantizationSettingFactory.espdl_setting()

    ESPDL_MODEL_PATH = "handrecognition_model4.espdl"
    INPUT_SHAPE = [1, 1, 45, 45]
    TARGET = "esp32s3"
    NUM_OF_BITS = 16  # 8 bits gave results that were not precise enough
    DEVICE = "cpu"

    # Get all of the dataset, used for calibration
    x_test, y_test = generate_data()

    dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model and load weights from previous training
    model = GesturePredictor()
    model.load_state_dict(torch.load('handrecognition_model3.pth', weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    # Config for quantization
    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,  # This will perform calibration
        input_shape=INPUT_SHAPE,
        inputs=None,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )

    criterion = nn.CrossEntropyLoss() 

    executor = TorchExecutor(graph=quant_ppq_graph, device=DEVICE)
    loss = 0
    for batch_x, batch_y in dataloader:
        y_pred = executor(batch_x)
        loss += criterion(y_pred[0], batch_y)
    loss /= len(dataloader)
    print(f"quant model loss: {loss.item():.5f}")
