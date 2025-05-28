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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

IMG_WIDTH, IMG_HEIGHT = 45, 45

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
    
import albumentations as A

augment = A.Compose([
    A.Rotate(limit=5, p=0.5),  
    A.Affine(scale=(0.8, 1.05), translate_percent=(0.2, 0.2), p=0.5)
    # A.RandomBrightnessContrast(brightness_limit=0.1, p=0.5),
    # A.GaussianBlur(blur_limit=(3, 7), p=0.3)
    # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5), # this one makes it plateau at 78%
])


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
                    match = re.match(r'(\d{2})_', folder)
                    label = int(match.group(1))
                    labels.append(label - 1)

    # Convert to numpy arrays
    X = np.array(data, dtype=np.float32) # Normalize pixel values
    Y = np.array(labels, dtype=np.int64)

    # Reshape X to fit model input (Add channel dimension)
    X = X.reshape(-1, 1, IMG_WIDTH, IMG_HEIGHT)

    augmented_X = []
    augmented_Y = []

    for i in range(len(X)):
        img = X[i]  # NumPy array (H, W)
        img_aug = augment(image=img)['image']  # Apply transformation
        augmented_X.append(img_aug)
        augmented_Y.append(Y[i])  # Keep labels

    # Convert to NumPy array
    augmented_X = np.array(augmented_X)
    augmented_Y = np.array(augmented_Y)

    X_combined = np.concatenate((X, augmented_X), axis=0) / 255.0
    Y_combined = np.concatenate((Y, augmented_Y), axis=0)

    X_combined = augmented_X / 255.0
    Y_combined = augmented_Y

    X_train, X_val, Y_train, Y_val = train_test_split(X_combined, Y_combined, test_size=0.1, random_state=42)

    return X_train, Y_train, X_val, Y_val


def evaluate_model(model, x_train, y_train):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # Convert labels to torch.long (int64)
        
        # Forward pass
        outputs = model(x_train)
        _, predicted = torch.max(outputs, 1)  # Get class index with maximum score
        total += y_train.size(0)
        correct += (predicted == y_train).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test data: {accuracy}%')


if __name__ == "__main__":
    print(torch.cuda.is_available())  # Should return True
    print(torch.cuda.device_count())  # Should return 1 (or more if multiple GPUs)
    print(torch.cuda.get_device_name(0))  # Should return "NVIDIA GeForce RTX 2070"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GesturePredictor().to(device)
    from torchsummary import summary
    summary(model, input_size=(1, 45, 45))

    x_train, y_train, x_val, y_val = generate_data()

    # x_train, y_train, x_val, y_val = torch.load("gesture_dataset.pt")

    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    x_val = torch.tensor(x_val)
    y_val = torch.tensor(y_val)

    # torch.save((x_train, y_train, x_val, y_val), "gesture_dataset.pt")

    x_val = x_val.to(device)
    y_val = y_val.to(device)

    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)

    # model.load_state_dict(torch.load('handrecognition_model3.pth', weights_only=True))
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-4)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.8, verbose=True)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)  # AdamW > Adam
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)  # Learning rate decay

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Helps generalization

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device) 
            # 前向传播
            y_pred = model(batch_x)

            # 计算损失
            loss = criterion(y_pred, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # 打印训练进度
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader)}')
        scheduler.step(running_loss)
        model.eval()
        evaluate_model(model, x_val, y_val)
        torch.save(model.state_dict(), "handrecognition_model3.pth")
        model.train()

    # 切换到评估模式
    model.eval()

    # evaluate_model(model, x_train, y_train)

    # 保存pytorch模型权重
    torch.save(model.state_dict(), "handrecognition_model3.pth")

    # 导出onnx模型
    # esp-dl当前只支持 batch size=1
    dummy_input = torch.randn([1, 1, 45, 45], dtype=torch.float32)
    torch.onnx.export(
        model.to(0),
        dummy_input,
        "handrecognition_model3.onnx",
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
    )
    onnx_model = onnx.load_model("handrecognition_model3.onnx")
    onnx.checker.check_model(onnx_model)
    onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx_model, "handrecognition_model3.onnx")
