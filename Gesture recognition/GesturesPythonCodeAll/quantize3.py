import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import re

import torch
from torch.utils.data import DataLoader, TensorDataset
from ppq.api import espdl_quantize_onnx

IMG_WIDTH, IMG_HEIGHT = 45, 45

def collate_fn(batch):
    # TensorDataset 迭代的时候返回的是 Tuple(x, y), 量化的时候只需要x, 不需要label y。
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
                    match = re.match(r'(\d{2})_', folder)
                    label = int(match.group(1))
                    labels.append(label - 1)

    # Convert to numpy arrays
    X = np.array(data, dtype=np.float32) # Normalize pixel values
    Y = np.array(labels, dtype=np.int64)

    # Reshape X to fit model input (Add channel dimension)
    X = X.reshape(-1, 1, IMG_WIDTH, IMG_HEIGHT) / 255.0

    return X, Y

if __name__ == "__main__":

    ONNX_MODEL_PATH = "handrecognition_model3.onnx"
    ESPDL_MODEL_PATH = "handrecognition_model3.espdl"
    INPUT_SHAPE = [1, 1, 45, 45]  # 1 个输入特征
    TARGET = "esp32s3"  # 目标量化精度
    NUM_OF_BITS = 16  # 量化位数
    DEVICE = "cpu"  # 'cuda' or 'cpu', if you use cuda, please make sure that cuda is available

    x, y = generate_data()
    print("X_train shape:", x.shape)
    print("Y_train shape:", y.shape)
    x = torch.tensor(x)
    y = torch.tensor(y)
    print("X_train shape:", x.shape)
    print("Y_train shape:", y.shape)
    # dataloader shuffle必须设置为False。
    # 因为计算量化误差的时候会多次遍历数据集，如果shuffle是True的话，会得到错误的量化误差。
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=ONNX_MODEL_PATH,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,  # 校准的步数
        input_shape=INPUT_SHAPE,  # 输入形状，批次为 1
        inputs=None,
        target=TARGET,  # 目标量化类型
        num_of_bits=NUM_OF_BITS,  # 量化位数
        collate_fn=collate_fn,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,  # 输出详细日志信息
    )
