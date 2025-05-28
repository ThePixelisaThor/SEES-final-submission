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

def convert_relu6_to_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU6):
            setattr(model, child_name, nn.ReLU())
        else:
            convert_relu6_to_relu(child)
    return model

from typing import Iterable, List, Tuple
from ppq import QuantizationSettingFactory, QuantizationSetting
from ppq.api import espdl_quantize_torch, get_target_platform
def quant_setting_mobilenet_v2(
    model: nn.Module,
    optim_quant_method: List[str] = None,
) -> Tuple[QuantizationSetting, nn.Module]:
    """Quantize torch model with optim_quant_method.

    Args:
        optim_quant_method (List[str]): support 'MixedPrecision_quantization', 'LayerwiseEqualization_quantization'
        -'MixedPrecision_quantization': if some layers in model have larger errors in 8-bit quantization, dispathching
                                        the layers to 16-bit quantization. You can remove or add layers according to your
                                        needs.
        -'LayerwiseEqualization_quantization'： using weight equalization strategy, which is proposed by Markus Nagel.
                                                Refer to paper https://openaccess.thecvf.com/content_ICCV_2019/papers/Nagel_Data-Free_Quantization_Through_Weight_Equalization_and_Bias_Correction_ICCV_2019_paper.pdf for more information.
                                                Since ReLU6 exists in MobilenetV2, convert ReLU6 to ReLU for better precision.

    Returns:
        [tuple]: [QuantizationSetting, nn.Module]
    """
    quant_setting = QuantizationSettingFactory.espdl_setting()
    if optim_quant_method is not None:
        if "MixedPrecision_quantization" in optim_quant_method:
            # These layers have larger errors in 8-bit quantization, dispatching to 16-bit quantization.
            # You can remove or add layers according to your needs.
            quant_setting.dispatching_table.append(
                "/fc1/Gemm",
                get_target_platform(TARGET, 16),
            )
            quant_setting.dispatching_table.append(
                "/fc2/Gemm",
                get_target_platform(TARGET, 16),
            )
        elif "LayerwiseEqualization_quantization" in optim_quant_method:
            # layerwise equalization
            quant_setting.equalization = True
            quant_setting.equalization_setting.iterations = 4
            quant_setting.equalization_setting.value_threshold = 0.4
            quant_setting.equalization_setting.opt_level = 2
            quant_setting.equalization_setting.interested_layers = None
            # replace ReLU6 with ReLU
            model = convert_relu6_to_relu(model)
        else:
            raise ValueError(
                "Please set optim_quant_method correctly. Support 'MixedPrecision_quantization', 'LayerwiseEqualization_quantization'"
            )

    return quant_setting, model

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

if __name__ == "__main__":

    quant_setting = QuantizationSettingFactory.espdl_setting()


    ESPDL_MODEL_PATH = "handrecognition_model4.espdl"
    INPUT_SHAPE = [1, 1, 45, 45]  # 1 个输入特征
    TARGET = "esp32s3"  # 目标量化精度
    NUM_OF_BITS = 16  # 量化位数
    DEVICE = "cpu"  # 'cuda' or 'cpu', if you use cuda, please make sure that cuda is available

    x_test, y_test = generate_data()
    # dataloader shuffle必须设置为False。
    # 因为计算量化误差的时候会多次遍历数据集，如果shuffle是True的话，会得到错误的量化误差。
    dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GesturePredictor()
    model.load_state_dict(torch.load('handrecognition_model3.pth', weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    """
    quant_setting, model = quant_setting_mobilenet_v2(
        model, ["MixedPrecision_quantization"]
    )
    """
    
    quant_ppq_graph = espdl_quantize_torch(
        model=model,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,  # 校准的步数
        input_shape=INPUT_SHAPE,  # 输入形状，批次为 1
        inputs=None,
        target=TARGET,  # 目标量化类型
        num_of_bits=NUM_OF_BITS,  # 量化位数
        collate_fn=collate_fn,
        # setting=quant_setting,
        dispatching_override=None,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,  # 输出详细日志信息
    )

    criterion = nn.CrossEntropyLoss() 
    # 原始模型测试集精度
    """
    loss = 0
    for batch_x, batch_y in dataloader:
        y_pred = model(batch_x)
        loss += criterion(y_pred, batch_y)
    loss /= len(dataloader)
    print(f"origin model loss: {loss.item():.5f}")
    """
    # 量化模型测试集精度
    executor = TorchExecutor(graph=quant_ppq_graph, device=DEVICE)
    loss = 0
    for batch_x, batch_y in dataloader:
        y_pred = executor(batch_x)
        loss += criterion(y_pred[0], batch_y)
    loss /= len(dataloader)
    print(f"quant model loss: {loss.item():.5f}")
