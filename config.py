import torch
import os
import random

LPW_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "LPW")

# 训练集、数据集划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
RANDOM_SEED = 42  # 随机数种子

# 训练参数
IMG_HEIGHT = 45
IMG_WIDTH  = 60
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "checkpoint.pth"