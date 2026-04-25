import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import *
from video_dataset import VideoFrameDataset
from model import SingleFrameCNN
import os
from tqdm import tqdm
import numpy as np

class EarlyStopping:
    """早停机制实现"""
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA, model_save_path=MODEL_SAVE_PATH):
        self.patience = patience
        self.min_delta = min_delta
        self.model_save_path = model_save_path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), self.model_save_path)
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")
        else:
            self.counter += 1
            print(f"  -> Early stopping counter: {self.counter} out of {self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True

def create_datasets():
    """创建训练和验证数据集，只调用一次以避免重复加载"""
    # 获取所有受试者编号
    all_subjects = [int(d) for d in os.listdir(LPW_ROOT) if os.path.isdir(os.path.join(LPW_ROOT, d)) and d.isdigit()]
    all_subjects.sort()

    # 随机打乱
    if RANDOM_SEED:
        random.seed(RANDOM_SEED)
    random.shuffle(all_subjects)

    # 按比例划分
    split_idx = int(len(all_subjects) * TRAIN_RATIO)
    train_subjects = all_subjects[:split_idx]
    val_subjects = all_subjects[split_idx:]

    print(f"数据集划分（按受试者，共 {len(all_subjects)} 个）:")
    print(f"Train subjects:{train_subjects}")
    print(f"Val subjects:{val_subjects}")

    # 创建训练和验证数据集
    train_dataset = VideoFrameDataset(LPW_ROOT, train_subjects, img_size=(IMG_HEIGHT, IMG_WIDTH))
    val_dataset = VideoFrameDataset(LPW_ROOT, val_subjects, img_size=(IMG_HEIGHT, IMG_WIDTH))
    
    return train_dataset, val_dataset

def main():
    train_dataset, val_dataset = create_datasets()
    
    # 检查数据集是否为空
    if len(train_dataset) == 0:
        raise ValueError(f"训练数据集为空，请检查训练受试者数据是否存在")
    if len(val_dataset) == 0:
        raise ValueError(f"验证数据集为空，请检查验证受试者数据是否存在")
    
    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"验证数据集大小: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = SingleFrameCNN().to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # 初始化早停机制
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA)

    for epoch in range(1, EPOCHS+1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 使用tqdm包装训练数据加载器，显示训练进度
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Training", leave=False)
        for imgs, targets in train_pbar:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            
            # 更新进度条显示当前损失
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        # 随机选择一个验证样本用于展示预测结果
        display_sample = True
         
        # 使用tqdm包装验证数据加载器，显示验证进度
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - Validation", leave=False)
        with torch.no_grad():
            for batch_idx, (imgs, targets) in enumerate(val_pbar):
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item() * imgs.size(0)
                
                # 随机选择一个验证样本展示预测结果
                if display_sample and batch_idx == 0:  # 在第一个批次中选择一个样本展示
                    # 随机选择批次中的一个样本
                    sample_idx = random.randint(0, imgs.size(0)-1)
                    sample_img = imgs[sample_idx:sample_idx+1]  # 保持batch维度
                    sample_target = targets[sample_idx]
                    sample_pred = model(sample_img)
                    
                    # 将归一化坐标转换回像素坐标（原始图像尺寸为640x480）
                    orig_w, orig_h = 640, 480  # 原始图像尺寸
                    pred_x_pixel = sample_pred[0, 0].item() * orig_w
                    pred_y_pixel = sample_pred[0, 1].item() * orig_h
                    target_x_pixel = sample_target[0].item() * orig_w
                    target_y_pixel = sample_target[1].item() * orig_h
                    
                    print(f"\n随机验证样本预测结果 (Epoch {epoch}):")
                    print(f"  预测坐标: ({pred_x_pixel:.2f}, {pred_y_pixel:.2f})")
                    print(f"  真实坐标: ({target_x_pixel:.2f}, {target_y_pixel:.2f})")
                    print(f"  坐标误差: {np.sqrt((pred_x_pixel - target_x_pixel)**2 + (pred_y_pixel - target_y_pixel)**2):.2f} 像素")
                    
                    display_sample = False  # 确保只显示一次
                
                # 更新进度条显示当前损失
                val_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        # 使用早停机制替代原来的保存逻辑
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered after {epoch} epochs")
            break

if __name__ == "__main__":
    main()