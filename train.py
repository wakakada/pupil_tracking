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

    best_val_loss = float('inf')
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
        
        # 使用tqdm包装验证数据加载器，显示验证进度
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - Validation", leave=False)
        with torch.no_grad():
            for imgs, targets in val_pbar:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item() * imgs.size(0)
                
                # 更新进度条显示当前损失
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")

if __name__ == "__main__":
    main()