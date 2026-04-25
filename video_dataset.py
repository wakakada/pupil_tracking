import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm

class VideoFrameDataset(Dataset):
    def __init__(self, lpw_root, subject_ids, img_size=(45, 60)):
        """
        lpw_root: LPW 数据集根目录 (如 ./LPW/)
        subject_ids: 受试者ID列表 (例如 [1, 2, 3, ...])
        img_size: (height, width) 模型输入尺寸
        """
        self.lpw_root = lpw_root
        self.subject_ids = subject_ids
        self.img_h, self.img_w = img_size
        self.cached_samples = []   # 存储所有 (frame_tensor, label_tensor)

        # 收集所有视频文件路径
        all_video_paths = []
        for subject_id in self.subject_ids:
            subject_path = os.path.join(self.lpw_root, str(subject_id))
            
            if not os.path.isdir(subject_path):
                print(f"Warning: Subject directory {subject_path} does not exist, skipping...")
                continue
                
            # 获取该受试者的所有视频和标签文件
            video_files = [f for f in os.listdir(subject_path) if f.lower().endswith('.avi')]
            
            for vf in video_files:
                video_path = os.path.join(subject_path, vf)
                label_name = os.path.splitext(vf)[0] + '.txt'
                label_path = os.path.join(subject_path, label_name)
                
                if os.path.exists(label_path):
                    all_video_paths.append((video_path, label_path, subject_path))
                else:
                    print(f"Warning: Label file {label_path} not found, skipping {vf}")

        print(f"正在加载数据集，共 {len(all_video_paths)} 个视频...")
        
        # 使用进度条处理所有视频
        for video_path, label_path, subject_path in tqdm(all_video_paths, desc="Processing Videos"):
            # 获取视频属性（总帧数，原始尺寸）
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            if total_frames == 0:
                print(f"Warning: Video {video_path} has 0 frames, skipping...")
                continue

            # 读取所有标签（原始像素坐标）
            with open(label_path, 'r') as f:
                label_lines = [line.strip() for line in f.readlines()]
            
            # 确保标签行数足够
            if len(label_lines) < total_frames:
                print(f"Warning: Label file {label_path} has fewer lines ({len(label_lines)}) than frames ({total_frames}), skipping video")
                continue

            # 逐帧读取和处理
            cap = cv2.VideoCapture(video_path)
            for idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Cannot read frame {idx} from {video_path}, skipping")
                    continue
                    
                # 处理图像
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (self.img_w, self.img_h))
                normalized = resized.astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(normalized).float().unsqueeze(0)   # (1, H, W)

                # 解析标签
                line = label_lines[idx]
                parts = line.split()
                x, y = float(parts[0]), float(parts[1])
                
                # 归一化坐标
                x_norm = x / orig_w
                y_norm = y / orig_h
                label_tensor = torch.tensor([x_norm, y_norm], dtype=torch.float32)

                self.cached_samples.append((frame_tensor, label_tensor))
            
            cap.release()

        print(f"数据集加载完成，共加载了 {len(self.cached_samples)} 个样本")

    def __len__(self):
        return len(self.cached_samples)

    def __getitem__(self, idx):
        return self.cached_samples[idx]   # 直接返回缓存的张量