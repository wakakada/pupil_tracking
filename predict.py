# 使用训练模型对avi视频逐帧预测
import cv2
import torch
import numpy as np
from model import SingleFrameCNN
from config import DEVICE, IMG_HEIGHT, IMG_WIDTH

def predict_video(video_path, model_path, orig_size=(640,480)):
    model = SingleFrameCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    cap = cv2.VideoCapture(video_path)
    results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
        inp = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        inp = inp.to(DEVICE)
        with torch.no_grad():
            pred_norm = model(inp).cpu().numpy()[0]
        x_pixel = pred_norm[0] * orig_size[0]
        y_pixel = pred_norm[1] * orig_size[1]
        results.append((x_pixel, y_pixel))
    cap.release()
    return results

if __name__ == "__main__":
    coords = predict_video("test.avi", "single_frame_cnn.pth")
    for i, (x, y) in enumerate(coords):
        print(f"Frame {i}: ({x:.1f}, {y:.1f})")
    # 保存到CSV
    import csv
    with open("predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y"])
        for i, (x, y) in enumerate(coords):
            writer.writerow([i, f"{x:.2f}", f"{y:.2f}"])