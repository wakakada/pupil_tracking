import cv2
import torch
import numpy as np
from model import SingleFrameCNN
from config import DEVICE, IMG_HEIGHT, IMG_WIDTH

def predict_and_annotate_video(video_path, model_path, output_path, orig_size=(640,480), confidence_threshold=None):
    """
    使用训练模型对视频逐帧预测并在原视频上标注瞳孔位置
    
    Args:
        video_path: 输入视频路径
        model_path: 模型权重路径
        output_path: 输出视频路径
        orig_size: 原始视频尺寸 (width, height)
        confidence_threshold: 置信度阈值（如果模型输出置信度分数）
    """
    model = SingleFrameCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 打开输入视频
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    print(f"开始处理视频: {video_path}")
    print(f"总帧数: {total_frames}, FPS: {fps}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 预测瞳孔位置
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
        inp = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        inp = inp.to(DEVICE)
        
        with torch.no_grad():
            pred_norm = model(inp).cpu().numpy()[0]
            
        # 转换归一化坐标到原始视频尺寸
        x_pixel = int(pred_norm[0] * orig_size[0])
        y_pixel = int(pred_norm[1] * orig_size[1])
        
        # 在帧上绘制瞳孔位置
        # 绘制一个圆圈标记瞳孔位置
        cv2.circle(frame, (x_pixel, y_pixel), radius=20, color=(0, 255, 0), thickness=2)
        
        # 绘制十字标记
        cv2.line(frame, (x_pixel-25, y_pixel), (x_pixel+25, y_pixel), (0, 255, 0), 3)
        cv2.line(frame, (x_pixel, y_pixel-25), (x_pixel, y_pixel+25), (0, 255, 0), 3)
        
        # 添加坐标文本
        cv2.putText(frame, f'Pupil: ({x_pixel}, {y_pixel})', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加帧数信息
        cv2.putText(frame, f'Frame: {frame_count}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 将处理后的帧写入输出视频
        out.write(frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧")

    # 释放资源
    cap.release()
    out.release()
    print(f"视频处理完成! 输出保存至: {output_path}")
    return frame_count

def predict_video_with_coordinates(video_path, model_path, orig_size=(640,480)):
    """
    保留原始功能：返回预测坐标但不标注视频
    """
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
    # 标注视频并保存
    input_video_path = "./LPW/22/1.avi"
    model_path = "checkpoint.pth"
    output_video_path = "annotated_pupil_tracking.mp4"
    
    # 处理视频并添加标注
    total_frames = predict_and_annotate_video(
        video_path=input_video_path,
        model_path=model_path,
        output_path=output_video_path,
        orig_size=(640, 480)  # 根据你的实际视频尺寸调整
    )
    
    # 如果你还想获取坐标数据，可以使用原来的函数
    coords = predict_video_with_coordinates(input_video_path, model_path)
    
    # 打印部分坐标信息
    for i, (x, y) in enumerate(coords[:10]):  # 只打印前10帧
        print(f"Frame {i}: ({x:.1f}, {y:.1f})")
    
    # 保存坐标到CSV文件
    import csv
    with open("predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y"])
        for i, (x, y) in enumerate(coords):
            writer.writerow([i, f"{x:.2f}", f"{y:.2f}"])
    
    print("预测坐标已保存到 predictions.csv")