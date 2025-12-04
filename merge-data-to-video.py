import traceback
import logging

import cv2
import json

def add_text_to_video(video_path, data, output_path):
    """
    在视频中添加文本信息
    
    Args:
        video_path: 输入视频路径
        data: 包含分析数据的列表
        output_path: 输出视频路径
    """
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频的基本信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 将数据按帧索引建立字典，便于快速查找
    data_dict = {}
    for item in data:
        data_dict[item['frame_index']] = item
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # 获取当前帧的数据
            current_data = data_dict.get(frame_count)
            
            # 添加文本信息
            if current_data and current_data['persons']:
                person = current_data['persons'][0]  # 假设只处理第一个人
                
                # 提取数据
                score = person['score']
                balance_metric = person['derived_metrics']['balance_metric']
                left_knee_bend = person['derived_metrics']['left_knee_bend']
                right_knee_bend = person['derived_metrics']['right_knee_bend']
                torso_twist = person['derived_metrics']['torso_twist']
                vertical_stability = person['derived_metrics']['vertical_stability']
                
                # 定义文字参数
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                color = (0, 255, 0)  # 绿色
                thickness = 2
                line_spacing = 30  # 行间距
                
                # 文字内容
                texts = [
                    f"score: {score:.3f}",
                    f"balanced_metric (balance: {balance_metric:.2f})",
                    f"left_knee_bend (left: {left_knee_bend:.2f})",
                    f"right_knee_bend (right: {right_knee_bend:.2f})",
                    f"torso_twist (torso: {torso_twist:.2f})",
                    f"vertical_stability (vertical: {vertical_stability:.2f})"
                ]
                
                # 在视频上添加文字
                start_y = 30  # 起始Y坐标
                for i, text in enumerate(texts):
                    y_position = start_y + i * line_spacing
                    cv2.putText(frame, text, (10, y_position), font, font_scale, color, thickness)
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.info(f"FRAME: {frame_count}")
        
        # 写入帧到输出视频
        out.write(frame)
        
        frame_count += 1
        
        # 显示进度
        if frame_count % 100 == 0:
            print(f"处理进度: {frame_count}/{total_frames} 帧")
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"视频处理完成，输出文件: {output_path}")


# 使用示例
if __name__ == "__main__":
    input_video_path = "/Users/howardwang/Desktop/playground/ski-pose/dataset/chill-snowboard-13.mp4"  # 输入视频路径
    output_video_path = "/Users/howardwang/Desktop/playground/ski-pose/dataset/chill-snowboard-with-text-13.mp4"  # 输出视频路径
    data_path = '/Users/howardwang/Desktop/playground/ski-pose/chill-snowboard-13-pose-analysis.json'
    with open(data_path) as f:
        data = json.load(f)
    
    add_text_to_video(input_video_path, data, output_video_path)