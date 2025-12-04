import torch
import os
import json
import numpy as np  # 导入Numpy进行数学计算
import cv2          # 导入OpenCV以获取视频FPS
from super_gradients.training import models
from super_gradients.common.object_names import Models

# --- ⬇️ (1) 在这里设置你要处理的文件和时间 ⬇️ ---
INPUT_VIDEO_FILENAME = "edward_sz_01.mp4"
START_TIME_SECONDS = 0   # 开始时间
END_TIME_SECONDS = 100    # 结束时间
CONFIDENCE = 0.6         # 识别的置信度阈值
# --- ⬆️ (1) 设置完成 ⬆️ ---

# COCO 17个关键点名称 (按顺序)
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# --- ⬇️ (NEW) 辅助函数：计算三个点之间的角度 ⬇️ ---
def calculate_angle(p1, p2, p3):
    """
    计算 p1-p2-p3 在 p2 点的角度 (0-180度)
    p1, p2, p3 是 (x, y) 坐标
    """
    try:
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return None
        
        cosine_angle = dot_product / (norm_v1 * norm_v2)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    except:
        return None

# --- ⬇️ (NEW) 辅助函数：计算两点连线的水平角度 ⬇️ ---
def calculate_horizontal_angle(p1, p2):
    """
    计算 p1 到 p2 连线与水平线x轴的角度
    """
    try:
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1] # 图像y轴是向下的
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    except:
        return None
# --- (END NEW) ---


# --- 创建输出目录 (根据您的代码更新) ---
os.makedirs("output", exist_ok=True)
os.makedirs("output/video", exist_ok=True)
os.makedirs("output/json", exist_ok=True)

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# --- 模型加载 ---
try:
    checkpoint_path = "model/yolo_nas_pose_n_coco_pose.pth"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join("model", "yolo_nas_pose_n_coco_pose.pth")
        
    if os.path.exists(checkpoint_path):
        print(f"使用本地权重文件: {checkpoint_path}")
        model = models.get(
            Models.YOLO_NAS_POSE_N,
            num_classes=17,
            checkpoint_path=checkpoint_path
        )
    else:
        raise FileNotFoundError(f"在指定路径找不到本地权重文件: {checkpoint_path}")
    
    model.to(device)
    model.eval()
    print("模型加载成功！")
    
    # --- (2) 自动生成输入和输出路径 ---
    input_video_path = os.path.join("dataset", INPUT_VIDEO_FILENAME)
    base_filename = os.path.splitext(INPUT_VIDEO_FILENAME)[0]
    
    output_video_path = os.path.join("output/video", f"{base_filename}-nas-pose.mp4")
    # (NEW) 我们保存的是分析(analysis)文件
    output_json_path = os.path.join("output/json", f"{base_filename}-pose-analysis.json")
    
    
    if not os.path.exists(input_video_path):
        print(f"错误: 输入文件 {input_video_path} 不存在")
    else:
        # --- (NEW) 处理时间范围 ---
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        start_frame = int(START_TIME_SECONDS * fps)
        end_frame = int(END_TIME_SECONDS * fps)
        
        # 确保不超过视频总帧数
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)
        
        print(f"视频总帧率: {fps:.2f} FPS. 总帧数: {total_frames}")
        print(f"将处理帧范围: {start_frame} 到 {end_frame} (对应 {START_TIME_SECONDS}s 到 {END_TIME_SECONDS}s)")
        
        # -----------------------------------
        
        print(f"开始处理视频: {input_video_path}")
        result = model.predict(input_video_path, conf=CONFIDENCE)
        # print(result)
        # with open('/Users/howardwang/Desktop/playground/ski-pose/output/json/log.json', 'w') as f:
        #     json.dump(result)
        print("视频预测完成。")

        # --- (3) 核心：缓存生成器 ---
        print("正在将预测结果（生成器）缓存到内存列表...")
        try:
            # 这将消耗 _images_prediction_gen 并将其内容存储在一个列表中
            frame_predictions_list = list(result._images_prediction_gen)
            print(f"缓存了 {len(frame_predictions_list)} 帧数据。")
            
            # (NEW) 只保留指定范围内的帧预测结果 (来自您的代码)
            filtered_predictions = frame_predictions_list[start_frame:end_frame]
            print(f"已筛选出 {len(filtered_predictions)} 帧数据进行处理")
            
            # 关键：用这个*已筛选*的列表替换掉生成器
            # 这样，我们和 .save() 方法都只会处理这个片段
            result._images_prediction_gen = filtered_predictions
            
        except Exception as e:
            print(f"缓存生成器时出错: {e}")
            raise e
        # --- (3) 缓存和筛选完成 ---


        # --- (4) 第一步：二次提炼数据并保存为JSON ---
        print(f"开始二次提炼，分析并保存到: {output_json_path}")
        all_frames_analysis = [] 
        
        # 我们现在遍历的是已筛选的列表 (filtered_predictions)
        for frame_index, frame_prediction in enumerate(result._images_prediction_gen):
            
            # (NEW) 使用 (frame_index + start_frame) 来获取在*原始视频*中的真实帧号
            frame_analysis = {
                "frame_index": frame_index + start_frame, 
                "persons": []
            }
            
            poses = frame_prediction.prediction.poses
            scores = frame_prediction.prediction.scores
            
            for i, (pose, score) in enumerate(zip(poses, scores)):
                
                keypoints = {}
                for j, (x, y, conf) in enumerate(pose):
                    if conf > 0.5: # 只考虑高置信度的点
                        keypoints[COCO_KEYPOINTS[j]] = (float(x), float(y))
                
                # --- (NEW) 开始计算关键指标 ---
                derived_metrics = {
                    "balance_metric": None,
                    "left_knee_bend": None,
                    "right_knee_bend": None,
                    "torso_twist": None,
                    "vertical_stability": None
                }

                try:
                    # 指标 1: 重心 (我们需要这个来进行其他计算)
                    lh_hip = keypoints.get("left_hip")
                    rh_hip = keypoints.get("right_hip")
                    if lh_hip and rh_hip:
                        cog_x = (lh_hip[0] + rh_hip[0]) / 2
                        cog_y = (lh_hip[1] + rh_hip[1]) / 2
                        derived_metrics["vertical_stability"] = cog_y
                        
                        # 指标 2: 平衡点 (Balance Metric)
                        lh_ankle = keypoints.get("left_ankle")
                        rh_ankle = keypoints.get("right_ankle")
                        if lh_ankle and rh_ankle:
                            ankle_center_x = (lh_ankle[0] + rh_ankle[0]) / 2
                            derived_metrics["balance_metric"] = cog_x - ankle_center_x
                
                    # 指标 3: 膝盖弯曲度 (Knee Bend)
                    if "left_hip" in keypoints and "left_knee" in keypoints and "left_ankle" in keypoints:
                        derived_metrics["left_knee_bend"] = calculate_angle(
                            keypoints["left_hip"], keypoints["left_knee"], keypoints["left_ankle"]
                        )
                    if "right_hip" in keypoints and "right_knee" in keypoints and "right_ankle" in keypoints:
                        derived_metrics["right_knee_bend"] = calculate_angle(
                            keypoints["right_hip"], keypoints["right_knee"], keypoints["right_ankle"]
                        )

                    # 指标 4: 躯干扭转 (Torso Twist)
                    if "left_shoulder" in keypoints and "right_shoulder" in keypoints and lh_hip and rh_hip:
                        shoulder_angle = calculate_horizontal_angle(
                            keypoints["left_shoulder"], keypoints["right_shoulder"]
                        )
                        hip_angle = calculate_horizontal_angle(lh_hip, rh_hip)
                        
                        if shoulder_angle is not None and hip_angle is not None:
                            twist = shoulder_angle - hip_angle
                            derived_metrics["torso_twist"] = (twist + 180) % 360 - 180
                
                except Exception:
                    pass
                
                # --- (NEW) 计算结束 ---
                
                person_analysis = {
                    "person_id": i,
                    "score": float(score),
                    "derived_metrics": derived_metrics
                }
                
                frame_analysis["persons"].append(person_analysis)
            
            all_frames_analysis.append(frame_analysis)
            
            if frame_index % 100 == 0 and frame_index > 0:
                print(f"已提炼 {frame_index} 帧...")

        print("--- 数据提炼完成 ---")

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_frames_analysis, f, indent=4, ensure_ascii=False)
        print(f"JSON分析数据保存成功！(保存至: {output_json_path})")
        
        
        # --- (5) 第二步：保存可视化视频 ---
        print(f"正在保存可视化视频剪辑到: {output_video_path}")
        try:
            # .save() 现在遍历的是我们筛选后的列表，因此只会保存剪辑后的片段
            result.save(output_video_path)
            print(f"处理完成！视频剪辑已保存至: {output_video_path}")
        except Exception as e:
            print(f"保存视频时出错: {e}")

except Exception as e:
    print(f"发生致命错误: {str(e)}")