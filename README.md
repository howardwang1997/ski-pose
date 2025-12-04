# 单板滑雪姿态分析系统

基于YOLO-NAS-Pose模型的单板滑雪姿态分析系统，能够从视频中提取滑雪者的姿态信息，分析下半身姿态和重心变化，为滑雪技巧改进提供数据支持。

## 目录
- [部署环境](#部署环境)
- [模型下载](#模型下载)
- [实现步骤和逻辑](#实现步骤和逻辑)
- [使用方法](#使用方法)
- [输出结果](#输出结果)

## 部署环境

### 系统要求
- Python 3.8 或更高版本
- CUDA 兼容的 GPU (推荐) 或 CPU

### 安装步骤

1. 克隆或下载项目到本地

2. 创建并激活虚拟环境
```bash
# 使用 conda 创建环境
conda create -n skipose python=3.8
conda activate skipose

# 或使用 venv 创建环境
python -m venv skipose
# Windows
skipose\Scripts\activate
# Linux/Mac
source skipose/bin/activate
```

3. 安装依赖包
```bash
pip install -r requirements.txt
```

如果没有 `requirements.txt`，请手动安装以下依赖：
```bash
pip install torch torchvision torchaudio
pip install super-gradients
pip install opencv-python
pip install numpy
```

## 模型下载

本系统使用 YOLO-NAS-Pose 模型进行姿态检测。模型文件应放置在 `model` 目录下：

1. 创建 `model` 目录
```bash
mkdir model
```

2. 下载预训练模型文件 (推荐使用 YOLO-NAS-POSE-L 版本以获得最佳精度)
   - `yolo_nas_pose_l_coco_pose.pth` - 大型模型，精度最高
   - `yolo_nas_pose_m_coco_pose.pth` - 中型模型，平衡精度和速度
   - `yolo_nas_pose_s_coco_pose.pth` - 小型模型，速度最快
   - `yolo_nas_pose_n_coco_pose.pth` - 微型模型，速度极快

模型文件可以从 SuperGradients 官方仓库或通过以下方式获取：
```python
from super_gradients.training import models
model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
model.save("model/yolo_nas_pose_l_coco_pose.pth")
```

## 实现步骤和逻辑

### 1. 系统架构

本系统主要由以下几个部分组成：
- **模型加载模块**：加载YOLO-NAS-Pose预训练模型
- **视频处理模块**：读取视频，提取指定时间段
- **姿态检测模块**：使用模型检测每一帧的人体关键点
- **姿态分析模块**：计算下半身姿态指标和重心位置
- **结果输出模块**：生成可视化视频和JSON格式的分析数据

### 2. 核心算法流程

1. **初始化阶段**
   - 设置设备 (GPU/CPU)
   - 加载YOLO-NAS-Pose模型
   - 配置输入输出路径

2. **视频预处理**
   - 读取视频文件获取基本信息 (FPS、总帧数)
   - 计算需要处理的帧范围 (根据指定的开始和结束时间)
   - 使用模型对整个视频进行预测

3. **数据筛选与缓存**
   - 将预测结果生成器转换为列表
   - 根据时间范围筛选需要的帧
   - 用筛选后的结果替换原始生成器

4. **姿态分析**
   - 遍历每一帧的预测结果
   - 提取17个COCO关键点
   - 计算以下关键指标：
     - **平衡指标** (balance_metric)：重心与脚踝中心的水平偏移
     - **膝盖弯曲度** (left_knee_bend/right_knee_bend)：髋-膝-踝三点角度
     - **躯干扭转** (torso_twist)：肩线与髋线的角度差
     - **垂直稳定性** (vertical_stability)：重心的垂直位置

5. **结果输出**
   - 将分析结果保存为JSON文件
   - 生成带有姿态标注的可视化视频

### 3. 关键技术点

1. **关键点检测**
   - 使用COCO数据集的17个关键点，包括鼻子、眼睛、耳朵、肩膀、肘部、手腕、臀部、膝盖和脚踝
   - 特别关注下半身关键点 (臀部、膝盖、脚踝) 用于滑雪姿态分析

2. **角度计算**
   - 使用向量点积计算三点间角度
   - 使用反正切函数计算水平角度
   - 处理异常值和边界情况

3. **重心计算**
   - 以左右髋关节中心作为重心近似点
   - 计算重心与脚踝中心的相对位置评估平衡

## 使用方法

### 基本使用

1. 准备视频文件
   - 将滑雪视频放入 `dataset` 目录
   - 支持常见视频格式 (MP4, AVI, MOV等)

2. 配置参数
   - 在 `yolo-nas-pose-detail.py` 中修改以下参数：
     ```python
     INPUT_VIDEO_FILENAME = "your_video.mp4"  # 视频文件名
     START_TIME_SECONDS = 0                  # 开始时间 (秒)
     END_TIME_SECONDS = 30                   # 结束时间 (秒)
     CONFIDENCE = 0.6                        # 检测置信度阈值
     ```

3. 运行分析
   ```bash
   python yolo-nas-pose-detail.py
   ```

### 简化版使用 (适合快速测试)

1. 使用简化版脚本进行基本姿态检测：
   ```bash
   python simple_pose.py
   ```

2. 修改 `simple_pose.py` 中的视频路径：
   ```python
   input_video = "dataset/chill-snowboard-01.mp4"  # 修改为其他视频文件
   output_video = "output/pose_result_01.mp4"      # 修改输出文件名
   ```

## 输出结果

系统会生成两种输出文件：

1. **可视化视频** (`output/video/视频名-nas-pose.mp4`)
   - 原始视频片段 (指定时间段)
   - 关键点标注 (绿色点)
   - 下半身关键点特别标注 (红色点)
   - 骨架连接线
   - 重心位置标记 (黄色点)

2. **分析数据** (`output/json/视频名-pose-analysis.json`)
   ```json
   [
     {
       "frame_index": 0,
       "persons": [
         {
           "person_id": 0,
           "score": 0.95,
           "derived_metrics": {
             "balance_metric": 5.2,
             "left_knee_bend": 120.5,
             "right_knee_bend": 118.3,
             "torso_twist": 5.7,
             "vertical_stability": 320.1
           }
         }
       ]
     }
   ]
   ```

### 指标说明

- **balance_metric**: 重心与脚踝中心的水平偏移 (像素值)。正值表示重心偏向右侧，负值表示偏向左侧。
- **left_knee_bend/right_knee_bend**: 左右膝盖弯曲角度 (度)。值越小表示膝盖弯曲越大。
- **torso_twist**: 躯干扭转角度 (度)。正值表示顺时针扭转，负值表示逆时针扭转。
- **vertical_stability**: 重心的垂直位置 (像素值)。可用于分析身体姿态的稳定性。

## 项目结构

```
sbPose/
├── dataset/                 # 视频数据集
│   ├── chill-snowboard-01.mp4
│   ├── chill-snowboard-02.mp4
│   └── ...
├── model/                   # 模型文件目录
│   ├── yolo_nas_pose_l_coco_pose.pth
│   └── ...
├── output/                  # 输出结果目录
│   ├── video/              # 可视化视频
│   └── json/               # 分析数据
├── yolo-nas-pose-detail.py # 详细分析脚本
├── simple_pose.py          # 简化版YOLOv8 Pose脚本
├── yolo-nas-pose.py        # 基础YOLO-NAS-Pose脚本
├── requirements.txt        # 依赖包列表
└── README.md               # 项目说明
```

## 关键点说明

模型检测的17个COCO关键点包括：
- 0: 鼻子
- 1: 左眼
- 2: 右眼
- 3: 左耳
- 4: 右耳
- 5: 左肩
- 6: 右肩
- 7: 左肘
- 8: 右肘
- 9: 左腕
- 10: 右腕
- 11: 左髋
- 12: 右髋
- 13: 左膝
- 14: 右膝
- 15: 左踝
- 16: 右踝

## 注意事项

1. 视频中滑雪者应清晰可见，避免严重遮挡
2. 建议使用侧面或斜侧面视角的视频，以便更好地分析姿态
3. 处理时间取决于视频长度和硬件性能，建议使用GPU加速
4. 模型检测置信度可根据视频质量调整，质量较低时可适当降低阈值

## 故障排除

1. **模型加载失败**
   - 检查模型文件是否存在于 `model` 目录
   - 确认模型文件名是否正确

2. **视频处理失败**
   - 检查视频文件路径是否正确
   - 确认视频格式是否受支持

3. **内存不足**
   - 减少处理的视频长度
   - 降低视频分辨率
   - 使用更小的模型 (如YOLO-NAS-POSE-S)

## 技术栈

- Python 3.8+
- SuperGradients (YOLO-NAS-Pose)
- OpenCV
- PyTorch
- NumPy