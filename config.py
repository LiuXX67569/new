import torch

# --- 数据集配置 ---
MVTEC_DATA_PATH = "C:/Users/28022/Desktop/new/src/data/mvtec" # 请修改为你的 MVTec AD 数据集实际路径
MVTEC_CATEGORIES = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
#MVTEC_CATEGORIES = ["bottle"] # 为了快速测试，先只用一个类别，后续可以取消注释使用所有类别

# --- 模型配置 ---
# 可选: 'efficientnet-b4', 'resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2' 等 torchvision 支持的模型
PRETRAINED_MODEL_NAME = 'efficientnet-b4'
# 特征提取层 - 根据 PRETRAINED_MODEL_NAME 选择，需要查看模型结构确定
# 例如对于 wide_resnet50_2, 可能是 'layer1', 'layer2', 'layer3'
FEATURE_LAYER_NAMES = [
    'features.2',  # Level 3
    'features.3',  # Level 4
    'features.4',  # Level 5
    'features.5',  # Level 6
    'features.6',  # Level 7
]
# FEATURE_LAYER_NAMES = ['blocks_3', 'blocks_5'] # EfficientNet-B4 的例子 (可能需要根据 efficientnet-pytorch 库调整)
# # 根据论文分析，使用 EfficientNet-B4 并提取所有主要特征层的输出
# FEATURE_LAYER_NAMES = [
#     'features.0',  # Level 1
#     'features.1',  # Level 2
#     'features.2',  # Level 3
#     'features.3',  # Level 4
#     'features.4',  # Level 5
#     'features.5',  # Level 6
#     'features.6',  # Level 7
#     'features.7',  # Level 8
#     'features.8',  # Level 9
# ]
# # --- 修改：减少使用的特征层数以降低内存消耗 ---
# FEATURE_LAYER_NAMES = [
#     'features.2',  # Level 3
#     'features.3',  # Level 4
#     'features.4',  # Level 5
#     'features.5',  # Level 6
#     'features.6',  # Level 7
# ]

# --- 预处理配置 ---
IMAGE_SIZE = (256, 256) # 调整图像大小
CENTER_CROP = 224 # 中心裁剪大小 (通常等于预训练模型的输入大小)
# ImageNet 均值和标准差，用于归一化
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.405, 0.424, 0.425]

# --- 训练/拟合配置 ---
MVG_PARAMS_OUTPUT_DIR = "src/app/mvg_params" # 保存多元高斯参数的目录
BATCH_SIZE_FIT = 32 # 拟合高斯分布时的批处理大小

# --- 检测/推理配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 自动选择 GPU 或 CPU

# --- 可视化配置 ---
SCORE_MAP_THRESHOLD = 0.5 # 用于生成轮廓的异常得分阈值 (可能需要调整)