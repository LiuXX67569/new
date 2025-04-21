import torch
import numpy as np
import cv2
from PIL import Image
import os
import joblib
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter

from src.app.feature_extractor import FeatureExtractor
from src.app.config import (MVG_PARAMS_OUTPUT_DIR, PRETRAINED_MODEL_NAME,
                     FEATURE_LAYER_NAMES, DEVICE, IMAGE_SIZE)

class AnomalyDetector:
    """
    使用预训练模型提取的特征和拟合的高斯分布进行异常检测。
    """
    def __init__(self, category, device=DEVICE):
        """
        初始化异常检测器。

        Args:
            category (str): 要检测的 MVTec AD 类别。
            device (torch.device): 运行模型的设备。
        """
        self.category = category
        self.device = device

        # 1. 加载特征提取器
        self.feature_extractor = FeatureExtractor(model_name=PRETRAINED_MODEL_NAME,
                                                  layer_names=FEATURE_LAYER_NAMES,
                                                  device=self.device)
        self.preprocessing_transforms = self.feature_extractor.get_preprocessing()

        # 2. 加载预计算的 MVG 参数
        self.mvg_params = self._load_mvg_params()

    def _load_mvg_params(self):
        """
        加载指定类别的预计算 MVG 参数。
        """
        params_path = os.path.join(MVG_PARAMS_OUTPUT_DIR, f"{self.category}_{PRETRAINED_MODEL_NAME}_mvg_params.joblib")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"MVG parameters not found for category '{self.category}' at {params_path}. "
                                    f"Please run mvg_fitter.py first.")
        try:
            mvg_params = joblib.load(params_path)
            print(f"Loaded MVG parameters from: {params_path}")
            # 验证加载的参数是否包含所有需要的层和键
            for layer in FEATURE_LAYER_NAMES:
                if layer not in mvg_params:
                     raise ValueError(f"Loaded parameters missing expected layer: {layer}")
                # --- 修改: 检查 pca_model 是否存在 --- #
                required_keys = ['mean', 'precision', 'pca_model']
                for key in required_keys:
                    if key not in mvg_params[layer]:
                        raise ValueError(f"Parameters for layer {layer} missing key: '{key}'")
                # 如果 pca_model 是 None (表示拟合时 PCA 失败)，允许，但需要处理
                if mvg_params[layer]['pca_model'] is None:
                     print(f"Warning: PCA model for layer {layer} is None. Will skip PCA transform for this layer during detection.")
            return mvg_params
        except Exception as e:
            print(f"Error loading MVG parameters from {params_path}: {e}")
            raise

    def detect(self, image_path):
        """
        对单个图像执行异常检测。

        Args:
            image_path (str): 输入图像的文件路径。

        Returns:
            tuple: (image_score, score_map)
                - image_score (float): 图像级的异常得分。
                - score_map (np.ndarray): 像素级的异常得分图 (与输入图像大小相同)。
        """
        # 1. 加载和预处理图像
        try:
            img = Image.open(image_path).convert('RGB')
            input_tensor = self.preprocessing_transforms(img).unsqueeze(0) # 添加 batch 维度
            input_tensor = input_tensor.to(self.device)
            original_size = img.size # (宽度, 高度)
        except Exception as e:
            print(f"Error loading or preprocessing image {image_path}: {e}")
            return None, None

        # 2. 提取特征
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        # 3. 计算马氏距离并生成得分图
        score_maps_per_layer = []
        for layer_name in FEATURE_LAYER_NAMES:
            if layer_name not in self.mvg_params:
                print(f"Warning: Skipping layer {layer_name} as no MVG params found.")
                continue

            # 获取特征图和对应的 MVG 参数
            feature_map = features[layer_name].cpu().numpy() # (1, C, H, W)
            params = self.mvg_params[layer_name]
            mean = params['mean']
            precision = params['precision']
            pca_model = params['pca_model'] # <-- 加载 PCA 模型

            # 调整特征图形状: (1, C, H, W) -> (H*W, C)
            _, c, h, w = feature_map.shape
            feature_vectors = feature_map.transpose(0, 2, 3, 1).reshape(h * w, c)

            # --- 新增: 应用 PCA 变换 (如果 PCA 模型存在) --- #
            if pca_model is not None:
                try:
                    feature_vectors = pca_model.transform(feature_vectors)
                    # print(f"Layer {layer_name}: Applied PCA transform. New shape: {feature_vectors.shape}") # 调试信息
                except Exception as e:
                    print(f"Error applying PCA transform for layer {layer_name}: {e}. Skipping distance calculation for this layer.")
                    continue # 跳过此层
            else:
                print(f"Skipping PCA transform for layer {layer_name} as model was None.")
            # --- 结束 PCA 变换 ---

            # 计算每个特征向量的马氏距离平方
            distances = np.zeros(h * w)
            try:
                # 确保维度匹配 (PCA 后维度可能变化)
                if feature_vectors.shape[1] != mean.shape[0]:
                     raise ValueError(f"Dimension mismatch after PCA for layer {layer_name}. Features: {feature_vectors.shape[1]}, Mean: {mean.shape[0]}")

                # 逐点计算马氏距离 (使用 PCA 变换后的特征)
                for i in range(h * w):
                    diff = feature_vectors[i] - mean
                    dist_sq = diff @ precision @ diff.T
                    distances[i] = dist_sq
            except Exception as e:
                 print(f"Error calculating Mahalanobis distance for layer {layer_name} (potentially after PCA): {e}")
                 continue # 跳过此层

            # 将距离重塑回空间得分图 (H, W)
            score_map_layer = distances.reshape(h, w)

            # 对得分图进行上采样到原始图像大小
            score_map_resized = cv2.resize(score_map_layer, original_size, interpolation=cv2.INTER_LINEAR)
            score_maps_per_layer.append(score_map_resized)

        if not score_maps_per_layer:
            print("Error: No score maps generated from any layer.")
            return None, None

        # 4. 融合不同层的得分图 (简单平均)
        final_score_map = np.mean(np.stack(score_maps_per_layer, axis=0), axis=0)

        # 5. (可选) 对最终得分图进行平滑处理
        final_score_map = gaussian_filter(final_score_map, sigma=4)

        # 6. 计算图像级得分 (例如，得分图的最大值)
        image_score = np.max(final_score_map)

        # 归一化得分图到 [0, 1] (可选，便于可视化)
        min_val = np.min(final_score_map)
        max_val = np.max(final_score_map)
        if max_val > min_val:
            final_score_map_normalized = (final_score_map - min_val) / (max_val - min_val)
        else:
            final_score_map_normalized = np.zeros_like(final_score_map)


        print(f"Detection complete for {image_path}. Image score: {image_score:.4f}")
        # 返回原始得分图，归一化可以在可视化时进行
        return image_score, final_score_map
