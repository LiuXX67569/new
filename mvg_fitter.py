import sys
import os
import argparse # 添加
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # 添加

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
# import torch.nn.functional as F # <--- 移除导入
import numpy as np
from torch.utils.data import DataLoader
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA # <--- 恢复导入
import os
import joblib # 用于保存/加载拟合参数
from torch.utils.data import Dataset as TorchDataset # 重命名导入以避免冲突

# 尝试从项目结构导入 MVTecAD 数据集类
try:
    # 假设 MVTecAD 类在 src/datasets/mvtecad.py 中
    from src.datasets.mvtecad import MVTecAD
except ImportError:
    print("Warning: Could not import MVTecAD from src.datasets.mvtecad.")
    print("Make sure the file exists and the src directory is in PYTHONPATH.")
    # 提供一个占位符或引发更严重的错误，取决于后续逻辑是否依赖它
    MVTecAD = None

from src.app.feature_extractor import FeatureExtractor
from src.app.config import (MVTEC_DATA_PATH, MVG_PARAMS_OUTPUT_DIR, BATCH_SIZE_FIT,
                     PRETRAINED_MODEL_NAME, FEATURE_LAYER_NAMES, DEVICE)

# --- 新增包装器类 --- #
class TransformWrapperDataset(TorchDataset):
    """一个简单的数据集包装器，用于应用 torchvision 变换。"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 从原始 Subset -> MVTecAD 获取样本字典
        sample_dict = self.dataset[idx]

        # 从字典中提取图像 (假设键是 'image')
        # MVTecAD 可能返回 PIL Image 或 numpy array
        image = sample_dict.get('image')
        if image is None:
             raise ValueError(f"Sample dictionary at index {idx} does not contain an 'image' key or value is None.")

        # 将图像转换为 PIL RGB 格式 (如果需要)
        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image).convert('RGB')
        elif hasattr(image, 'convert') and callable(image.convert):
            # 确保是 RGB 格式
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            raise TypeError(f"Unsupported image type {type(image)} in sample dictionary.")

        # 应用 torchvision 变换
        transformed_image = self.transform(image)

        # 对于拟合高斯分布，我们通常只需要图像
        # 如果需要标签或其他信息，可以从 sample_dict 中提取并返回
        # return transformed_image, sample_dict.get('label')
        return transformed_image # 只返回变换后的图像
# --- 结束新增包装器类 --- #

def fit_gaussian_distribution(category):
    """
    为指定 MVTec AD 类别拟合多元高斯分布。

    Args:
        category (str): MVTec AD 数据集的类别名称 (例如 'bottle')。
    """
    print(f"\n--- Fitting Gaussian for category: {category} ---")

    # 1. 初始化特征提取器
    feature_extractor = FeatureExtractor(model_name=PRETRAINED_MODEL_NAME,
                                         layer_names=FEATURE_LAYER_NAMES,
                                         device=DEVICE)
    # 获取与模型匹配的预处理转换
    preprocessing_transforms = feature_extractor.get_preprocessing()
    print("Using preprocessing transforms:", preprocessing_transforms)
    # --- 恢复: 移除验证逻辑 ---
    # --- 结束恢复 ---

    # 2. 加载数据集 (仅限正常训练样本)
    if MVTecAD is None:
        raise RuntimeError("MVTecAD dataset class is not available.")

    try:
        mock_hparams = argparse.Namespace()
        mock_hparams.category = category

        try:
            original_dataset = MVTecAD(root=MVTEC_DATA_PATH,
                                       hparams=mock_hparams,
                                       train=True
                                      )
            good_indices = [i for i, sample in enumerate(original_dataset.samples)
                            if original_dataset.get_target(i, transform=False) == 'good']

            if not good_indices:
                 print(f"Error: No 'good' training samples found for category '{category}' after loading. Check dataset structure or MVTecAD implementation.")
                 return

            from torch.utils.data import Subset
            dataset_subset = Subset(original_dataset, good_indices)
            dataset = TransformWrapperDataset(dataset_subset, preprocessing_transforms)

            print(f"Loaded, filtered, and wrapped dataset for category '{category}'. Number of 'good' training samples: {len(dataset)}")

        except Exception as e:
            print(f"Error loading or filtering dataset for category {category}: {e}")
            import traceback
            traceback.print_exc()
            print("Please ensure MVTecAD dataset class and filtering logic are correct.")
            return

    except Exception as e:
        print(f"Error loading dataset for category {category}: {e}")
        print("Please ensure MVTecAD dataset class in src/datasets/mvtecad.py is correctly implemented and supports filtering.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_FIT, shuffle=False, num_workers=0)

    # 3. 提取特征 (恢复按层提取)
    all_features = {layer: [] for layer in FEATURE_LAYER_NAMES} # <--- 恢复按层字典
    # collected_aggregated_features = [] # <--- 移除
    # target_spatial_size = None # <--- 移除

    print("Extracting features...") # <--- 恢复原始打印
    with torch.no_grad():
        # --- 恢复: 移除确定目标尺寸的循环 --- #
        # --- 结束恢复 ---

        # --- 恢复: 单个循环，按层收集特征 --- #
        for batch in dataloader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(DEVICE)

            features = feature_extractor(images)

            # --- 恢复: 移除特征聚合逻辑 --- #
            # --- 结束恢复 ---

            # --- 恢复: 按层展平并收集到 all_features 字典 --- #
            for layer_name in FEATURE_LAYER_NAMES:
                feature_map = features[layer_name].cpu()
                b, c, h, w = feature_map.shape
                # 重塑为 (batch_size * num_patches, feature_dim)
                feature_vectors = feature_map.permute(0, 2, 3, 1).reshape(b * h * w, c)
                all_features[layer_name].append(feature_vectors)
            # --- 结束恢复 ---

    # 拼接所有批次的特征 (恢复按层拼接)
    # all_aggregated_features_torch = torch.cat(collected_aggregated_features, dim=0) # <--- 移除
    # print(f"Total aggregated features shape: {all_aggregated_features_torch.shape}")
    for layer_name in FEATURE_LAYER_NAMES:
        all_features[layer_name] = torch.cat(all_features[layer_name], dim=0).numpy() # (N*H*W, C)
        print(f"Layer '{layer_name}' features shape: {all_features[layer_name].shape}")
    # --- 结束恢复 ---

    # --- 恢复: 移除 PCA 降维 --- #
    # print("Performing PCA...")
    # ... (移除 PCA 相关代码) ...
    # --- 结束恢复 ---

    # --- 恢复: 按层拟合 MVG --- #
    mvg_parameters = {} # <--- 恢复为空字典
    pca_models = {} # 新增: 用于存储 PCA 模型
    print("Fitting Gaussian distributions...") # <--- 恢复原始打印
    # --- 恢复: 循环处理每一层特征 --- #
    for layer_name in FEATURE_LAYER_NAMES:
        features_np = all_features[layer_name]
        if features_np.shape[0] == 0:
            print(f"Error: No features extracted for layer {layer_name}. Skipping.")
            continue

        # --- 新增/恢复: PCA 降维 --- #
        print(f"Performing PCA for layer {layer_name}...")
        # TODO: 考虑实现 Negated PCA (NPCA) - 选择方差最小的主成分
        pca = PCA(n_components=0.99) # 保留 99% 的方差，或选择固定数量的主成分
        try:
            features_reduced = pca.fit_transform(features_np)
            print(f"Layer '{layer_name}' PCA complete. Original dim: {features_np.shape[1]}, Reduced dim: {features_reduced.shape[1]}")
            pca_models[layer_name] = pca # 保存 PCA 模型
            features_np = features_reduced # 使用降维后的特征进行后续处理
        except Exception as e:
            print(f"Error performing PCA for layer {layer_name}: {e}")
            print("Skipping PCA for this layer.")
            pca_models[layer_name] = None # 标记 PCA 未成功执行
        # --- 结束 PCA 降维 ---

        # 使用 LedoitWolf 进行协方差估计
        print(f"Fitting LedoitWolf covariance for layer {layer_name}...") # <--- 恢复带层名的打印
        try:
            cov_estimator = LedoitWolf()
            cov_estimator.fit(features_np) # 使用降维后的特征 (如果 PCA 成功)

            mean = cov_estimator.location_ # 均值向量 (C',)
            precision = cov_estimator.get_precision() # 精度矩阵 (C', C')

            # --- 恢复: 按层存储 MVG 参数 --- #
            # --- 修改: 同时存储 PCA 模型 --- #
            mvg_parameters[layer_name] = {
                'mean': mean,
                'precision': precision,
                'pca_model': pca_models[layer_name] # 存储对应的 PCA 模型 (可能为 None)
            }
            print(f"Layer '{layer_name}' MVG fitting complete. Mean shape: {mean.shape}, Precision shape: {precision.shape}")
        except Exception as e:
            print(f"Error fitting MVG for layer {layer_name}: {e}")
            print("Consider adding regularization or checking feature values.")
    # --- 结束恢复 --- #

    # --- 恢复: 保存按层参数的文件 --- #
    if not mvg_parameters:
        print("No MVG parameters were successfully fitted. Aborting save.")
        return

    os.makedirs(MVG_PARAMS_OUTPUT_DIR, exist_ok=True)
    # --- 恢复: 使用原始文件名格式 --- #
    save_path = os.path.join(MVG_PARAMS_OUTPUT_DIR, f"{category}_{PRETRAINED_MODEL_NAME}_mvg_params.joblib")
    try:
        joblib.dump(mvg_parameters, save_path)
        print(f"MVG parameters saved to: {save_path}") # <--- 恢复原始打印
    except Exception as e:
        print(f"Error saving MVG parameters: {e}")

if __name__ == '__main__':
    from src.app.config import MVTEC_CATEGORIES, MVG_PARAMS_OUTPUT_DIR, PRETRAINED_MODEL_NAME

    while True:
        untrained_categories = []
        trained_categories = []

        # --- 恢复: 检查原始参数文件名 --- #
        for category in MVTEC_CATEGORIES:
            params_filename = f"{category}_{PRETRAINED_MODEL_NAME}_mvg_params.joblib" # <--- 恢复原始文件名
            params_path = os.path.join(MVG_PARAMS_OUTPUT_DIR, params_filename)
            if os.path.exists(params_path):
                trained_categories.append(category)
            else:
                untrained_categories.append(category)

        if not untrained_categories:
            print("\n所有类别都已完成训练！") # <--- 恢复原始打印
            break

        print("\n-------------------------------------")
        if trained_categories:
            print(f"已完成训练的类别: {', '.join(trained_categories)}") # <--- 恢复原始打印
        print("请选择要训练的类别:") # <--- 恢复原始打印
        for i, category in enumerate(untrained_categories):
            print(f"  {i + 1}: {category}")
        print("\n或者输入:")
        print("  a / all: 训练所有剩余类别")
        print("  q / quit: 退出程序")
        print("-------------------------------------")

        choice = input("请输入选项: ").strip().lower()

        if choice == 'q' or choice == 'quit':
            break
        elif choice == 'a' or choice == 'all':
            categories_to_train = untrained_categories
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(untrained_categories):
                    categories_to_train = [untrained_categories[idx]]
                else:
                    print("无效的选项，请重试。")
                    continue
            except ValueError:
                print("无效的输入，请输入数字、'a' 或 'q'。")
                continue

        for cat in categories_to_train:
            fit_gaussian_distribution(cat)
