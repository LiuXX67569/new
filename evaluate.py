import os
import sys
import argparse
import json
import numpy as np
from tqdm import tqdm # 用于显示进度条
from sklearn.metrics import roc_auc_score
import time # 添加 time 模块用于计时

# Add project root to the Python path (if running script directly)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 在导入自定义模块前设置环境变量 (临时解决方案)
# 如果不设置，并且在其他地方也没有设置，可能会遇到 OMP 错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 尝试从项目结构导入所需模块
try:
    from src.datasets.mvtecad import MVTecAD
except ImportError:
    print("错误: 无法从 src.datasets.mvtecad 导入 MVTecAD。")
    print("请确保文件存在且 src 目录在 PYTHONPATH 中。")
    sys.exit(1)

try:
    from src.app.detector import AnomalyDetector # <--- 恢复 Detector 导入
    from src.app.config import (MVTEC_DATA_PATH, MVTEC_CATEGORIES,
                                MVG_PARAMS_OUTPUT_DIR, PRETRAINED_MODEL_NAME)
except ImportError as e:
     print(f"错误: 无法导入必要的 app 模块: {e}")
     print("请确保在项目根目录下运行，或者 src 目录在 PYTHONPATH 中。")
     sys.exit(1)

def evaluate_category(category):
    """
    恢复: 评估指定 MVTec AD 类别的异常检测性能 (图像和像素级 AUROC) 及速度。

    Args:
        category (str): MVTec AD 数据集的类别名称。

    Returns:
        dict: 包含评估指标的字典 (image_auroc, pixel_auroc, avg_time_ms, fps)。
              如果评估失败，则返回 None。
    """
    print(f"\n--- Evaluating category: {category} ---") # <--- 恢复原始打印

    # --- 恢复: 加载测试数据集，仅需路径和标签/掩码 --- #
    true_labels = []
    image_paths = [] # <--- 恢复使用路径
    true_masks = [] # <--- 恢复原始名称

    try:
        mock_hparams = argparse.Namespace()
        mock_hparams.category = category
        test_dataset = MVTecAD(root=MVTEC_DATA_PATH,
                               hparams=mock_hparams,
                               train=False,
                               load_masks=True,
                               transform=None,
                               cache=False)

        if len(test_dataset) == 0:
            print(f"错误: 未找到类别 '{category}' 的测试样本。")
            return None

        print(f"加载测试样本及掩码 for category '{category}'...")
        for i in tqdm(range(len(test_dataset)), desc="Collecting test samples & masks"):
            try:
                 # --- 恢复: 从 test_dataset.samples 获取路径和类名 --- #
                 sample_info = test_dataset.samples[i]
                 img_path = sample_info[0]
                 class_name = sample_info[1]
                 label = test_dataset.target_transform(class_name) # 使用 target_transform

                 mask = None
                 if test_dataset.load_masks: # 检查是否加载了掩码
                     try:
                         mask = test_dataset.get_mask(i)
                     except Exception as mask_e:
                         print(f"警告: 获取索引 {i} 的掩码时出错: {mask_e}. 将视为空掩码。")
                         # 可以在这里决定如何处理，例如设为None或创建零掩码
                         # mask = None # 保持 None

                 image_paths.append(img_path)
                 true_labels.append(label)
                 true_masks.append(mask)
                 # --- 结束恢复 ---

            except IndexError:
                 print(f"错误: 无法访问索引 {i} 处的样本信息。")
                 return None
            except Exception as e:
                 print(f"在索引 {i} 处加载样本/掩码时发生未知错误: {e}")
                 return None

        print(f"收集完成: {len(image_paths)} 个测试样本，{sum(true_labels)} 个异常样本。")
        if test_dataset.load_masks and not any(m is not None for m in true_masks):
             print("警告: 未能加载任何有效的真实掩码。像素级 AUROC 将无法计算。")

    except Exception as e:
        print(f"加载类别 '{category}' 的测试数据集或掩码时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- 恢复: 初始化 AnomalyDetector --- #
    try:
        detector = AnomalyDetector(category=category)
    except FileNotFoundError:
        # --- 恢复: 检查原始参数文件名 --- #
        print(f"错误: 未找到类别 '{category}' 的 MVG 参数文件 ({PRETRAINED_MODEL_NAME})。")
        print(f"请先运行 'python src/app/mvg_fitter.py' for category '{category}'.")
        return None
    except Exception as e:
        print(f"初始化类别 '{category}' 的检测器时出错: {e}")
        return None
    # --- 结束恢复 ---

    # --- 恢复: 使用 detector.detect() 获取分数并计时 --- #
    predicted_scores = [] # <--- 恢复原始名称
    predicted_score_maps = [] # <--- 恢复原始名称
    total_inference_time = 0.0
    valid_detection_count = 0

    print("开始检测测试图像并计时...") # <--- 恢复原始打印
    start_time_total = time.perf_counter()

    for idx, img_path in enumerate(tqdm(image_paths, desc=f"Detecting {category}")):
        start_time_single = time.perf_counter()
        try:
            # --- 恢复: 调用 detector.detect --- #
            image_score, score_map = detector.detect(img_path)
            end_time_single = time.perf_counter()

            if image_score is None or score_map is None:
                print(f"警告: 检测图像 {os.path.basename(img_path)} 时返回 None。跳过该样本。")
                predicted_scores.append(np.nan)
                predicted_score_maps.append(None)
            else:
                predicted_scores.append(image_score)
                predicted_score_maps.append(score_map)
                total_inference_time += (end_time_single - start_time_single)
                valid_detection_count += 1
            # --- 结束恢复 ---
        except Exception as e:
            print(f"检测图像 {os.path.basename(img_path)} 时出错: {e}。跳过该样本。")
            predicted_scores.append(np.nan)
            predicted_score_maps.append(None)

    end_time_total = time.perf_counter()
    overall_duration = end_time_total - start_time_total
    # --- 恢复: 原始有效计数检查 --- #
    if valid_detection_count == 0:
        print("错误: 所有样本检测均失败，无法计算指标。")
        return None
    print(f"检测完成。有效检测数: {valid_detection_count}/{len(image_paths)}.")
    print(f"总检测时间 (包括加载等): {overall_duration:.2f} 秒")
    print(f"累积纯推理时间: {total_inference_time:.2f} 秒")

    # 计算速度指标
    avg_time_ms = (total_inference_time * 1000 / valid_detection_count)
    fps = valid_detection_count / total_inference_time
    print(f"平均推理速度: {avg_time_ms:.2f} ms/image")
    print(f"帧率 (FPS): {fps:.2f}")

    # 清理无效样本
    valid_indices = [i for i, score in enumerate(predicted_scores) if not np.isnan(score)]
    if len(valid_indices) != len(true_labels):
        print(f"警告: {len(true_labels) - len(valid_indices)} 个样本未能成功计算得分，已从评估中移除。")
        true_labels = [true_labels[i] for i in valid_indices]
        predicted_scores = [predicted_scores[i] for i in valid_indices]
        true_masks = [true_masks[i] for i in valid_indices]
        predicted_score_maps = [predicted_score_maps[i] for i in valid_indices]

    if len(true_labels) < 2:
         print(f"错误: 有效样本不足 (<2)，无法计算 AUROC。")
         return None

    # --- 恢复: 计算图像级 AUROC --- #
    image_auroc = -1.0
    if len(set(true_labels)) < 2:
        print(f"警告: 有效样本只包含一个类别 ({set(true_labels)})，无法计算图像级 AUROC。")
    else:
        try:
            image_auroc = roc_auc_score(true_labels, predicted_scores)
            print(f"Category '{category}' - Image-Level AUROC: {image_auroc:.6f}")
        except Exception as e:
            print(f"计算类别 '{category}' 的图像级 AUROC 时出错: {e}")
    # --- 结束恢复 ---

    # --- 恢复: 计算像素级 AUROC --- #
    pixel_auroc = -1.0
    flat_true_masks = []
    flat_predicted_scores = []
    has_valid_masks = False
    print("准备像素级评估数据...")
    for i in tqdm(range(len(true_masks)), desc="Flattening masks & scores"):
        true_mask = true_masks[i]
        score_map = predicted_score_maps[i]

        if score_map is None: # 跳过检测失败的样本
             continue

        if true_mask is None: # 处理正常样本的掩码 (可能是 None 或全零)
            # 如果 AnomalyDetector 返回的 score_map 不为 None，我们需要一个全零掩码进行比较
            true_mask_flat = np.zeros_like(score_map).flatten().astype(np.uint8)
        else:
            if true_mask.max() > 1:
                 true_mask = (true_mask > 127).astype(np.uint8)

            if true_mask.shape != score_map.shape:
                 import cv2
                 # --- 恢复: 调整大小可能仍需，但基于原始 score_map --- #
                 try:
                     score_map = cv2.resize(score_map, (true_mask.shape[1], true_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
                 except Exception as resize_e:
                     print(f"警告: 调整 score_map 大小失败 (索引 {i}): {resize_e}. 跳过此像素评估样本。")
                     continue # 跳到下一个样本
            true_mask_flat = true_mask.flatten()
            has_valid_masks = True # 标记至少有一个有效的非空掩码

        flat_true_masks.append(true_mask_flat)
        flat_predicted_scores.append(score_map.flatten())

    if not flat_true_masks or not flat_predicted_scores:
        print("警告: 像素级标签或得分列表为空。无法计算像素级 AUROC。")
    # 检查拼接前是否有有效掩码且至少有两个类别 (0 和 1)
    elif not has_valid_masks and len(set(np.concatenate(flat_true_masks))) < 2:
        print(f"警告: 仅存在正常样本的掩码或只有一类像素标签，无法计算像素级 AUROC。")
    else:
        try:
            all_true_pixels = np.concatenate(flat_true_masks)
            all_pred_pixels = np.concatenate(flat_predicted_scores)

            if len(all_true_pixels) == 0:
                print("错误: 展平后的像素标签为空。")
            elif len(set(all_true_pixels)) < 2:
                print(f"警告: 所有有效像素标签都属于同一类别 ({set(all_true_pixels)})。无法计算像素级 AUROC。")
            else:
                pixel_auroc = roc_auc_score(all_true_pixels, all_pred_pixels)
                print(f"Category '{category}' - Pixel-Level AUROC: {pixel_auroc:.6f}")
        except Exception as e:
            print(f"计算类别 '{category}' 的像素级 AUROC 时出错: {e}")
    # --- 结束恢复 ---

    # --- 恢复: 准备返回结果 --- #
    results = {
        "image_auroc": image_auroc if image_auroc != -1.0 else None,
        "pixel_auroc": pixel_auroc if pixel_auroc != -1.0 else None,
        "avg_inference_time_ms": avg_time_ms,
        "fps": fps
    }
    print(f"Category '{category}' Evaluation Summary: {results}")
    return results

if __name__ == '__main__':
    # --- 恢复: 原始参数解析 --- #
    parser = argparse.ArgumentParser(description="评估 MVTec AD 异常检测模型的 AUROC")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=MVTEC_CATEGORIES + [None], # 允许不指定，则评估所有
        help="要评估的 MVTec AD 类别。如果未指定，则评估所有类别。"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="src/app/evaluation_results.json",
        help="保存评估结果的 JSON 文件路径。"
    )
    args = parser.parse_args()

    all_results = {}

    if args.category:
        categories_to_evaluate = [args.category]
    else:
        categories_to_evaluate = MVTEC_CATEGORIES
        print(f"未指定特定类别，将评估所有 {len(MVTEC_CATEGORIES)} 个类别...")

    # --- 恢复: 原始主循环逻辑 --- #
    evaluation_successful = True
    start_time_overall = time.perf_counter()
    for category in categories_to_evaluate:
        # 检查原始参数文件是否存在
        params_path = os.path.join(MVG_PARAMS_OUTPUT_DIR, f"{category}_{PRETRAINED_MODEL_NAME}_mvg_params.joblib")
        if not os.path.exists(params_path):
             print(f"警告: 未找到类别 '{category}' 的 MVG 参数文件 ({params_path})。跳过评估。")
             print(f"请先运行 'python src/app/mvg_fitter.py' for category '{category}'.")
             all_results[category] = None # 标记为未评估
             continue

        category_result = evaluate_category(category)
        if category_result is None:
            all_results[category] = None # 标记为评估失败
            evaluation_successful = False
        else:
            all_results[category] = category_result

    end_time_overall = time.perf_counter()
    total_eval_time = end_time_overall - start_time_overall
    print(f"\n--- Evaluation Complete ---")
    print(f"总评估时间: {total_eval_time:.2f} 秒")

    # 保存结果到 JSON 文件
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"评估结果已保存到: {args.output_file}")
    except Exception as e:
        print(f"保存评估结果到 JSON 时出错: {e}")
        evaluation_successful = False

    # 打印总结 (可选)
    print("\n--- Summary ---")
    avg_img_auroc = np.nanmean([res.get('image_auroc', np.nan) for res in all_results.values() if res and res.get('image_auroc') is not None])
    avg_pxl_auroc = np.nanmean([res.get('pixel_auroc', np.nan) for res in all_results.values() if res and res.get('pixel_auroc') is not None])
    avg_fps = np.nanmean([res.get('fps', np.nan) for res in all_results.values() if res])
    print(f"Average Image AUROC: {avg_img_auroc:.6f}")
    print(f"Average Pixel AUROC: {avg_pxl_auroc:.6f}")
    print(f"Average FPS: {avg_fps:.2f}")

    if not evaluation_successful:
        print("\n部分类别的评估或结果保存失败。请检查上面的日志。")
        sys.exit(1)
    else:
        print("\n所有指定类别的评估已完成。")
