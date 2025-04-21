import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def normalize_score_map(score_map):
    """将得分图归一化到 [0, 1] 范围。"""
    min_val = np.min(score_map)
    max_val = np.max(score_map)
    if max_val > min_val:
        return (score_map - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(score_map)

def apply_heatmap(image_pil, score_map, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    将异常得分图作为热力图叠加到原始图像上。

    Args:
        image_pil (PIL.Image): 原始图像 (RGB)。
        score_map (np.ndarray): 异常得分图 (与图像大小相同)。
        alpha (float): 热力图的透明度。
        colormap (int): OpenCV 的颜色映射标识符。

    Returns:
        PIL.Image: 叠加了热力图的图像。
    """
    # 归一化得分图到 [0, 1]
    normalized_map = normalize_score_map(score_map)

    # 转换为 8 位整数并应用颜色映射
    heatmap_uint8 = (normalized_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) # 转为 RGB

    # 将 PIL 图像转为 OpenCV 格式 (RGB)
    image_cv = np.array(image_pil)

    # 检查尺寸是否一致
    if image_cv.shape[:2] != heatmap_color_rgb.shape[:2]:
         # 如果不一致，可能是上采样问题，尝试调整热力图尺寸
         heatmap_color_rgb = cv2.resize(heatmap_color_rgb, (image_cv.shape[1], image_cv.shape[0]))


    # 叠加热力图
    overlay = cv2.addWeighted(image_cv, 1 - alpha, heatmap_color_rgb, alpha, 0)

    return Image.fromarray(overlay) # 转回 PIL 图像

def draw_contours(image_pil, score_map, threshold=0.5, color=(0, 0, 255), thickness=5):
    """
    在图像上绘制异常区域的轮廓。

    Args:
        image_pil (PIL.Image): 输入图像 (可以是原始图像或已叠加热力图的图像)。
        score_map (np.ndarray): 异常得分图。
        threshold (float): 用于二值化得分图的阈值。
        color (tuple): 轮廓颜色 (B, G, R)。
        thickness (int): 轮廓线宽。

    Returns:
        PIL.Image: 绘制了轮廓的图像。
    """
    # 归一化并应用阈值生成二值掩码
    normalized_map = normalize_score_map(score_map)
    binary_mask = (normalized_map > threshold).astype(np.uint8) * 255

    # 将 PIL 图像转为 OpenCV 格式 (需要 BGR)
    image_cv_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


    # 查找轮廓
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓
    image_with_contours = cv2.drawContours(image_cv_bgr.copy(), contours, -1, color, thickness)

    # 转回 PIL 图像 (RGB)
    return Image.fromarray(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))

def add_score_text(image_pil, score, position=(10, 10), font_size=20, color=(255, 255, 255)):
    """
    在图像上添加异常得分文本。

    Args:
        image_pil (PIL.Image): 输入图像。
        score (float): 异常得分。
        position (tuple): 文本左上角坐标。
        font_size (int): 字体大小。
        color (tuple): 文本颜色 (R, G, B)。

    Returns:
        PIL.Image: 添加了文本的图像。
    """
    draw = ImageDraw.Draw(image_pil)
    text = f"Anomaly Score: {score:.4f}"
    try:
        # 尝试加载字体，如果失败则使用默认字体
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        print("Arial font not found, using default font.")
    draw.text(position, text, fill=color, font=font)
    return image_pil
