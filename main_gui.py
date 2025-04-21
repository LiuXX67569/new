import os # 添加
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # 添加

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading # 用于在后台线程中运行检测，防止 GUI 冻结
import sys # 添加

# Add project root to the Python path # 添加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # 添加
if project_root not in sys.path: # 添加
    sys.path.insert(0, project_root) # 添加

from src.app.config import MVTEC_CATEGORIES, SCORE_MAP_THRESHOLD # 修改
from src.app.detector import AnomalyDetector # 修改
from src.app.visualizer import apply_heatmap, draw_contours, add_score_text # 修改

class AnomalyDetectionGUI:
    """
    异常检测系统的图形用户界面。
    """
    def __init__(self, root):
        self.root = root
        self.root.title("图像异常检测系统 (MVG 方法)")
        self.root.geometry("800x600") # 设置初始窗口大小

        self.detector = None # 检测器将在选择类别后初始化
        self.image_path = None
        self.original_pil_image = None
        self.result_pil_image = None

        # --- GUI 组件 ---
        # 框架
        top_frame = ttk.Frame(root, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)

        image_frame = ttk.Frame(root, padding="10")
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 类别选择
        ttk.Label(top_frame, text="选择类别:").pack(side=tk.LEFT, padx=(0, 5))
        self.category_var = tk.StringVar()
        self.category_combobox = ttk.Combobox(top_frame, textvariable=self.category_var,
                                              values=MVTEC_CATEGORIES, state="readonly")
        self.category_combobox.pack(side=tk.LEFT, padx=(0, 10))
        self.category_combobox.bind("<<ComboboxSelected>>", self.on_category_select)
        if MVTEC_CATEGORIES:
            self.category_combobox.current(0) # 默认选择第一个

        # 上传按钮
        self.upload_button = ttk.Button(top_frame, text="上传图像", command=self.upload_image)
        self.upload_button.pack(side=tk.LEFT, padx=(0, 10))
        self.upload_button["state"] = "disabled" # 初始禁用，选择类别后启用

        # 状态标签
        self.status_label = ttk.Label(top_frame, text="请先选择一个类别。")
        self.status_label.pack(side=tk.LEFT)

        # 图像显示区域 (左右分割)
        self.left_canvas = tk.Canvas(image_frame, bg='lightgrey')
        self.left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.left_label = ttk.Label(image_frame, text="原始图像")
        self.left_label.pack(side=tk.LEFT, anchor='n') # 需要调整布局使标签在 Canvas 上方
        # TODO: Improve label placement above canvas

        self.right_canvas = tk.Canvas(image_frame, bg='lightgrey')
        self.right_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.right_label = ttk.Label(image_frame, text="检测结果")
        self.right_label.pack(side=tk.RIGHT, anchor='n')
        # TODO: Improve label placement above canvas


        # 初始化时尝试加载第一个类别的检测器
        if MVTEC_CATEGORIES:
             self.on_category_select()


    def on_category_select(self, event=None):
        """当用户选择一个新类别时调用。"""
        selected_category = self.category_var.get()
        if not selected_category:
            return

        self.status_label.config(text=f"正在加载类别 '{selected_category}' 的模型...")
        self.root.update_idletasks() # 强制更新 GUI

        try:
            # 在后台线程加载，避免卡顿 (如果加载慢的话)
            # threading.Thread(target=self._load_detector_thread, args=(selected_category,), daemon=True).start()
            # 简单起见，先直接加载
            self.detector = AnomalyDetector(category=selected_category)
            self.status_label.config(text=f"类别 '{selected_category}' 加载完成。请上传图像。")
            self.upload_button["state"] = "normal" # 启用上传按钮
        except FileNotFoundError as e:
            self.status_label.config(text=f"错误: 未找到 '{selected_category}' 的参数文件。请先运行拟合。")
            messagebox.showerror("加载错误", str(e))
            self.upload_button["state"] = "disabled"
            self.detector = None
        except Exception as e:
            self.status_label.config(text=f"加载类别 '{selected_category}' 时出错。")
            messagebox.showerror("加载错误", f"加载模型时发生未知错误: {e}")
            self.upload_button["state"] = "disabled"
            self.detector = None

    def upload_image(self):
        """打开文件对话框让用户选择图像并开始检测。"""
        if not self.detector:
            messagebox.showwarning("未加载模型", "请先成功选择一个类别以加载模型。")
            return

        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]
        )
        if not file_path:
            return

        self.image_path = file_path
        self.status_label.config(text=f"正在检测图像: {os.path.basename(file_path)}...")
        self.root.update_idletasks()

        # 在后台线程中运行检测
        threading.Thread(target=self._detect_thread, daemon=True).start()

    def _detect_thread(self):
        """在后台线程中执行异常检测。"""
        try:
            image_score, score_map = self.detector.detect(self.image_path)

            if image_score is None or score_map is None:
                self.root.after(0, self._update_status, "检测失败。请查看控制台输出。")
                return

            # 检测完成，现在进行可视化
            self.original_pil_image = Image.open(self.image_path).convert('RGB')

            # 1. 创建热力图叠加图像
            heatmap_image = apply_heatmap(self.original_pil_image, score_map)

            # 2. (可选) 在热力图上绘制轮廓
            contour_image = draw_contours(heatmap_image, score_map, threshold=SCORE_MAP_THRESHOLD)

            # 3. 在最终图像上添加得分文本
            self.result_pil_image = add_score_text(contour_image, image_score) # 或者在 heatmap_image 上添加

            # 操作完成，回到主线程更新 GUI
            self.root.after(0, self._display_results)

        except Exception as e:
            print(f"Detection thread error: {e}") # 打印详细错误到控制台
            import traceback
            traceback.print_exc()
            self.root.after(0, self._update_status, f"检测时发生错误: {e}")

    def _display_results(self):
        """在 GUI 中显示原始图像和检测结果图像。"""
        if self.original_pil_image and self.result_pil_image:
            self._display_image_on_canvas(self.left_canvas, self.original_pil_image)
            self._display_image_on_canvas(self.right_canvas, self.result_pil_image)
            self.status_label.config(text="检测完成。")
        else:
             self.status_label.config(text="无法显示结果。")


    def _display_image_on_canvas(self, canvas, pil_image):
         """调整图像大小以适应画布并显示。"""
         canvas.delete("all") # 清除旧图像
         canvas_width = canvas.winfo_width()
         canvas_height = canvas.winfo_height()

         if canvas_width <= 1 or canvas_height <= 1: # 画布尚未完全渲染
              self.root.after(100, lambda: self._display_image_on_canvas(canvas, pil_image)) # 稍后重试
              return

         img_copy = pil_image.copy()
         img_copy.thumbnail((canvas_width - 10, canvas_height - 10), Image.Resampling.LANCZOS) # 保持纵横比缩放

         # 必须保持对 PhotoImage 的引用，否则会被垃圾回收
         photo_image = ImageTk.PhotoImage(img_copy)
         canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=photo_image)
         canvas.image = photo_image # 保持引用

    def _update_status(self, message):
         """安全地从后台线程更新状态标签。"""
         self.status_label.config(text=message)


if __name__ == '__main__':
    # --- 如何运行 ---
    # 1. 确保 MVTec AD 数据集已下载并放置在 config.py 中指定的路径。
    # 2. 运行 `python -m src.app.mvg_fitter` 为所需的类别拟合高斯模型。
    #    (确保在项目根目录 new/ 下运行此命令)
    #    这会在 src/app/mvg_params/ 目录下生成 .joblib 文件。
    # 3. 运行 `python -m src.app.main_gui` 启动 GUI 应用程序。
    #    (同样在 new/ 目录下运行)

    print("启动 GUI 应用程序...")
    print("运行步骤:")
    print("1. 确保 MVTec AD 数据集路径在 src/app/config.py 中设置正确。")
    print("2. (如果尚未完成) 在项目根目录运行: python -m src.app.mvg_fitter")
    print("3. 在项目根目录运行此脚本: python -m src.app.main_gui")

    root = tk.Tk()
    app = AnomalyDetectionGUI(root)
    root.mainloop()
