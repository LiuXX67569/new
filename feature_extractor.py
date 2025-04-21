import torch
import torch.nn as nn
import torchvision.models as models
from .config import PRETRAINED_MODEL_NAME, FEATURE_LAYER_NAMES, DEVICE

# --- 新增: 尝试导入 timm --- #
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("Warning: timm library not found. EfficientNet support requires timm. Please install it: pip install timm")
# --- 结束新增 --- #

class FeatureExtractor(nn.Module):
    """
    用于从预训练模型中提取指定中间层特征的类。
    """
    def __init__(self, model_name=PRETRAINED_MODEL_NAME, layer_names=FEATURE_LAYER_NAMES, device=DEVICE):
        """
        初始化特征提取器。

        Args:
            model_name (str): 预训练模型的名称 (例如 'wide_resnet50_2')。
            layer_names (list): 需要提取特征的层的名称列表。
            device (torch.device): 运行模型的设备 (CPU 或 CUDA)。
        """
        super().__init__()
        self.model_name = model_name
        self.layer_names = layer_names
        self.device = device
        self.model = None
        self.preprocessing = None

        # 加载预训练模型
        if model_name.startswith('efficientnet'):
            if HAS_TIMM:
                print(f"Loading {model_name} using timm...")
                self.model = timm.create_model(model_name, pretrained=True, features_only=True)
                # 获取 timm 模型的推荐预处理
                data_config = timm.data.resolve_model_data_config(self.model)
                self.preprocessing = timm.data.create_transform(**data_config, is_training=False)
                print(f"Using timm recommended preprocessing for {model_name}: {self.preprocessing}")
            else:
                 raise ImportError("timm library is required for EfficientNet models but not installed.")
        elif hasattr(models, model_name):
            # 从 torchvision 加载模型
            model_func = getattr(models, model_name)
            # 注意：torchvision >= 0.13 推荐使用 weights 参数
            if hasattr(models, f"{model_name}_Weights"):
                weights = getattr(models, f"{model_name}_Weights").DEFAULT
                self.model = model_func(weights=weights)
                self.preprocessing = weights.transforms() # 获取推荐的预处理变换
                print(f"Using recommended preprocessing for {model_name}: {self.preprocessing}")
            else:
                # 旧版 torchvision 或无推荐权重
                self.model = model_func(pretrained=True)
                # 定义默认预处理 (如果 weights.transforms() 不可用)
                from torchvision import transforms
                self.preprocessing = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                print(f"Warning: Using default preprocessing for {model_name}.")
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        if self.model is None:
            raise RuntimeError(f"Failed to load model: {model_name}")

        self.model.to(self.device)
        self.model.eval() # 设置为评估模式

        # 注册钩子 (hooks) 来捕获中间层的输出
        self.features = {layer: torch.empty(0) for layer in layer_names}
        self._register_hooks()

    def _register_hooks(self):
        """
        遍历模型层并注册前向钩子以捕获所需层的输出。
        """
        def get_activation(name):
            def hook(model, input, output):
                # 检查 timm 的 features_only 输出格式
                # 如果 output 是 list 或 tuple, 可能需要根据索引提取
                # (此处假设 timm features_only=True 时, hook 注册在正确模块上，输出是 Tensor)
                self.features[name] = output.detach()
            return hook

        # 修改: 适应 timm 的 features_only=True 结构
        if self.model_name.startswith('efficientnet') and HAS_TIMM and hasattr(self.model, 'feature_info'):
            # timm 模型设置 features_only=True 后，可以直接访问特征层
            # feature_info 提供了每层的信息，包括 module 名称
            registered_layers = set()
            target_layer_set = set(self.layer_names)

            # 尝试直接通过名称匹配模块 (适用于 features.X 格式)
            found_modules = {name: module for name, module in self.model.named_modules()} # Cache named modules

            for layer_name in self.layer_names:
                if layer_name in found_modules:
                    found_modules[layer_name].register_forward_hook(get_activation(layer_name))
                    registered_layers.add(layer_name)
                    print(f"Registered hook for timm layer: {layer_name}")
                else:
                     print(f"Warning: Could not find module named '{layer_name}' directly in timm model.")

            # 如果有层未直接找到，可以尝试基于 feature_info (如果需要更复杂的匹配)
            # (当前假设直接名称匹配足够)

            missing_layers = target_layer_set - registered_layers
            if missing_layers:
                print(f"Warning: Could not register hooks for all requested layers: {missing_layers}")
                print("Available modules in timm model:", list(found_modules.keys())) # 打印可用模块名以供调试

        else:
            # 保持原有的 torchvision 逻辑
            for name, layer_module in self.model.named_modules():
                if name in self.layer_names:
                    layer_module.register_forward_hook(get_activation(name))

        # 打印最终注册的层
        final_registered = list(self.features.keys()) # Keys in features dict where hooks might be attached
        print(f"Attempted to register hooks for: {self.layer_names}. Actual hooks might depend on module finding.")


    def forward(self, x):
        """
        执行模型的前向传递并返回提取的特征。

        Args:
            x (torch.Tensor): 输入图像张量 (需要先进行预处理)。

        Returns:
            dict: 一个字典，键是层名称，值是对应的特征图张量。
        """
        with torch.no_grad():
            # 修改: 对于 timm features_only=True, forward 可能返回一个列表
            # 但钩子应该在内部层级触发，所以这里调用模型本身即可
            _ = self.model(x.to(self.device))
        return self.features # 返回捕获的特征

    def get_preprocessing(self):
        """
        返回与加载的模型关联的推荐预处理变换。
        """
        return self.preprocessing
