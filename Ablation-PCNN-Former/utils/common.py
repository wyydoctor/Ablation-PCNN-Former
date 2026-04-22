import os
import torch
import yaml
import numpy as np
from typing import Tuple, Dict, Optional
from models.pcnn import PCNN

def get_device() -> torch.device:
    """获取可用设备（自动检测GPU/CPU）"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_path: str) -> Dict:
    """加载配置文件，补充异常处理"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件解析失败: {e}")

def load_pcnn_model(
    config: Dict,
    checkpoint_path: str,
    device: torch.device
) -> PCNN:
    """加载PCNN模型，补充异常处理"""
    model = PCNN.from_config(config).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint缺少model_state_dict键")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")

def normalize_feature(
    x: np.ndarray,
    feature_name: str,
    norm_params: Dict
) -> np.ndarray:
    """特征归一化（MinMaxScaler）"""
    min_val, max_val = norm_params[feature_name]
    return (x - min_val) / (max_val - min_val)

def denormalize_feature(
    x: np.ndarray,
    feature_name: str,
    norm_params: Dict
) -> np.ndarray:
    """特征反归一化"""
    min_val, max_val = norm_params[feature_name]
    return x * (max_val - min_val) + min_val

def save_fold_results(
    fold: int,
    metrics: Dict,
    save_dir: str = "results/fold_metrics"
) -> None:
    """保存单折训练指标，用于后续融合"""
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"fold_{fold}_metrics.npy"), metrics)