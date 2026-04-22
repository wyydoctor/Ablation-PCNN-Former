import torch
import torch.nn as nn
from typing import Optional


class PCNN(nn.Module):
    """Penalty-based Constrained Neural Network (PCNN)
    论文2.3节：4层全连接，每层64神经元，ReLU激活
    优化：适配电极间距作为输入特征（input_dim=4）
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入参数 (B, 4) -> [电压U, 脉宽PW, 总脉宽PD, 电极间距d]
        Returns:
            pred: 预测消融面积 (B, 1)
        """
        return self.layers(x)

    @classmethod
    def from_config(cls, config: dict) -> "PCNN":
        """从配置文件初始化模型（解耦硬编码）"""
        return cls(
            input_dim=config["model"]["input_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            output_dim=config["model"]["output_dim"]
        )