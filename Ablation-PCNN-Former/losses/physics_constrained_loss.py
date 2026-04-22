import torch
import torch.nn as nn
from typing import Tuple, Dict

class PhysicsConstrainedLoss(nn.Module):
    def __init__(
        self,
        e_ire: float = 478.0,  # IRE电场阈值 (V/cm)
        electrode_radius: float = 0.05,  # 电极半径 (cm)
        max_ablation_area: float = 70.0,  # 最大实验消融面积 (mm²)
        w_physics: float = 0.3,  # 物理损失权重
        lambda_: float = 0.1,    # 电导率损失权重
        k_sigmoid: float = 100.0, # 平滑指示函数斜率
        s_sat_coeff: float = 12.5, # 消融平台系数
        s_sat_bias: float = -22.5  # 消融平台偏置
    ):
        super().__init__()
        self.e_ire = e_ire
        self.r = electrode_radius
        self.s_max = max_ablation_area
        self.w_physics = w_physics
        self.lambda_ = lambda_
        self.k = k_sigmoid
        self.s_sat_coeff = s_sat_coeff
        self.s_sat_bias = s_sat_bias

    def smooth_indicator(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """可微分的sigmoid近似指示函数 I(x < t)"""
        return torch.sigmoid(self.k * (t - x))

    def s_sat(self, d: torch.Tensor) -> torch.Tensor:
        """几何自适应饱和消融面积 (Eq.7)
        优化：解硬编码，从配置读取系数/偏置
        """
        return self.s_sat_coeff * d + self.s_sat_bias

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        u: torch.Tensor,    # 电压 (V)
        d: torch.Tensor     # 电极间距 (mm)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            pred: 预测消融面积 (B, 1)
            gt: 真实消融面积 (B, 1)
            u: 输入电压 (B,)
            d: 输入电极间距 (B,)
        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量
        """
        # 维度检查
        assert pred.dim() == 2 and pred.shape[1] == 1, f"pred维度应为(B,1)，实际为{pred.shape}"
        assert gt.shape == pred.shape, f"gt维度应与pred一致，实际为{gt.shape}"
        assert u.dim() == 1 and u.shape[0] == pred.shape[0], f"u维度应为(B,)，实际为{u.shape}"
        assert d.shape == u.shape, f"d维度应与u一致，实际为{d.shape}"

        # 数据损失 (Eq.4)
        data_loss = torch.mean(((pred - gt) / self.s_max) ** 2)

        # 计算近似最大电场强度 (Eq.5)
        d_cm = d / 10.0  # 转换为cm
        e_max = u / (d_cm - 2 * self.r)

        # 阈值约束损失 (Eq.6)
        threshold_mask = self.smooth_indicator(e_max, self.e_ire)
        threshold_loss = torch.mean(threshold_mask * (pred / self.s_max))

        # 电导率/消融平台约束损失 (Eq.8)
        s_sat = self.s_sat(d)
        over_sat_mask = self.smooth_indicator(s_sat, pred.squeeze(dim=1))  # 安全降维
        conductivity_loss = torch.mean(over_sat_mask * ((pred - s_sat.unsqueeze(1)) / self.s_max) ** 2)

        # 总物理损失
        physics_loss = threshold_loss + self.lambda_ * conductivity_loss

        # 总损失 (Eq.3)
        total_loss = data_loss + self.w_physics * physics_loss

        loss_dict = {
            "total_loss": total_loss.item(),
            "data_loss": data_loss.item(),
            "physics_loss": physics_loss.item(),
            "threshold_loss": threshold_loss.item(),
            "conductivity_loss": conductivity_loss.item()
        }

        return total_loss, loss_dict