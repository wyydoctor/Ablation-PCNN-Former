import torch
import torch.nn as nn
from timm import create_model


class SwinTransformerSegmenter(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        # 使用timm的Swin-Tiny预训练模型
        self.backbone = create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )

        # 上采样解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入TTC图像 (B, 3, 256, 256)
        Returns:
            mask: 二值分割掩码 (B, 1, 256, 256)
        """
        features = self.backbone(x)  # (B, 768, 16, 16)
        mask = self.decoder(features)
        return torch.sigmoid(mask)