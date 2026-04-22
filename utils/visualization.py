import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional, List, Tuple

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def plot_ablation_curve(
        voltages: np.ndarray,
        pred_areas: np.ndarray,
        gt_areas: np.ndarray,
        electrode_spacing: float,
        save_path: Optional[str] = None,
        show: bool = True
):
    """
    绘制消融面积随电压变化曲线 (复现论文Figure 6)
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(voltages, gt_areas, c='black', s=30, label='Experimental Data', zorder=5)
    ax.plot(voltages, pred_areas, c='#d62728', linewidth=2.5, label='Ablation-PCNN-Former', zorder=4)

    # 绘制消融平台线
    s_sat = 12.5 * electrode_spacing - 22.5
    ax.axhline(y=s_sat, c='gray', linestyle='--', linewidth=1.5, label=f'Ablation Plateau (~{s_sat:.1f} mm²)', zorder=3)

    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Ablation Area (mm²)')
    ax.set_title(f'Ablation Area vs. Voltage (d={electrode_spacing} mm)')
    ax.legend(frameon=True, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_segmentation_results(
        image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True
):
    """
    绘制TTC图像、真实掩码和预测掩码的对比图
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(image)
    axes[0].set_title('TTC-Stained Image')
    axes[0].axis('off')

    # 真实掩码
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    # 预测掩码
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_training_curves(
        train_losses: List[float],
        val_losses: List[float],
        val_metrics: Optional[List[float]] = None,
        metric_name: str = 'MAE',
        save_path: Optional[str] = None,
        show: bool = True
):
    """
    绘制训练/验证损失曲线
    """
    fig, axes = plt.subplots(1, 2 if val_metrics else 1, figsize=(12 if val_metrics else 6, 5))

    if val_metrics:
        ax1, ax2 = axes
    else:
        ax1 = axes

    # 损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, c='#1f77b4', linewidth=2, label='Train Loss')
    ax1.plot(epochs, val_losses, c='#ff7f0e', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # 指标曲线
    if val_metrics:
        ax2.plot(epochs, val_metrics, c='#2ca02c', linewidth=2, label=f'Val {metric_name}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name)
        ax2.set_title(f'Validation {metric_name}')
        ax2.legend(frameon=True)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()