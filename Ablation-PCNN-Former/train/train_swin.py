import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data.segmentation_dataset import SegmentationDataset
from models.swin_transformer import SwinTransformerSegmenter
from utils.metrics import calculate_segmentation_metrics
from utils.visualization import plot_training_curves
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train Swin-Transformer for TTC image segmentation')
    parser.add_argument('--config', type=str, default='configs/swin_config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/', help='Directory to save logs')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 创建目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集
    train_dataset = SegmentationDataset(
        image_dir=config['data']['train_image_dir'],
        mask_dir=config['data']['train_mask_dir'],
        transform=True
    )
    val_dataset = SegmentationDataset(
        image_dir=config['data']['val_image_dir'],
        mask_dir=config['data']['val_mask_dir'],
        transform=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # 模型
    model = SwinTransformerSegmenter(pretrained=config['model']['pretrained']).to(device)

    # 损失函数 (二分类交叉熵)
    criterion = nn.BCELoss()

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay']
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # 训练循环
    best_val_iou = 0.0
    train_losses = []
    val_losses = []
    val_ious = []

    for epoch in range(config['epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['epochs']} ===")

        # 训练
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc='Training'):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_gts = []

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                all_preds.append(outputs.cpu().numpy())
                all_gts.append(masks.cpu().numpy())

        val_loss /= len(val_dataset)
        val_losses.append(val_loss)

        # 计算指标
        all_preds = np.concatenate(all_preds)
        all_gts = np.concatenate(all_gts)
        metrics = calculate_segmentation_metrics(all_preds, all_gts)
        val_ious.append(metrics['IoU'])

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(
            f"IoU: {metrics['IoU']:.4f} | DSC: {metrics['DSC']:.4f} | Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f}")

        # 保存最佳模型
        if metrics['IoU'] > best_val_iou:
            best_val_iou = metrics['IoU']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': best_val_iou,
                'config': config
            }, os.path.join(args.checkpoint_dir, 'swin_best.pth'))
            print(f"Best model saved with IoU: {best_val_iou:.4f}")

        scheduler.step(metrics['IoU'])

    # 绘制训练曲线
    plot_training_curves(
        train_losses, val_losses, val_ious,
        metric_name='IoU',
        save_path=os.path.join(args.log_dir, 'swin_training_curves.png')
    )

    print(f"\nTraining complete! Best Val IoU: {best_val_iou:.4f}")


if __name__ == '__main__':
    main()