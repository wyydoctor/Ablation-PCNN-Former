import os
import yaml
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.regression_dataset import RegressionDataset
from models.pcnn import PCNN
from utils.metrics import calculate_physical_inconsistency_rate
from utils.visualization import plot_ablation_curve
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Test PCNN model for ablation area prediction')
    parser.add_argument('--config', type=str, default='configs/pcnn_config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--output-dir', type=str, default='results/', help='Directory to save results')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载测试数据
    test_dataset = RegressionDataset(args.data)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # 加载模型
    model = PCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {args.checkpoint}")

    # 推理
    all_preds = []
    all_gts = []
    all_u = []
    all_d = []
    all_pw = []
    all_pd = []

    with torch.no_grad():
        for x, y, u, d in tqdm(test_loader, desc='Testing'):
            x, y, u, d = x.to(device), y.to(device), u.to(device), d.to(device)
            pred = model(x)

            all_preds.append(pred.cpu().numpy())
            all_gts.append(y.cpu().numpy())
            all_u.append(u.cpu().numpy())
            all_d.append(d.cpu().numpy())
            all_pw.append(x[:, 1].cpu().numpy())  # PW是第二个特征
            all_pd.append(x[:, 2].cpu().numpy())  # PD是第三个特征

    # 合并结果
    all_preds = np.concatenate(all_preds).flatten()
    all_gts = np.concatenate(all_gts).flatten()
    all_u = np.concatenate(all_u)
    all_d = np.concatenate(all_d)
    all_pw = np.concatenate(all_pw)
    all_pd = np.concatenate(all_pd)

    # 计算指标
    mse = np.mean((all_preds - all_gts) ** 2)
    mae = np.mean(np.abs(all_preds - all_gts))
    pir = calculate_physical_inconsistency_rate(all_preds, all_u, all_d)

    print("\n=== Test Results ===")
    print(f"MSE: {mse * 100:.2f} x10^-2 mm^4")
    print(f"MAE: {mae:.4f} mm²")
    print(f"Physical Inconsistency Rate: {pir:.2f}%")

    # 保存结果到CSV
    results_df = pd.DataFrame({
        'Voltage (V)': all_u,
        'Pulse Width (ns)': all_pw,
        'Total Pulse Duration (ms)': all_pd,
        'Electrode Spacing (mm)': all_d,
        'GT Area (mm²)': all_gts,
        'Pred Area (mm²)': all_preds,
        'Error (mm²)': np.abs(all_preds - all_gts)
    })
    results_df.to_csv(os.path.join(args.output_dir, 'test_results.csv'), index=False)
    print(f"Results saved to {os.path.join(args.output_dir, 'test_results.csv')}")

    # 按电极间距分组绘制消融曲线
    unique_ds = np.unique(all_d)
    for d in unique_ds:
        mask = (all_d == d)
        # 按电压排序
        sort_idx = np.argsort(all_u[mask])
        plot_ablation_curve(
            voltages=all_u[mask][sort_idx],
            pred_areas=all_preds[mask][sort_idx],
            gt_areas=all_gts[mask][sort_idx],
            electrode_spacing=d,
            save_path=os.path.join(args.output_dir, f'ablation_curve_d{int(d)}mm.png'),
            show=False
        )
    print(f"Ablation curves saved to {args.output_dir}")


if __name__ == '__main__':
    main()