import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import os

from data.regression_dataset import RegressionDataset  # 需确保数据集返回4维特征（含电极间距）
from losses.physics_constrained_loss import PhysicsConstrainedLoss
from utils.metrics import calculate_physical_inconsistency_rate
from utils.seed import set_seed
from utils.common import get_device, load_config, save_fold_results

def main(config_path: str):
    # 加载配置+异常处理
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"配置加载失败: {e}")
        return

    set_seed(config["seed"])
    device = get_device()
    print(f"使用设备: {device}")

    # 加载数据集（需修改RegressionDataset返回4维特征：[U, PW, PD, d]）
    try:
        dataset = RegressionDataset(config["data"]["csv_path"])
        # 数据集归一化（示例：需在RegressionDataset中实现，或此处预处理）
        # 注：实际应在Dataset中完成归一化，避免数据泄露
        norm_params = config["data"]["norm_params"]
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return

    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=config["seed"])
    all_fold_metrics = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n=== Fold {fold + 1}/5 ===")
        try:
            # 数据集拆分
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

            # 初始化模型（从配置读取参数）
            from models.pcnn import PCNN
            model = PCNN.from_config(config).to(device)
            # 初始化损失（从配置读取所有参数）
            criterion = PhysicsConstrainedLoss(**config["loss"]).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

            # 训练循环
            best_val_mae = float("inf")
            for epoch in range(config["epochs"]):
                model.train()
                train_loss = 0.0
                for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                    x, y, u, d = [b.to(device) for b in batch]
                    optimizer.zero_grad()
                    pred = model(x)
                    loss, _ = criterion(pred, y, u, d)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * x.size(0)

                # 验证
                model.eval()
                val_mae = 0.0
                all_preds = []
                all_u = []
                all_d = []
                with torch.no_grad():
                    for batch in val_loader:
                        x, y, u, d = [b.to(device) for b in batch]
                        pred = model(x)
                        val_mae += torch.mean(torch.abs(pred - y)).item() * x.size(0)
                        all_preds.append(pred.cpu().numpy())
                        all_u.append(u.cpu().numpy())
                        all_d.append(d.cpu().numpy())

                # 计算指标
                train_loss /= len(train_dataset)
                val_mae /= len(val_dataset)
                all_preds = np.concatenate(all_preds)
                all_u = np.concatenate(all_u)
                all_d = np.concatenate(all_d)
                pir = calculate_physical_inconsistency_rate(all_preds, all_u, all_d)

                print(f"Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | PIR: {pir:.2f}%")

                # 保存最佳模型
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save({
                        "fold": fold + 1,
                        "model_state_dict": model.state_dict(),
                        "config": config
                    }, f"checkpoints/pcnn_fold{fold + 1}_best.pth")
                    fold_models.append(model.state_dict())

                scheduler.step(val_mae)

            # 记录单折指标
            fold_metrics = {"best_val_mae": best_val_mae, "pir": pir}
            all_fold_metrics.append(fold_metrics)
            save_fold_results(fold + 1, fold_metrics)

        except Exception as e:
            print(f"Fold {fold + 1} 训练失败: {e}")
            continue

    # 5折模型融合（加权平均：按MAE倒数加权）
    if all_fold_metrics:
        # 计算权重（MAE越小，权重越高）
        mae_values = [m["best_val_mae"] for m in all_fold_metrics]
        weights = [1 / mae for mae in mae_values]
        weights = np.array(weights) / np.sum(weights)

        # 加载所有折的模型，融合参数
        final_model = PCNN.from_config(config).to(device)
        final_state_dict = final_model.state_dict()
        for key in final_state_dict.keys():
            final_state_dict[key] = torch.zeros_like(final_state_dict[key])
            for fold_idx, model_state in enumerate(fold_models):
                final_state_dict[key] += weights[fold_idx] * model_state[key]
        final_model.load_state_dict(final_state_dict)

        # 保存融合后的最终模型
        torch.save({
            "model_state_dict": final_state_dict,
            "fold_weights": weights,
            "config": config
        }, "checkpoints/pcnn_final_fused.pth")
        print("\n✅ 5折模型融合完成，保存至 checkpoints/pcnn_final_fused.pth")

    # 打印5折平均结果
    avg_mae = np.mean([m["best_val_mae"] for m in all_fold_metrics])
    avg_pir = np.mean([m["pir"] for m in all_fold_metrics])
    print(f"\n=== 5-Fold Average Results ===")
    print(f"Average Val MAE: {avg_mae:.4f} mm²")
    print(f"Average Physical Inconsistency Rate: {avg_pir:.2f}%")

if __name__ == "__main__":
    main("configs/pcnn_config.yaml")