import os
import argparse
import torch
import numpy as np
from scipy.optimize import minimize

from utils.seed import set_seed
from utils.common import get_device, load_config, load_pcnn_model


def parse_args():
    parser = argparse.ArgumentParser(description='Inverse parameter optimization for IRE')
    parser.add_argument('--config', type=str, default='configs/pcnn_config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--target-area', type=float, required=True, help='Target ablation area (mm²)')
    parser.add_argument('--electrode-spacing', type=float, required=True, help='Electrode spacing (mm)')
    parser.add_argument('--initial-voltage', type=float, default=1000.0, help='Initial guess for voltage (V)')
    parser.add_argument('--initial-pw', type=float, default=500.0, help='Initial guess for pulse width (ns)')
    return parser.parse_args()


def objective_function(
        params: np.ndarray,
        model: torch.nn.Module,
        target_area: float,
        electrode_spacing: float,
        device: torch.device,
        norm_params: dict
) -> float:
    """
    目标函数：(预测面积 - 目标面积)^2
    优化：加入电极间距到模型输入，特征归一化
    """
    voltage, pw = params
    pd = 8 * pw  # PN=8（固定）
    d = electrode_spacing

    # 归一化输入特征
    voltage_norm = (voltage - norm_params["voltage"][0]) / (norm_params["voltage"][1] - norm_params["voltage"][0])
    pw_norm = (pw - norm_params["pw"][0]) / (norm_params["pw"][1] - norm_params["pw"][0])
    pd_norm = (pd - norm_params["pd"][0]) / (norm_params["pd"][1] - norm_params["pd"][0])
    d_norm = (d - norm_params["electrode_spacing"][0]) / (
                norm_params["electrode_spacing"][1] - norm_params["electrode_spacing"][0])

    # 构造4维输入：[电压, 脉宽, PD, 电极间距]
    x = torch.tensor([[voltage_norm, pw_norm, pd_norm, d_norm]], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_area = model(x).item()

    return (pred_area - target_area) ** 2


def main():
    args = parse_args()

    # 加载配置+模型（带异常处理）
    try:
        config = load_config(args.config)
        device = get_device()
        model = load_pcnn_model(config, args.checkpoint, device)
        norm_params = config["data"]["norm_params"]
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    set_seed(config["seed"])

    # 参数边界约束
    bounds = [
        (norm_params["voltage"][0], norm_params["voltage"][1]),  # 电压范围 (V)
        (norm_params["pw"][0], norm_params["pw"][1])  # 脉宽范围 (ns)
    ]

    # 初始猜测
    x0 = [args.initial_voltage, args.initial_pw]

    print(f"\n=== Inverse Optimization ===")
    print(f"Target Ablation Area: {args.target_area:.2f} mm²")
    print(f"Electrode Spacing: {args.electrode_spacing:.1f} mm")
    print(f"Initial Guess: Voltage={x0[0]:.1f} V, PW={x0[1]:.1f} ns")

    # 执行优化（主策略+备用策略）
    try:
        # 主优化器
        result = minimize(
            fun=objective_function,
            x0=x0,
            args=(model, args.target_area, args.electrode_spacing, device, norm_params),
            method=config["inverse_optim"]["method"],
            bounds=bounds,
            options={'maxiter': config["inverse_optim"]["maxiter"], 'ftol': config["inverse_optim"]["ftol"]}
        )

        # 备用策略（主策略失败时）
        if not result.success:
            print(f"主优化器失败，尝试备用策略 {config['inverse_optim']['fallback_method']}...")
            result = minimize(
                fun=objective_function,
                x0=x0,
                args=(model, args.target_area, args.electrode_spacing, device, norm_params),
                method=config["inverse_optim"]["fallback_method"],
                bounds=bounds,
                options={'maxiter': config["inverse_optim"]["maxiter"]}
            )

        # 输出结果
        if result.success:
            opt_voltage, opt_pw = result.x
            opt_pd = 8 * opt_pw
            opt_pd_ms = opt_pd / 1e6  # 修正单位：ns → ms

            # 验证预测
            voltage_norm = (opt_voltage - norm_params["voltage"][0]) / (
                        norm_params["voltage"][1] - norm_params["voltage"][0])
            pw_norm = (opt_pw - norm_params["pw"][0]) / (norm_params["pw"][1] - norm_params["pw"][0])
            pd_norm = (opt_pd - norm_params["pd"][0]) / (norm_params["pd"][1] - norm_params["pd"][0])
            d_norm = (args.electrode_spacing - norm_params["electrode_spacing"][0]) / (
                        norm_params["electrode_spacing"][1] - norm_params["electrode_spacing"][0])
            x = torch.tensor([[voltage_norm, pw_norm, pd_norm, d_norm]], dtype=torch.float32).to(device)

            with torch.no_grad():
                final_pred = model(x).item()

            print("\n✅ Optimization Successful!")
            print(f"Optimized Parameters:")
            print(f"  Voltage (U): {opt_voltage:.1f} V")
            print(f"  Pulse Width (PW): {opt_pw:.1f} ns")
            print(f"  Total Pulse Duration (PD): {opt_pd_ms:.4f} ms (原始值: {opt_pd:.1f} ns)")
            print(f"  Pulse Number (PN): 8 (fixed)")
            print(f"  Repetition Frequency (RF): 400 Hz (fixed)")
            print(f"\nPredicted Ablation Area: {final_pred:.2f} mm²")
            print(f"Target Ablation Area: {args.target_area:.2f} mm²")
            print(f"Absolute Error: {abs(final_pred - args.target_area):.4f} mm²")
        else:
            print(f"\n❌ Optimization Failed!")
            print(f"Message: {result.message}")
    except Exception as e:
        print(f"优化执行失败: {e}")


if __name__ == '__main__':
    main()