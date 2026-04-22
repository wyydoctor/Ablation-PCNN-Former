import random
import os
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    固定所有随机种子以确保可复现性
    Args:
        seed: 随机种子值
        deterministic: 是否使用确定性算法
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False