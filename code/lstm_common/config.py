# -*- coding: utf-8 -*-
"""
全项目唯一配置（契约文件）。

修改纪律：本文件中的特征顺序、T 列下标、窗口参数、划分比例是三方契约，
任何一方要改，先群里说明、达成一致后改一次、三人各自重跑。
禁止"顺手优化"。

负责人：数据管线
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
# config.py 位于 <ROOT>/code/common/config.py
ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT / "data"
OUT_DIR: Path = ROOT / "outputs"
CKPT_DIR: Path = OUT_DIR / "checkpoints"
PRED_DIR: Path = OUT_DIR / "preds"
FIG_DIR: Path = OUT_DIR / "figures"
LOG_DIR: Path = OUT_DIR / "logs"

for _d in (DATA_DIR, OUT_DIR, CKPT_DIR, PRED_DIR, FIG_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CSV: Path = DATA_DIR / "jena_climate_2009_2016.csv"
RAW_ZIP: Path = DATA_DIR / "jena_climate_2009_2016.csv.zip"
RAW_URL: str = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

# 清洗后的逐小时数据（含缺失小时标记），供体检与画图复用
HOURLY_PARQUET: Path = DATA_DIR / "jena_hourly.parquet"
HOURLY_CSV: Path = DATA_DIR / "jena_hourly.csv"
CLEAN_SUMMARY_JSON: Path = OUT_DIR / "logs" / "data_qc.json"

# ---------------------------------------------------------------- 数据契约
# 注意：CSV 里实际列名是 'rho (g/m**3)'，不是 README 里写的 'rho (g/m³)'。
# 用错会直接 KeyError，所以这里写死真实列名。
FEATURES: list[str] = [
    "p (mbar)",
    "T (degC)",
    "VPmax (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "rho (g/m**3)",
    "wv (m/s)",
]
N_FEATURES: int = len(FEATURES)                      # 7
TARGET_COL: str = "T (degC)"
TARGET_IDX: int = FEATURES.index(TARGET_COL)         # 1
DATETIME_COL: str = "Date Time"
DATETIME_FMT: str = "%d.%m.%Y %H:%M:%S"

# 原始数据中的异常占位值。-9999 不止出现在 wv，也出现在 max. wv (m/s)。
BAD_VALUES: tuple[float, ...] = (-9999.0,)

# 超过该小时数（含）的连续缺失视为"真实断档"，把序列切成互不相连的片段。
# 依据：数据中真实断档为 30min / 20min / 20min / 16h / 74h，
# 取 6h 可以把 16h 与 74h 切开，同时保留 30min/20min 用插值补。
MAX_GAP_HOURS: int = 6

# ---------------------------------------------------------------- 任务契约
SEQ_LEN: int = 72          # 输入历史长度（小时）
PRED_LEN: int = 24         # 预测长度（小时）
STRIDE: int = 1            # 滑窗步长

SPLIT_RATIO: tuple[float, float, float] = (0.70, 0.10, 0.20)   # train / val / test

# ---------------------------------------------------------------- 可复现性
SEED: int = 42

# ---------------------------------------------------------------- LSTM 超参（起始配置）
LSTM_HIDDEN: int = 64
LSTM_LAYERS: int = 2
LSTM_DROPOUT: float = 0.1
BATCH_SIZE: int = 128
EPOCHS: int = 60
LR: float = 1e-3
WEIGHT_DECAY: float = 0.0
GRAD_CLIP: float = 1.0
PATIENCE: int = 10          # 验证集连续多少个 epoch 不提升就早停
LR_PATIENCE: int = 4        # ReduceLROnPlateau 的 patience
LR_FACTOR: float = 0.5

# ---------------------------------------------------------------- 可选改进开关
# 以下开关只用于 B 的消融实验，默认关闭以保证主对比表严格使用 7 特征契约。
USE_TIME_FEATURES: bool = False   # + 时刻/年内日的 sin-cos 周期特征（输入变为 72x11）
USE_RESIDUAL: bool = False        # 预测相对"最后一个观测"的增量，而不是绝对气温
FORGET_BIAS_INIT: float = 0.0     # 遗忘门偏置初值；1.0 是常用技巧，见 Jozefowicz et al. 2015

# 周期特征（USE_TIME_FEATURES=True 时生效）
TIME_FEATURE_NAMES: list[str] = ["tod_sin", "tod_cos", "doy_sin", "doy_cos"]

# ---------------------------------------------------------------- 指标口径
# 全局 MAE/RMSE：把所有 (样本 x 预测步) 的误差一起池化后计算（本项目的正式口径）。
# 分时 MAE：先按预测步 h 分组算，再报告 24 个数（用于分时误差曲线）。
METRIC_POOLING: str = "global"

# ---------------------------------------------------------------- 画图
def set_plot_style() -> None:
    """统一 matplotlib 样式，并挑一个支持中文的字体（Windows 常见字体优先）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = [f for f in candidates if f in available]
    plt.rcParams.update({
        "font.sans-serif": picked or ["DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
    })


def safe_shutdown() -> None:
    """
    正常结束后立刻退出进程。

    原因：本机是 Python 3.13 + torch 2.11 + cuDNN + RTX 5060(Blackwell)。
    只要用过 cuDNN 版的 nn.LSTM，进程退出时会在 cuDNN 的 DLL 卸载阶段触发
    0xC0000409（-1073740791）。这是**纯退出阶段**的问题：所有结果在崩溃前
    都已落盘。注意 `os._exit(0)` 也绕不过去，因为崩溃发生在 Windows 进程
    终止时的 DLL 分离阶段，所以这里只做提示，不再假装能修好。
    想要干净退出码的话，训练时加 `--no-cudnn`（约慢 3~5 倍）。
    """
    import os
    import sys
    try:
        print("\n[提示] 本机 torch+cuDNN 在进程退出时会返回 0xC0000409(-1073740791)，"
              "属于已知环境问题；上面的结果已经全部写盘，可忽略。"
              "\n        若需要干净的退出码，下次训练加 --no-cudnn。")
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # pragma: no cover
        pass
    os._exit(0)


def env_info() -> dict:
    """记录运行环境，便于报告里写"实验环境"一栏。"""
    import platform
    import sys

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception:  # pragma: no cover
        info["torch"] = None
    return info


if __name__ == "__main__":
    import json
    print(json.dumps({"root": str(ROOT), "features": FEATURES, "target_idx": TARGET_IDX,
                      "seq_len": SEQ_LEN, "pred_len": PRED_LEN, "split": SPLIT_RATIO,
                      "env": env_info()}, ensure_ascii=False, indent=2))
