# -*- coding: utf-8 -*-
"""
config.py —— 三方契约文件(README §九):特征顺序、窗口参数、划分比例、SEED。
任何一方要改这里的值,先在群里说明、达成一致后改一次、三人各自重跑。
"""
from pathlib import Path

# ---------------- 复现性 ----------------
SEED = 42

# ---------------- 路径 ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]           # .../课题报告
DATA_DIR    = PROJECT_ROOT / "data"                          # 原始 csv(下载脚本会放这里)
RAW_ZIP     = DATA_DIR / "jena_climate_2009_2016.csv.zip"
RAW_CSV     = DATA_DIR / "jena_climate_2009_2016.csv"
# 数据口径版本:v3 = 数据管线 方案(2026-11-23 全组拍板):
#   全行去重 → -9999 置缺失 → 整点均值重采样(README §3.3"按小时聚合")
#   → 连续缺失 ≥ MAX_GAP_HOURS 视为真实断档:删行并把序列切成互不相连的片段,
#     片段内部短缺口时间插值 → 目标 24h 完整落 split、输入可回溯前一 split、
#     窗口绝不跨片段(README §5.2)。
# 旧口径产物:v1 = outputs/{checkpoints,preds}(原始切片口径),v2 = outputs/v2
# (A 整点行口径),均已作废不再复用。v3 下 Transformer 侧窗口集合与 y 真值和 LSTM 侧逐窗口
# 一致,三模型预测 npy 按行直接对齐,融合无需再做对齐处理。
DATA_VERSION = "v3"
VERSION_DIR  = PROJECT_ROOT / "outputs" / DATA_VERSION
CKPT_DIR     = VERSION_DIR / "checkpoints"                   # transformer_best.pt / patchtst_best.pt
PREDS_DIR    = VERSION_DIR / "preds"                         # 预测缓存 npy(摄氏度)
EXPERIMENTS_DIR = VERSION_DIR / "experiments"                # 冒烟/调参产物(只推 val)
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"          # 【C】收图区(与数据口径无关)

# ---------------- 数据契约 ----------------
DATASET_URL = ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
               "jena_climate_2009_2016.csv.zip")

# 特征顺序即契约:T 必须在下标 1(README §九表格)。列名 = 原始 csv 表头,逐字一致。
FEATURE_COLUMNS = [
    "p (mbar)",       # 0 气压
    "T (degC)",       # 1 气温 —— 预测目标
    "VPmax (mbar)",   # 2 饱和水汽压
    "VPdef (mbar)",   # 3 水汽压亏缺
    "sh (g/kg)",      # 4 比湿
    "rho (g/m**3)",   # 5 空气密度
    "wv (m/s)",       # 6 风速
]
TARGET_IDX = 1            # T 在特征序列中的下标
SENTINEL   = -9999.0      # wv 风速列的异常占位值(README §3.2 / §5.3)

TIME_COL = "Date Time"
TIME_FMT = "%d.%m.%Y %H:%M:%S"

# 真实断档阈值:连续缺失小时数 ≥ 该值即删行切片段(数据管线 实测:断档为
# 30min/20min/20min/16h/74h,取 6h 把 16h/74h 切开,30min/20min 用插值补)。
MAX_GAP_HOURS = 6

# 降采样:原始 10 分钟 → 每小时一条(只保留整点行,README §3.3;全项目只用这一种)
SEQ_IN, SEQ_OUT = 72, 24            # 过去 72h → 未来 24h
SPLIT_RATIO = (0.7, 0.1, 0.2)       # 训练 / 验证 / 测试,按时间顺序切(README §5.2)

# ---------------- 训练超参(Transformer计划.md §5) ----------------
BATCH_SIZE   = 128
MAX_EPOCHS   = 50
EARLY_STOP   = 5            # 验证 MSE 连续 5 轮无改善即停
LR           = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP    = 1.0

# ---------------- 模型结构 ----------------
# 方案 A 主线:encoder-only
# 2026-11-23 v3 容量定稿:d128-L3 + pool 头 + 时间特征(11 维) + 残差 ΔT 预测,
# 参数量 34 万;val MAE 1.662 / test MAE 1.785(测试集只推理一次)。
# 容量消融(val MAE):d64L2 1.691 → d128L2 1.678 → d128L3 1.662 → d160L3 1.700(回退);
# d128L3 三种子 1.662/1.689/1.692 均值 1.681(净改善 0.014);train/val gap≤0.96x 无过拟合。
# LSTM 侧对照:LSTM 1.86 / LSTM+tf+res 1.73。
ENC_CFG = dict(n_feat=11, d_model=128, nhead=4, n_layer=3,
               d_ff=128, dropout=0.1, head="pool",
               residual=True, target_idx=1)
# 方案 B 附加:PatchTST(patch_len=stride=12 → 直切 6 个 patch,精确铺满 72 步)
PATCH_CFG = dict(n_feat=7, d_model=64, nhead=4, n_layer=2,
                 d_ff=128, dropout=0.2, patch_len=12, stride=12)

# checkpoint / 预测 npy 的文件名(README §九契约;PatchTST 是附加实验故前缀 patchtst_)
ENC_CKPT, ENC_TAG       = "transformer_best.pt", "transformer"
PATCH_CKPT, PATCH_TAG   = "patchtst_best.pt", "patchtst"


def get_device():
    """cuda 优先;训练脚本与导出脚本共用。"""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
