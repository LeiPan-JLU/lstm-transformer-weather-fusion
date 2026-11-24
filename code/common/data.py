# -*- coding: utf-8 -*-
"""
data.py —— 数据清洗与处理(v3 口径 = 数据管线 方案,2026-11-23 全组拍板;README §5.3)。

    下载解压 → 读 csv/解析时间/排序 → 全行去重(记录仪重启造成 327 条重复)
    → 选 7 特征 → -9999 置缺失 → 整点均值重采样(每小时 6 条取均值)
    → 连续缺失 ≥ MAX_GAP_HOURS 视为真实断档:删行,序列切成互不相连的片段;
      片段内部短缺口用时间插值补 → 按行数 70/10/20 切 split
    → StandardScaler(只在训练段行上 fit)
    → 滑窗:目标 24h 完整落在 split 内,输入可回溯前一 split,窗口绝不跨片段
    → DataLoader

get_loaders() 是对接契约(README §九)的唯一起口:
LSTM(B) 与 Transformer(A) 都从这里拿数据,样本为 (x[72,7], y[24,1]),
全部处于标准化空间;要摄氏度时用 to_celsius_T() 逆变换。

与数据管线 实现的对应关系(逐行等价移植,保证 两条管线窗口与真值一致):
    load_raw()          → B 的 load_raw()(读 csv/排序/去重/置缺失)
    to_hourly()         → B 的 to_hourly()(resample("1h").mean,不用 [::6] 切片)
    _segments()         → B 的 _segment_ids + 片段内插值
    _windows()          → B 的 window_indices 语义(目标完整落 split + 不跨片段)
防泄漏对照(README 清单):
  #1 scaler 只在训练段行 fit  → get_loaders() 内实现
  #2 窗口标签不跨 split       → 每个 split 独立开窗(见 _windows)
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# 保证无论作为脚本直跑还是被 common.data 导入,都能找到 common.config
_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from lstm_common.config import (BATCH_SIZE, DATA_DIR, DATASET_URL, FEATURE_COLUMNS,
                           MAX_GAP_HOURS, RAW_CSV, RAW_ZIP, SEQ_IN, SEQ_OUT,
                           SENTINEL, SPLIT_RATIO, TARGET_IDX, TIME_COL, TIME_FMT)


def download_dataset(force: bool = False) -> Path:
    """下载 Jena Climate 原始压缩包并解压(约 13.6 MB)。csv 已存在则跳过。
    用 Python urllib 下载(本机 Git Bash 的 curl 走 schannel 常挂)。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists() and not force:
        return RAW_CSV
    if not RAW_ZIP.exists() or force:
        print(f"下载 {DATASET_URL} → {RAW_ZIP}")
        urllib.request.urlretrieve(DATASET_URL, RAW_ZIP)
        print(f"  完成:{RAW_ZIP.stat().st_size / 1e6:.1f} MB")
    with zipfile.ZipFile(RAW_ZIP) as zf:
        for name in zf.namelist():                      # 防路径穿越
            assert ".." not in Path(name).parts, f"压缩包路径异常:{name}"
        zf.extractall(DATA_DIR)
    assert RAW_CSV.exists(), "解压后未找到预期的 csv"
    print(f"原始数据就绪:{RAW_CSV.name}({RAW_CSV.stat().st_size / 1e6:.1f} MB)")
    return RAW_CSV


def load_clean_dataframe(csv_path=None) -> pd.DataFrame:
    """读 csv → 解析时间 → 排序 → 去重 → 取 7 特征 → -9999 置缺失
    → 整点均值重采样 → 删除长断档小时并切片段 → 片段内短缺口时间插值。

    返回 DatetimeIndex(仅实际存在观测的小时,断档小时已删)、7 列
    (顺序 = config.FEATURE_COLUMNS)的 DataFrame,无缺失。
    """
    csv_path = Path(csv_path) if csv_path else RAW_CSV
    raw = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError(f"csv 缺少契约特征列:{missing}(实际列:{list(raw.columns)})")

    dt = pd.to_datetime(raw[TIME_COL], format=TIME_FMT)
    df = pd.DataFrame(raw[FEATURE_COLUMNS].to_numpy(dtype=np.float64),
                      index=dt, columns=FEATURE_COLUMNS).sort_index()

    # 记录仪重启造成的完全重复行(327 条),去重(数据管线 实测发现)
    n_dup = int(df.index.duplicated().sum())
    if n_dup:
        df = df[~df.index.duplicated(keep="first")]

    # 异常占位值(-9999,不止 wv 列可能含)→ 缺失(README §5.3 第 4 步)
    vals = df.to_numpy()
    n_bad = int((vals == SENTINEL).sum())
    vals[vals == SENTINEL] = np.nan
    df = pd.DataFrame(vals, index=df.index, columns=df.columns)

    # 整点均值重采样:每小时 6 条取均值。不用 [::6] 等距切片 —— 原始数据有
    # 重复行与断档,等距切片会让"小时"错位、把断档两侧拼在一起(数据管线 实测)。
    hourly = df.resample("1h").mean(numeric_only=True)
    n_hourly = len(hourly)

    # 连续缺失 ≥ MAX_GAP_HOURS = 真实断档:删行、切片段;片段内部短缺口插值
    col_nan = hourly[FEATURE_COLUMNS].isna().to_numpy().any(axis=1)
    keep = np.ones(len(hourly), dtype=bool)
    runs = _nan_runs(col_nan)
    gaps = [(a, b) for a, b in runs if (b - a) >= MAX_GAP_HOURS]
    for a, b in gaps:
        keep[a:b] = False
    hourly = hourly[keep].copy()

    seg_id = _segment_ids(hourly.index)
    for k in range(seg_id.max() + 1):
        m = seg_id == k
        block = hourly.loc[m, FEATURE_COLUMNS]
        if block.isna().to_numpy().any():                # 片段内部短缺口
            hourly.loc[m, FEATURE_COLUMNS] = (
                block.interpolate(method="time", limit_direction="both").ffill().bfill())
    assert not hourly.isna().to_numpy().any(), "清洗后仍有缺失值"

    dropped = sum(b - a for a, b in gaps)
    print(f"[data] 原始 {len(raw)} 行(10 分钟)→ 去重 {n_dup} 行 → 整点均值 {n_hourly} 行"
          f" → 删长断档 {dropped} 行({len(gaps)} 段)→ 有效 {len(hourly)} 行"
          f";异常值 {n_bad} 个已置缺失并插值;时间范围 {hourly.index[0]} ~ {hourly.index[-1]}")
    return hourly


def _nan_runs(mask: np.ndarray):
    """mask 中连续 True 区间的 [start, end) 列表(含端点风格与 B 实现一致)。"""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def _segment_ids(index: pd.DatetimeIndex) -> np.ndarray:
    """按相邻时间戳间隔 >1h 处断开,给每行编号所属片段。"""
    if len(index) == 0:
        return np.empty(0, dtype=np.int32)
    breaks = (np.diff(index.values).astype("timedelta64[s]").astype(np.int64)
              > 3600)
    return np.concatenate(([0], np.cumsum(breaks))).astype(np.int32)


class StandardScaler:
    """按列 z-score。只在训练段行 fit(防泄漏 #1);逆变换用于把 T 还原成摄氏度。"""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, arr: np.ndarray):
        self.mean_ = arr.mean(axis=0)
        self.std_ = arr.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0                # 防零方差列
        return self

    def transform(self, arr):
        return (arr - self.mean_) / self.std_

    def inverse_transform(self, arr):
        return arr * self.std_ + self.mean_


def to_celsius_T(y_std, scaler) -> np.ndarray:
    """只把目标列 T(下标 1)逆标准化 → 摄氏度。
    输入 [N,24] 或 [N,24,1](标准化空间),输出 float32 [N,24]。"""
    y = np.asarray(y_std, dtype=np.float64)
    if y.ndim == 3 and y.shape[-1] == 1:
        y = y[..., 0]
    return (y * scaler.std_[TARGET_IDX] + scaler.mean_[TARGET_IDX]).astype(np.float32)


def _split_windows(chunk: np.ndarray, index: pd.DatetimeIndex, lo: int, hi: int):
    """v3 语义开窗:目标 24h 完整落 [lo,hi),输入可回溯 split 前面的行,不跨片段。

    chunk/index 是清洗后的**全量**数据;[lo,hi) 是该 split 的行范围。
    窗口起点 i(全量行号)需满足:目标行 [i+72, i+95] ⊆ [lo, hi-1],
    且 96 行相邻时间差全为 1h(即输入+目标同片段,不跨断档)。
    """
    step = np.diff(index.values) == np.timedelta64(1, "h")          # 长度 n-1
    # ok[i] = 窗口起点 i 的 96 行是否连续(长度 n_all - 96 + 1)
    ok = np.convolve(step.astype(np.int32), np.ones(SEQ_IN + SEQ_OUT - 1, np.int32),
                     mode="valid") == SEQ_IN + SEQ_OUT - 1
    starts = np.where(ok)[0]                                         # 全量行号
    # 目标 24h 完整落在 split 内:目标行 [i+72, i+95] ⊆ [lo, hi-1]
    starts = starts[(starts + SEQ_IN >= lo)
                    & (starts + SEQ_IN + SEQ_OUT - 1 <= hi - 1)]
    if len(starts) == 0:
        z = np.empty((0, SEQ_IN, chunk.shape[1]), np.float32)
        return z, np.empty((0, SEQ_OUT, 1), np.float32)
    # 组装:每行起点 i,取 chunk[i:i+96] 的前 72 为 x、后 24 的 T 列为 y
    w = np.lib.stride_tricks.sliding_window_view(
        chunk, SEQ_IN + SEQ_OUT, axis=0)[starts]                     # [N, 7, 96]
    x = np.ascontiguousarray(w[:, :, :SEQ_IN].transpose(0, 2, 1), dtype=np.float32)
    y = np.ascontiguousarray(w[:, TARGET_IDX, SEQ_IN:], dtype=np.float32)[..., None]
    return x, y


def get_window_timestamps(split):
    """返回某 split 每个窗口的"预测起点"时间戳(与 _split_windows 的窗口一一对应)。

    split ∈ {"train","val","test"}。训练脚本注入时间特征时用它推算每个窗口的
    时刻/年积日(与数据管线 的 _time_features 口径一致)。
    """
    df = load_clean_dataframe()
    arr_n = len(df)
    n_tr = int(arr_n * SPLIT_RATIO[0])
    n_va = int(arr_n * SPLIT_RATIO[1])
    bounds = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va),
              "test": (n_tr + n_va, arr_n)}
    lo, hi = bounds[split]
    step = np.diff(df.index.values) == np.timedelta64(1, "h")
    ok = np.convolve(step.astype(np.int32),
                     np.ones(SEQ_IN + SEQ_OUT - 1, np.int32),
                     mode="valid") == SEQ_IN + SEQ_OUT - 1
    starts = np.where(ok)[0]
    starts = starts[(starts + SEQ_IN >= lo)
                    & (starts + SEQ_IN + SEQ_OUT - 1 <= hi - 1)]
    return df.index[starts + SEQ_IN]


def get_loaders(batch_size=BATCH_SIZE, download=True, verbose=True, csv_path=None):
    """契约入口:返回 (train, val, test, scaler)。

    样本 (x[72,7], y[24,1]),标准化空间;train 打乱,val/test 按时间顺序。
    划分按清洗后有效行的原行数 70/10/20;目标 24h 完整落 split,输入可回溯
    前一 split,窗口绝不跨片段(v3 口径,与数据管线 一致)。
    """
    if download:
        download_dataset()
    df = load_clean_dataframe(csv_path)
    arr = df.to_numpy(dtype=np.float64)
    n = len(arr)
    n_tr = int(n * SPLIT_RATIO[0])
    n_va = int(n * SPLIT_RATIO[1])
    bounds = {"train": (0, n_tr), "val": (n_tr, n_tr + n_va), "test": (n_tr + n_va, n)}

    scaler = StandardScaler().fit(arr[:n_tr])            # ← 防泄漏 #1:只用训练段行拟合

    loaders = []
    scaled = scaler.transform(arr)
    for name, (lo, hi) in bounds.items():
        x, y = _split_windows(scaled, df.index, lo, hi)
        ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
        loaders.append(DataLoader(ds, batch_size=batch_size,
                                  shuffle=(name == "train"),
                                  pin_memory=torch.cuda.is_available()))
        if verbose:
            tgt_lo = min(lo + SEQ_IN, hi)                # 第一个目标起点(信息用)
            print(f"[data] {name}:行数 {hi - lo}({df.index[lo]} ~ {df.index[hi - 1]})"
                  f" → 窗口样本 {len(ds)},x{x.shape[1:]} y{y.shape[1:]}")
    if verbose:
        print(f"[data] scaler(T 列)mean={scaler.mean_[TARGET_IDX]:.3f} "
              f"std={scaler.std_[TARGET_IDX]:.3f}(仅由训练段行拟合)")
    return tuple(loaders) + (scaler,)


if __name__ == "__main__":
    # 管线自检:下载(如缺)→ 清洗 → 建三套 loader 并打印契约信息
    get_loaders()
