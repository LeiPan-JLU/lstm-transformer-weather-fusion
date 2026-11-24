# -*- coding: utf-8 -*-
"""
数据处理与 DataLoader（契约文件）。

本文件实现 README §5.2 / §5.3 的全部预处理约定，并额外处理了原始数据里
README 没有提到的三类问题（数据管线 实测发现，详见 outputs/logs/data_qc.json）：

1. **327 条完全重复行**（记录仪重启造成：2010-07-01、2014-03-20、2014-03-21）。
2. **5 段真实断档**（30min / 20min / 20min / 16h / 74h），其中 74h 那段
   （2016-10-25 10:30 → 2016-10-28 12:50）**落在测试集内**。若按普通滑窗，
   模型会把相隔 3 天多的两个观测当成"相邻小时"，物理上不成立。
3. `-9999` 不只出现在 `wv (m/s)`（18 个），`max. wv (m/s)` 也有 20 个。

处理策略（已与全组确认）
------------------------
去重 → `-9999` 置缺失 → 按小时重采样 →
用 **缺失长度 >= MAX_GAP_HOURS 的连续缺失** 把序列切成互不相连的片段：
    * 长缺口不插值，直接从序列中删除；
    * 片段内部的零星缺失（< MAX_GAP_HOURS）用时间插值补；
    * 滑窗绝不跨片段，因此跨 74 小时缺口的样本根本不存在。

划分与滑窗（严格按契约）
------------------------
* 时间顺序 70/10/20。样本 i 属于某 split，当且仅当**预测目标 [i, i+PRED_LEN)
  完整落在该 split**；输入窗口 [i-SEQ_LEN, i) 允许回溯到前一个 split
  （README §5.2 允许，且这符合真实业务：预测时能拿到全部历史）。
* StandardScaler 只在训练段的行上 fit。

两份小时级表示
--------------
* ``compact``：断档小时已删除，供滑窗使用（保证任何窗口都不含 NaN）。
* ``regular``：完整 1 小时网格，断档处线性插值，仅供**基线方法**与画图使用。
  基线以 `t_start` 时间戳对齐到 LSTM 的样本集合，因此不会引入不公平。

负责人：数据管线
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from lstm_common import config as C  # noqa: E402

Split = Literal["train", "val", "test"]


# =================================================================== 读取与清洗
def load_raw() -> pd.DataFrame:
    """读原始 CSV：解析时间、排序、去重、把 -9999 变成 NaN。"""
    if not C.RAW_CSV.exists():
        raise FileNotFoundError(
            f"找不到原始数据 {C.RAW_CSV}\n"
            f"请先下载 {C.RAW_URL} 并解压到 {C.DATA_DIR}"
        )
    df = pd.read_csv(C.RAW_CSV)
    df[C.DATETIME_COL] = pd.to_datetime(df[C.DATETIME_COL], format=C.DATETIME_FMT)
    df = df.sort_values(C.DATETIME_COL, kind="mergesort").reset_index(drop=True)

    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    num_cols = [c for c in df.columns if c != C.DATETIME_COL]
    n_bad = {c: int((df[c] == C.BAD_VALUES[0]).sum()) for c in num_cols}
    for c in num_cols:
        df[c] = df[c].replace(list(C.BAD_VALUES), np.nan)

    df = df.set_index(C.DATETIME_COL)
    df.attrs["n_duplicates_dropped"] = n_dup
    df.attrs["n_bad_values"] = {k: v for k, v in n_bad.items() if v}
    return df


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    降采样为逐小时（按时间戳重采样取均值）。

    **为什么不用 `df[::6]`**：原始数据有 327 条重复行和 5 段断档，等距切片会让
    "小时"逐渐错位，还会把断档两侧拼在一起。按时间戳重采样不会。
    """
    return df.resample("1h").mean(numeric_only=True)


def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """返回 mask 中连续 True 区间的 [start, end) 列表。"""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


# =================================================================== 主容器
@dataclass
class HourlyData:
    """清洗后的小时级数据容器。"""
    compact: pd.DataFrame          # 断档小时已删除，供滑窗
    regular: pd.DataFrame          # 完整网格（断档处插值），供基线/画图
    qc: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.compact)


def _segment_ids(index: pd.DatetimeIndex, gap: pd.Timedelta = pd.Timedelta("1h")) -> np.ndarray:
    """按相邻时间戳的间隔给片段编号。间隔 > gap 处断开。"""
    if len(index) == 0:
        return np.empty(0, dtype=np.int32)
    breaks = np.diff(index.values).astype("timedelta64[s]").astype(np.int64) > int(gap.total_seconds())
    return np.concatenate(([0], np.cumsum(breaks))).astype(np.int32)


@lru_cache(maxsize=1)
def build_hourly() -> HourlyData:
    """
    完整清洗流程：返回 HourlyData(compact, regular, qc)。

    只读一次盘，结果在进程内缓存。
    """
    raw = load_raw()
    n_raw = len(raw)
    hourly = to_hourly(raw)

    missing = hourly[C.TARGET_COL].isna().to_numpy()
    runs = find_runs(missing)
    long_runs = [(a, b) for a, b in runs if (b - a) >= C.MAX_GAP_HOURS]

    gap_info = []
    for a, b in runs:
        length = b - a
        gap_info.append({
            "start": str(hourly.index[a]),
            "end": str(hourly.index[b - 1]),
            "missing_hours": int(length),
            "action": "drop_and_split" if length >= C.MAX_GAP_HOURS else "interpolate",
        })

    # ---- compact：删掉长缺口所在的小时
    keep = np.ones(len(hourly), dtype=bool)
    for a, b in long_runs:
        keep[a:b] = False
    compact = hourly[keep].copy()

    # 片段编号（删完之后，相隔 >1h 的位置就是片段边界）
    seg_id = _segment_ids(compact.index)
    compact["seg_id"] = seg_id

    # 片段内部插值（短缺口）
    feats = [c for c in C.FEATURES if c in compact.columns]
    for k in range(seg_id.max() + 1 if len(seg_id) else 0):
        m = seg_id == k
        block = compact.loc[m, feats]
        if block.isna().to_numpy().any():
            filled = (block.interpolate(method="time", limit_direction="both")
                           .ffill().bfill())
            compact.loc[m, feats] = filled

    if compact[feats].isna().to_numpy().any():
        raise RuntimeError("清洗后仍有缺失值，请检查断档处理逻辑")

    # ---- regular：完整网格，供基线
    regular = hourly.copy()
    regular[feats] = (regular[feats].interpolate(method="time", limit_direction="both")
                                      .ffill().bfill())

    segments = [
        {"seg_id": int(k), "start": str(compact.index[m][0]), "end": str(compact.index[m][-1]),
         "hours": int(m.sum())}
        for k in range(seg_id.max() + 1 if len(seg_id) else 0)
        for m in [seg_id == k]
    ]

    qc = {
        "raw_rows": int(n_raw),
        "duplicates_dropped": int(raw.attrs.get("n_duplicates_dropped", 0)),
        "bad_values_found": raw.attrs.get("n_bad_values", {}),
        "time_range": [str(raw.index.min()), str(raw.index.max())],
        "hourly_rows_before": int(len(hourly)),
        "hourly_rows_after_drop": int(len(compact)),
        "hours_dropped_in_gaps": int(len(hourly) - len(compact)),
        "gaps": gap_info,
        "segments": segments,
        "n_segments": len(segments),
        "target_stats": {
            "mean": float(compact[C.TARGET_COL].mean()),
            "std": float(compact[C.TARGET_COL].std()),
            "min": float(compact[C.TARGET_COL].min()),
            "max": float(compact[C.TARGET_COL].max()),
        },
    }
    return HourlyData(compact=compact, regular=regular, qc=qc)


# =================================================================== 划分
def split_bounds(n: int) -> dict[str, tuple[int, int]]:
    """按时间顺序 70/10/20 切出三段的 [lo, hi) 小时下标。"""
    r_tr, r_va, _ = C.SPLIT_RATIO
    i_tr = int(n * r_tr)
    i_va = int(n * (r_tr + r_va))
    return {"train": (0, i_tr), "val": (i_tr, i_va), "test": (i_va, n)}


def window_indices(split: Split, seg_id: np.ndarray, bounds: dict[str, tuple[int, int]]) -> np.ndarray:
    """
    该 split 内所有合法滑窗的起点 i。

    合法性 = 三条同时成立：
      1) 预测目标 [i, i+PRED_LEN) 全在该 split 内；
      2) 输入 [i-SEQ_LEN, i) 与目标属于同一个片段（不跨断档）；
      3) 有足够历史（i >= SEQ_LEN）。
    """
    lo, hi = bounds[split]
    start_min = max(lo, C.SEQ_LEN)
    start_max = hi - C.PRED_LEN
    if start_max < start_min:
        return np.empty(0, dtype=np.int64)
    i = np.arange(start_min, start_max + 1, C.STRIDE, dtype=np.int64)
    ok = (seg_id[i - C.SEQ_LEN] == seg_id[i]) & (seg_id[i + C.PRED_LEN - 1] == seg_id[i])
    return i[ok]


# =================================================================== 标准化
def fit_scaler(compact: pd.DataFrame, train_hi: int) -> dict:
    """在训练段的行上拟合逐特征 z-score（绝不会看到验证/测试段）。"""
    block = compact[C.FEATURES].iloc[:train_hi].to_numpy(dtype=np.float64)
    mean = block.mean(axis=0)
    scale = block.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "target_idx": C.TARGET_IDX,
        "target_mean": float(mean[C.TARGET_IDX]),
        "target_scale": float(scale[C.TARGET_IDX]),
        "fitted_on_rows": int(train_hi),
        "features": list(C.FEATURES),
    }


def _time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """时刻（24h）与年积日（365d）的 sin-cos 编码，形状 [T, 4]。"""
    tod = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    doy = index.dayofyear.to_numpy() - 1
    return np.stack([
        np.sin(2 * np.pi * tod / 24.0),
        np.cos(2 * np.pi * tod / 24.0),
        np.sin(2 * np.pi * doy / 365.0),
        np.cos(2 * np.pi * doy / 365.0),
    ], axis=1)


# =================================================================== 组装数组
def get_arrays(
    split: Split,
    hd: HourlyData | None = None,
    scaler: dict | None = None,
    use_time_features: bool | None = None,
    residual: bool | None = None,
) -> dict:
    """
    取某个 split 的全部数组（全部无 NaN）。

    x         [N, SEQ_LEN, F] 或 [N, SEQ_LEN, F+4]，标准化空间
    y         [N, PRED_LEN, 1]，标准化空间（残差模式下是标准化增量）
    y_celsius [N, PRED_LEN]，摄氏度真值
    t_last    [N]，窗口最后一个观测的气温（℃）
    t_start   [N]，第一个被预测小时的时刻（int64 纳秒）
    idx       [N]，窗口起点在 compact 表中的行号
    """
    hd = build_hourly() if hd is None else hd
    compact = hd.compact
    use_time_features = C.USE_TIME_FEATURES if use_time_features is None else use_time_features
    residual = C.USE_RESIDUAL if residual is None else residual

    bounds = split_bounds(len(compact))
    if scaler is None:
        scaler = fit_scaler(compact, bounds["train"][1])

    seg_id = compact["seg_id"].to_numpy()
    idx = window_indices(split, seg_id, bounds)

    feats = compact[C.FEATURES].to_numpy(dtype=np.float64)
    mu = np.asarray(scaler["mean"])
    sd = np.asarray(scaler["scale"])
    z = (feats - mu) / sd
    if use_time_features:
        z = np.concatenate([z, _time_features(compact.index)], axis=1)

    x = np.stack([z[i - C.SEQ_LEN:i] for i in idx]).astype(np.float32)
    y_raw = np.stack([feats[i:i + C.PRED_LEN, C.TARGET_IDX] for i in idx]).astype(np.float64)
    t_last = feats[idx - 1, C.TARGET_IDX].astype(np.float64)

    if residual:
        y_std = (y_raw - t_last[:, None]) / scaler["target_scale"]
    else:
        y_std = (y_raw - scaler["target_mean"]) / scaler["target_scale"]

    return {
        "x": x,
        "y": y_std.astype(np.float32)[..., None],
        "y_celsius": y_raw,
        "t_last": t_last,
        "t_start": compact.index[idx].to_numpy().astype("datetime64[ns]").view("int64"),
        "idx": idx,
        "residual": residual,
        "use_time_features": use_time_features,
        "scaler": scaler,
    }


def unscale_y(pred_std: np.ndarray, arr: dict) -> np.ndarray:
    """把模型输出的标准化预测还原成摄氏度（残差模式自动加回 t_last）。"""
    sc = arr["scaler"]
    p = np.asarray(pred_std, dtype=np.float64)
    if p.ndim == 3:
        p = p[..., 0]
    if arr["residual"]:
        return p * sc["target_scale"] + np.asarray(arr["t_last"])[:, None]
    return p * sc["target_scale"] + sc["target_mean"]


# =================================================================== DataLoader
def get_loaders(
    batch_size: int | None = None,
    use_time_features: bool | None = None,
    residual: bool | None = None,
    num_workers: int = 0,
    hd: HourlyData | None = None,
):
    """
    契约函数：返回 (train_loader, val_loader, test_loader, scaler)。

    * 样本 = (x[72,F], y[24,1])，全部为标准化空间；
    * train 打乱，val / test 不打乱（保证顺序与 t_start 对齐）。
    附加数组请用 get_arrays() 单独取。
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    hd = build_hourly() if hd is None else hd
    bounds = split_bounds(len(hd.compact))
    scaler = fit_scaler(hd.compact, bounds["train"][1])

    class _DS(Dataset):
        def __init__(self, arr: dict):
            self.x = torch.from_numpy(arr["x"])
            self.y = torch.from_numpy(arr["y"])

        def __len__(self) -> int:
            return len(self.x)

        def __getitem__(self, i):
            return self.x[i], self.y[i]

    bs = batch_size or C.BATCH_SIZE
    arrays = {s: get_arrays(s, hd=hd, scaler=scaler,
                            use_time_features=use_time_features, residual=residual)
              for s in ("train", "val", "test")}

    loaders = {
        s: DataLoader(_DS(arrays[s]), batch_size=bs, shuffle=(s == "train"),
                      drop_last=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())
        for s in ("train", "val", "test")
    }
    return loaders["train"], loaders["val"], loaders["test"], scaler


# =================================================================== 命令行
def main() -> None:
    """生成清洗后的数据 + 质量报告 + 全组共用的 y_val / y_test。"""
    hd = build_hourly()
    qc = dict(hd.qc)
    compact = hd.compact
    n = len(compact)
    bounds = split_bounds(n)
    qc["split"] = {
        s: {"hours": int(hi - lo), "start": str(compact.index[lo]), "end": str(compact.index[hi - 1])}
        for s, (lo, hi) in bounds.items()
    }
    qc["split"]["test_period_mean_T"] = float(compact[C.TARGET_COL].iloc[bounds["test"][0]:bounds["test"][1]].mean())
    qc["split"]["val_period_mean_T"] = float(compact[C.TARGET_COL].iloc[bounds["val"][0]:bounds["val"][1]].mean())
    qc["split"]["train_period_mean_T"] = float(compact[C.TARGET_COL].iloc[bounds["train"][0]:bounds["train"][1]].mean())

    seg_id = compact["seg_id"].to_numpy()
    qc["windows"] = {}
    for s in ("train", "val", "test"):
        idx = window_indices(s, seg_id, bounds)
        qc["windows"][s] = {
            "n_samples": int(len(idx)),
            "first_target_time": str(compact.index[idx[0]]) if len(idx) else None,
            "last_target_time": str(compact.index[idx[-1] + C.PRED_LEN - 1]) if len(idx) else None,
        }
        # 检查：有没有样本的输入跨了片段
        cross = int((seg_id[idx - C.SEQ_LEN] != seg_id[idx]).sum()) if len(idx) else 0
        qc["windows"][s]["samples_crossing_segment"] = cross

    qc["scaler"] = fit_scaler(compact, bounds["train"][1])

    C.CLEAN_SUMMARY_JSON.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    compact.reset_index().to_csv(C.HOURLY_CSV, index=False, encoding="utf-8-sig")
    try:
        compact.reset_index().to_parquet(C.HOURLY_PARQUET, index=False)
    except Exception:
        pass

    for s in ("val", "test"):
        arr = get_arrays(s, hd=hd)
        np.save(C.PRED_DIR / f"y_{s}.npy", arr["y_celsius"])
        np.save(C.PRED_DIR / f"t_start_{s}.npy", arr["t_start"])
        np.save(C.PRED_DIR / f"t_last_{s}.npy", arr["t_last"])

    print(json.dumps(qc, ensure_ascii=False, indent=2))
    print(f"\n已写出 {C.CLEAN_SUMMARY_JSON}")
    print(f"已写出 {C.HOURLY_CSV}")
    print(f"已写出 {C.PRED_DIR}\\y_val.npy / y_test.npy / t_start_*.npy / t_last_*.npy")


if __name__ == "__main__":
    main()
    C.safe_shutdown()