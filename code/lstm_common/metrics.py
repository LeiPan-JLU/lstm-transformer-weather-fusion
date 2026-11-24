# -*- coding: utf-8 -*-
"""
全项目唯一指标实现（契约文件）。

所有模型（LSTM / Transformer / 融合 / 基线）的评估都必须调用这里的函数，
禁止各写各的。

输入约定
--------
y_true, y_pred 单位一律为 **摄氏度**，形状为 [N, 24] 或 [N, 24, 1]
（若为 [B, 24, 1] 会先自动 squeeze 掉最后一维）。

指标口径（METRIC_POOLING = "global"）
------------------------------------
把 N x 24 个误差一起池化后计算 MAE / RMSE。等价于"对所有预测步、所有样本
同等看待"。分时误差请用 per_horizon_error()，不要用 24 个 MAE 求平均去冒充
全局 RMSE（两者不相等）。

负责人：数据管线
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

__all__ = [
    "mae", "rmse", "bias", "evaluate", "per_horizon_error",
    "skill_score", "diebold_mariano", "format_table",
]


# ------------------------------------------------------------------ 工具
def _as2d(a: np.ndarray, name: str = "array") -> np.ndarray:
    """把 [N,24] / [N,24,1] / [24] 统一成 [N,24]。"""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    elif a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"{name} 形状应为 [N,24] 或 [N,24,1]，实际 {a.shape}")
    return a


def _check(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = _as2d(y_true, "y_true")
    yp = _as2d(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise ValueError(f"y_true 与 y_pred 形状不一致：{yt.shape} vs {yp.shape}")
    if not (np.isfinite(yt).all() and np.isfinite(yp).all()):
        raise ValueError("y_true / y_pred 中存在 NaN 或 inf，拒绝计算指标")
    return yt, yp


# ------------------------------------------------------------------ 单指标
def mae(y_true, y_pred) -> float:
    """平均绝对误差，单位与输入一致（摄氏度）。"""
    yt, yp = _check(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    """均方根误差，单位与输入一致（摄氏度）。对较大误差更敏感。"""
    yt, yp = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def bias(y_true, y_pred) -> float:
    """平均偏差（有符号），正数表示系统性偏高。用于诊断而非主指标。"""
    yt, yp = _check(y_true, y_pred)
    return float(np.mean(yp - yt))


def evaluate(y_true, y_pred, extra: Iterable[str] = ()) -> dict:
    """
    正式评估入口。返回 {"MAE": float, "RMSE": float}，可附加 "Bias"。

    参数
    ----
    y_true, y_pred : [N,24] 或 [N,24,1]，摄氏度
    extra : 额外的键，例如 ("Bias",)
    """
    yt, yp = _check(y_true, y_pred)
    out = {"MAE": mae(yt, yp), "RMSE": rmse(yt, yp)}
    for k in extra:
        if k == "Bias":
            out["Bias"] = bias(yt, yp)
        else:
            raise KeyError(f"不支持的 extra 指标：{k}")
    return out


# ------------------------------------------------------------------ 分时
def per_horizon_error(y_true, y_pred) -> dict:
    """
    按预测步 h = 1..24 计算误差，用于"分时误差曲线"。

    返回
    ----
    {"MAE": [24], "RMSE": [24]}
    """
    yt, yp = _check(y_true, y_pred)
    err = yt - yp
    return {
        "MAE": np.mean(np.abs(err), axis=0),
        "RMSE": np.sqrt(np.mean(err ** 2, axis=0)),
    }


# ------------------------------------------------------------------ 相对技巧
def skill_score(model_error: float, ref_error: float) -> float:
    """
    相对参考预报的技巧评分：SS = 1 - E_model / E_ref。

    SS > 0 表示比参考预报好；SS = 0 表示没有技巧；SS < 0 表示更差。
    参考预报通常取"持续法"或"季节朴素法"（气象领域惯例，见 WeatherBench）。
    """
    if ref_error <= 0:
        raise ValueError("参考预报误差必须为正")
    return float(1.0 - model_error / ref_error)


# ------------------------------------------------------------------ 显著性
def diebold_mariano(y_true, pred_a, pred_b, h: int = 24) -> dict:
    """
    Diebold-Mariano 检验：两个预报的精度是否有显著差异。

    零假设 H0：两个预报的期望损失相同。
    损失函数用绝对误差（对应 MAE 口径）。

    实现细节
    --------
    24 步预测存在严重的重叠自相关，因此用 Newey-West (Bartlett) 核估计
    长期方差，滞后阶数取 h-1。返回单侧 p 值（模型 a 优于 b）。

    参数
    ----
    y_true : [N,24]
    pred_a, pred_b : [N,24]
    h : 预测步数，用于确定截断滞后

    返回
    ----
    {"DM": 统计量, "p_value": 单侧 p 值, "mean_diff": d 的均值}
        约定 mean_diff = loss_a - loss_b，DM 为负表示 a 的损失更小（a 更好）。
    """
    yt, pa = _check(y_true, pred_a)
    _, pb = _check(y_true, pred_b)
    la = np.abs(yt - pa).ravel()
    lb = np.abs(yt - pb).ravel()
    d = la - lb
    n = d.size
    dbar = d.mean()
    # Newey-West 长期方差
    dc = d - dbar
    gamma0 = np.dot(dc, dc) / n
    var = gamma0
    for lag in range(1, h):
        w = 1.0 - lag / h
        cov = np.dot(dc[lag:], dc[:-lag]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return {"DM": float("nan"), "p_value": float("nan"), "mean_diff": float(dbar)}
    dm = dbar / np.sqrt(var / n)
    # 单侧检验：a 更好 => dm < 0 => p 应接近 0
    from math import erf, sqrt
    p = 0.5 * (1.0 + erf(dm / sqrt(2.0)))
    return {"DM": float(dm), "p_value": float(p), "mean_diff": float(dbar)}


# ------------------------------------------------------------------ 展示
def format_table(rows: dict[str, dict], ref: str | None = None,
                 ref_metric: str = "MAE") -> str:
    """
    把 {模型名: {"MAE":x,"RMSE":y}} 渲染成 markdown 表格。

    ref 不为空时，额外给出相对该参考模型的技巧评分列。
    """
    lines = []
    head = "| 模型 | MAE (℃) | RMSE (℃) |"
    sep = "|---|---:|---:|"
    if ref is not None:
        head += f" SS({ref_metric}) vs {ref} |"
        sep += "---:|"
    lines += [head, sep]
    for name, m in rows.items():
        line = f"| {name} | {m['MAE']:.4f} | {m['RMSE']:.4f} |"
        if ref is not None:
            base = rows[ref][ref_metric]
            line += f" {skill_score(m[ref_metric], base):+.2%} |"
        lines.append(line)
    return "\n".join(lines)


# ------------------------------------------------------------------ 自测
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    yt = rng.normal(10, 5, size=(200, 24))
    good = yt + rng.normal(0, 0.5, size=yt.shape)
    bad = yt + rng.normal(0, 1.5, size=yt.shape)
    print("good:", evaluate(yt, good))
    print("bad :", evaluate(yt, bad))
    ph = per_horizon_error(yt, good)
    print("per-horizon MAE[:3]:", np.round(ph["MAE"][:3], 4))
    print("DM(good vs bad):", diebold_mariano(yt, good, bad))
    print()
    print(format_table({"good": evaluate(yt, good), "bad": evaluate(yt, bad)}, ref="bad"))
