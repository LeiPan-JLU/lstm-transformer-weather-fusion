# -*- coding: utf-8 -*-
"""
fusion.py —— 可学习加权融合(README §4.3 / Transformer计划.md 的最后一块拼图)。

输入:三方预测 npy(摄氏度,[N,24],行行对齐 —— v3 口径下两套窗口逐行一致)
    Transformer: outputs/v3/preds/transformer_{val,test}.npy   (A,本方主线)
    LSTM:        outputs/preds/lstm_time_res_{val,test}.npy
    真值:        outputs/v3/preds/y_{val,test}.npy
纪律:权重只用验证集学,测试集只在最后跑一次(README 防泄漏 #3)。

三档递进(复杂度递增,全部闭式解或凸优化,无随机性):
    eq   等权平均          —— baseline,不学习
    alpha 全局标量 α       —— 1 个参数,minimize val MSE(黄金分割搜索,凸)
    alpha_t 逐时距 α_h     —— 24 个参数,每个预测小时独立求闭式最优权重
                             (逐时距最小二乘,带岭正则防过拟合)

输出:outputs/v3/fusion/fusion_report.json + 各融合预测 npy。
"""
import json
import sys
from pathlib import Path

import numpy as np

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from common.config import PREDS_DIR, PROJECT_ROOT
from common.metrics import evaluate

B_PREDS = PROJECT_ROOT / "outputs" / "preds"
OUT_DIR = PROJECT_ROOT / "outputs" / "v3" / "fusion"


def load_pair(split):
    """读某 split 的 Transformer/LSTM 预测与真值,校验形状一致。"""
    pT = np.load(PREDS_DIR / f"transformer_{split}.npy")
    pL = np.load(B_PREDS / f"lstm_time_res_{split}.npy")
    y = np.load(PREDS_DIR / f"y_{split}.npy")
    assert pT.shape == pL.shape == y.shape, \
        f"{split} 形状不一致: T{pT.shape} L{pL.shape} y{y.shape}"
    return pT.astype(np.float64), pL.astype(np.float64), y.astype(np.float64)


def fuse_eq(pT, pL):
    return (pT + pL) / 2


def fuse_alpha(pT, pL, y):
    """标量 α:p = α·T + (1-α)·L,α* = argmin val MSE(一维凸,黄金分割搜索)。"""
    def f(a):
        return ((a * pT + (1 - a) * pL - y) ** 2).mean()
    lo, hi, gr = 0.0, 1.0, (np.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    for _ in range(60):
        if f(c) < f(d):
            hi, d = d, c
            c = hi - gr * (hi - lo)
        else:
            lo, c = c, d
            d = lo + gr * (hi - lo)
    return (lo + hi) / 2


def fuse_alpha_t(pT, pL, y, ridge=1e-6):
    """逐时距 α_h:对每个预测小时 h 独立做一维岭最小二乘(闭式)。

    目标:min_α ||α·T_h + (1-α)·L_h − y_h||² + ridge·(α−0.5)²
    d/dα = 0 → α* = (Σ(T−L)(y−L) + 0.5·ridge·N) / (Σ(T−L)² + ridge·N),再截到 [0,1]。
    逐时距独立 ⇒ 参数 24 个,验证样本 6886 ⇒ 远不过拟合;ridge 只防数值退化。
    """
    d = pT - pL                                   # [N,24]
    num = ((y - pL) * d).sum(axis=0) + 0.5 * ridge * d.shape[0]
    den = (d * d).sum(axis=0) + ridge * d.shape[0]
    a = np.clip(num / den, 0.0, 1.0)
    return a


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pT_va, pL_va, y_va = load_pair("val")
    report = {"inputs": {"transformer": "outputs/v3/preds/transformer_*.npy",
                         "lstm": "outputs/preds/lstm_time_res_*.npy",
                         "n_val": len(y_va)}}

    # ---- 只用验证集学权重 ----
    a_scalar = fuse_alpha(pT_va, pL_va, y_va)
    a_t = fuse_alpha_t(pT_va, pL_va, y_va)
    report["learned_on_val"] = {
        "alpha_scalar": a_scalar,
        "alpha_t_first6": [round(x, 4) for x in a_t[:6]],
        "alpha_t_last6": [round(x, 4) for x in a_t[-6:]],
    }
    print(f"标量 α = {a_scalar:.4f}")
    print(f"逐时距 α(1h..24h) = "
          f"{' '.join(f'{x:.3f}' for x in a_t)}")

    # ---- val 上的三档表现(选型依据) ----
    val_rows = {}
    for name, pred in [
        ("transformer_alone", pT_va), ("lstm_alone", pL_va),
        ("eq", fuse_eq(pT_va, pL_va)),
        ("alpha", a_scalar * pT_va + (1 - a_scalar) * pL_va),
        ("alpha_t", a_t[None, :] * pT_va + (1 - a_t[None, :]) * pL_va),
    ]:
        val_rows[name] = {"MAE": evaluate(pred, y_va)["MAE"],
                          "RMSE": evaluate(pred, y_va)["RMSE"]}
        print(f"val  {name:<18} MAE {val_rows[name]['MAE']:.4f}  "
              f"RMSE {val_rows[name]['RMSE']:.4f}")
    report["val"] = val_rows

    best_val = min(val_rows, key=lambda k: val_rows[k]["MAE"])
    print(f"→ 按 val MAE 选定融合方式:{best_val}")
    report["selected"] = best_val

    # ---- 测试集:只跑一次,只跑选定的那档(以及单独模型作报告对照) ----
    pT_te, pL_te, y_te = load_pair("test")
    fused_by = {"eq": fuse_eq(pT_te, pL_te),
                "alpha": a_scalar * pT_te + (1 - a_scalar) * pL_te,
                "alpha_t": a_t[None, :] * pT_te + (1 - a_t[None, :]) * pL_te}
    fused_te = fused_by[best_val]
    test_rows = {}
    for name, pred in [("transformer_alone", pT_te), ("lstm_alone", pL_te),
                       ("fused_selected", fused_te)]:
        test_rows[name] = {"MAE": evaluate(pred, y_te)["MAE"],
                           "RMSE": evaluate(pred, y_te)["RMSE"]}
        print(f"test {name:<18} MAE {test_rows[name]['MAE']:.4f}  "
              f"RMSE {test_rows[name]['RMSE']:.4f}")
    report["test"] = test_rows
    report["n_test"] = len(y_te)

    # ---- 落盘 ----
    fused_va_by = {"eq": fuse_eq(pT_va, pL_va),
                   "alpha": a_scalar * pT_va + (1 - a_scalar) * pL_va,
                   "alpha_t": a_t[None, :] * pT_va + (1 - a_t[None, :]) * pL_va}
    np.save(OUT_DIR / "fusion_val.npy", fused_va_by[best_val])
    np.save(OUT_DIR / "fusion_test.npy", fused_te)
    with open(OUT_DIR / "fusion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"融合产物已存 {OUT_DIR}")


if __name__ == "__main__":
    main()
