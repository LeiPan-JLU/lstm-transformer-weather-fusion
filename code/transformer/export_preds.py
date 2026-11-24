# -*- coding: utf-8 -*-
"""
export_preds.py —— 预测导出(README §九契约【A】交付)。

加载 checkpoint → 推理 → 反标准化成摄氏度 → 存 npy。

文件名/形状契约(契约模式,即默认调用):
    transformer_val.npy / transformer_test.npy   (encoder-only,主线)
    patchtst_val.npy   / patchtst_test.npy       (PatchTST,附加实验)
    y_val.npy / y_test.npy                       (真值,由数据管线决定,写一次三方共用)
形状一律 [N,24],单位 °C —— 消费者(fusion / 画图)不需要各自反标准化。

调参实验模式(训练脚本内部传 ckpt_path / pred_tag / splits):产物用非契约名、
只推 val,测试集留给最终选定的模型(README 防泄漏 #3:探索阶段不反复碰测试集)。

用法:
    python export_preds.py --model enc
    python export_preds.py --model patch
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from common.config import (CKPT_DIR, ENC_CFG, ENC_CKPT, ENC_TAG, PATCH_CFG,
                           PATCH_CKPT, PATCH_TAG, PREDS_DIR, get_device)
from common.data import get_loaders, get_window_timestamps, to_celsius_T
from common.metrics import evaluate

import model_enc
import model_patch

_REGISTRY = {
    "enc":   {"cls": model_enc.TemporalEncoderTransformer, "cfg": ENC_CFG,
              "ckpt": ENC_CKPT, "tag": ENC_TAG},
    "patch": {"cls": model_patch.PatchTST,                 "cfg": PATCH_CFG,
              "ckpt": PATCH_CKPT, "tag": PATCH_TAG},
}


@torch.no_grad()
def _predict(model, loader, device, tf=None):
    """整段推理 → (pred_std, true_std),形状 [N,24](标准化空间)。
    tf 非 None 时按窗口把 4 列时间特征拼到每个时间步后(与训练一致)。"""
    model.eval()
    preds, trues = [], []
    i = 0
    for x, y in loader:
        xb = x.to(device, non_blocking=True)
        if tf is not None:
            n = xb.shape[0]
            t = torch.from_numpy(tf[i:i + n]).unsqueeze(1).expand(
                -1, xb.shape[1], -1).to(device)
            xb = torch.cat([xb, t], dim=2)
            i += n
        p = model(xb)
        preds.append(p.squeeze(-1).cpu().numpy())
        trues.append(y.squeeze(-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


def _save_npy(path, arr):
    """真值/预测落盘:已存在且一致则跳过(另一侧已写过),不一致则警告后覆盖。"""
    if path.exists():
        old = np.load(path)
        if old.shape == arr.shape and np.allclose(old, arr, atol=1e-4):
            return
        print(f"  ! {path.name} 已存在但内容不同(模型或数据管线变过)→ 覆盖")
    np.save(path, arr)


def export(which, data=None, ckpt_path=None, pred_tag=None, splits=("val", "test"),
           preds_dir=None):
    """导出一个模型的预测与指标。

    which ∈ {"enc", "patch"};data = get_loaders() 四元组(可复用避免重建)。
    默认契约模式:npy 前缀 = transformer / patchtst,val+test 各推理一次,
    写 y_*.npy 到 outputs/v3/preds(新口径目录,旧 npy 不覆盖)。
    实验模式:pred_tag 指定非契约前缀(如 transformer_L3),preds_dir 指定实验目录,
    splits 只给 ("val",)。返回 {split: {"MAE":…,"RMSE":…}, "n_<split>":…}。
    """
    reg = _REGISTRY[which]
    device = get_device()
    if data is None:
        _, val_loader, test_loader, scaler = get_loaders(download=False)
    else:
        _, val_loader, test_loader, scaler = data
    loaders = {"val": val_loader, "test": test_loader}

    ckpt_path = Path(ckpt_path) if ckpt_path else CKPT_DIR / reg["ckpt"]
    blob = torch.load(ckpt_path, map_location=device)
    cfg = blob.get("cfg", reg["cfg"])                # 用存档时的结构,防 config 事后被改
    model = reg["cls"](**cfg).to(device)
    model.load_state_dict(blob["model"])
    tag = pred_tag or reg["tag"]
    out_dir = Path(preds_dir) if preds_dir else PREDS_DIR
    print(f"[{tag}] 加载 {ckpt_path.name}(epoch {blob.get('epoch')}),开始推理…")

    use_time = blob.get("time_features", cfg.get("n_feat") == 11)
    per_step = bool(blob.get("cfg", {}).get("per_step_clock",
                                            cfg.get("per_step_clock", False)))
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for split in splits:
        loader = loaders[split]
        tf = None
        # 训练传入的 11 维 loader 已增强；仅给裸 7 维窗口补时间编码。
        # per_step_clock:每条历史观测带自己的真实钟点(起点 -71h…0h);
        # 否则:72 步复制同一份"预测起点"编码。
        if use_time and loader.dataset.tensors[0].shape[-1] == 7:
            ts = get_window_timestamps(split)
            if per_step:
                offs = np.arange(-loader.dataset.tensors[0].shape[1] + 1, 1)
                steps = ts.to_numpy()[:, None] + offs[None, :] * np.timedelta64(1, "h")
                flat = pd.DatetimeIndex(steps.ravel())
                tod = flat.hour.to_numpy() + flat.minute.to_numpy() / 60.0
                doy = flat.dayofyear.to_numpy() - 1
                tf = np.stack(
                    [np.sin(2 * np.pi * tod / 24.0), np.cos(2 * np.pi * tod / 24.0),
                     np.sin(2 * np.pi * doy / 365.0), np.cos(2 * np.pi * doy / 365.0)],
                    axis=1).reshape(len(ts), -1, 4).astype(np.float32)
            else:
                tod = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
                doy = ts.dayofyear.to_numpy() - 1
                tf = np.stack(
                    [np.sin(2 * np.pi * tod / 24.0), np.cos(2 * np.pi * tod / 24.0),
                     np.sin(2 * np.pi * doy / 365.0), np.cos(2 * np.pi * doy / 365.0)],
                    axis=1).astype(np.float32)
        pred_std, true_std = _predict(model, loader, device, tf=tf)
        pred_c = to_celsius_T(pred_std, scaler)      # 反标准化 → 摄氏度
        true_c = to_celsius_T(true_std, scaler)
        _save_npy(out_dir / f"{tag}_{split}.npy", pred_c)
        if pred_tag is None:                         # 只有契约模式写真值文件
            _save_npy(out_dir / f"y_{split}.npy", true_c)
        result[split] = evaluate(pred_c, true_c)
        result[f"n_{split}"] = len(pred_c)
        print(f"  {split}:n={len(pred_c)}  MAE={result[split]['MAE']:.3f}°C  "
              f"RMSE={result[split]['RMSE']:.3f}°C")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="导出预测 npy(摄氏度)")
    ap.add_argument("--model", choices=["enc", "patch"], required=True,
                    help="正式导出该模型的 checkpoint 到 outputs/v3/preds(val+test)")
    args = ap.parse_args()
    export(args.model)
