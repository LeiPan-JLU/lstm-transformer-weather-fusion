# -*- coding: utf-8 -*-
"""
把训练好的 LSTM 在验证集 / 测试集上各推理一次，反标准化成摄氏度后存盘。

契约产物（形状一律 [N, 24]，单位 ℃）
------------------------------------
    outputs/preds/lstm_val.npy
    outputs/preds/lstm_test.npy
消融实验用 --out-name 指定别的名字，例如 lstm_time_test.npy。

注意：本脚本只做推理，**不参与任何选择**。测试集只在这里被前向一次。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from lstm_common import config as C          # noqa: E402
from lstm_common import data as D            # noqa: E402
from lstm_common import metrics as M         # noqa: E402
from lstm.model import build_model      # noqa: E402
from lstm.train import pick_device, predict_std  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="默认 outputs/checkpoints/<tag>_best.pt")
    ap.add_argument("--tag", default="lstm")
    ap.add_argument("--out-name", default=None, help="预测文件名前缀，默认与 tag 相同")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else (C.CKPT_DIR / f"{args.tag}_best.pt")
    if not ckpt.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{ckpt}，请先跑 train.py")
    out_name = args.out_name or args.tag
    device = pick_device(args.device)

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    hp = ck["hparams"]
    model = build_model(
        input_size=ck["input_size"],
        hidden_size=hp["hidden"],
        num_layers=hp["layers"],
        dropout=hp["dropout"],
        pred_len=C.PRED_LEN,
        forget_bias_init=hp.get("forget_bias", C.FORGET_BIAS_INIT),
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    hd = D.build_hourly()
    scaler = ck["scaler"]
    report = {"checkpoint": str(ckpt), "tag": args.tag, "best_epoch": ck.get("epoch")}

    for split in ("val", "test"):
        arr = D.get_arrays(split, hd=hd, scaler=scaler,
                           use_time_features=ck["use_time_features"],
                           residual=ck["residual"])
        p_std = predict_std(model, arr["x"], device)
        p_c = D.unscale_y(p_std, arr)
        np.save(C.PRED_DIR / f"{out_name}_{split}.npy", p_c.astype(np.float64))

        # 对齐检查：这里算出的指标必须与训练日志里的一致
        m = M.evaluate(arr["y_celsius"], p_c)
        report[split] = m
        ref = ck.get("val_metrics") if split == "val" else None
        if ref is not None:
            report["val_matches_checkpoint"] = bool(
                abs(ref["MAE"] - m["MAE"]) < 1e-6 and abs(ref["RMSE"] - m["RMSE"]) < 1e-6)
        print(f"{split:5s} MAE {m['MAE']:.4f}  RMSE {m['RMSE']:.4f}  (N={len(p_c)})")

    (C.LOG_DIR / f"{out_name}_export.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写出 {C.PRED_DIR}\\{out_name}_val.npy / {out_name}_test.npy")


if __name__ == "__main__":
    main()
    C.safe_shutdown()