# -*- coding: utf-8 -*-
"""
train_patch.py —— 方案 B 附加实验:PatchTST 训练程序(Transformer计划.md §3.2)。

用法:
    python train_patch.py --final        # 正式训练:≤50 epoch、早停 5,发布正式产物并推 test(仅一次)
    python train_patch.py --final --stride 8 --patience 8 --epochs 30
                                         # 按 val 选好的配置正式训练(同上,带选定超参)
    python train_patch.py                # 默认只推 val,产物进 outputs/v3/experiments/ 下独占子目录
    python train_patch.py --smoke        # 冒烟:1 epoch × 30 batch,只推 val
    python train_patch.py --run-name s8 --stride 8 --patience 8
                                         # 调参变体:产物独立目录,只推理 val
                                         # (测试集留给最终选定的模型,防泄漏 #3)
    --final 与 --smoke / --run-name 互斥。

与 train_enc.py 唯一的区别是模型(PatchTST)与产物命名(patchtst_*),
数据管线 / 超参 / 早停 / 导出流程完全一致 —— 保证公平对比(README 防泄漏 #4)。
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # code/ 目录 → common 包

from common.config import (EARLY_STOP, GRAD_CLIP, LR, MAX_EPOCHS, PATCH_CFG,
                           PATCH_CKPT, SEED, WEIGHT_DECAY, get_device)
from common.data import get_loaders

from export_preds import export
from model_patch import PatchTST
from run_artifacts import resolve_run


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, opt=None, max_batches=None):
    """与 train_enc.run_epoch 相同:MSE(标准化空间),按样本数加权平均。"""
    training = opt is not None
    model.train(training)
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):           # 验证前向不建图,数值口径不变
            loss = nn.functional.mse_loss(model(x), y)
        if training:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="PatchTST 训练(附加实验)")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟模式:1 epoch、30 个训练 batch,只推 val")
    ap.add_argument("--final", action="store_true",
                    help="正式训练:发布正式 checkpoint 并推 test(与 --smoke/--run-name 互斥)")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS,
                    help="训练轮数上限")
    ap.add_argument("--patience", type=int, default=EARLY_STOP,
                    help="早停耐心值(默认 5)")
    ap.add_argument("--run-name", default=None,
                    help="实验变体名:产物独立目录,只推 val")
    ap.add_argument("--patch-len", type=int, default=None,
                    help="patch 长度 P(默认 12;论文稳健区间 8~16)")
    ap.add_argument("--stride", type=int, default=None,
                    help="切分步长 S(默认 12=直切;S<P 为重叠切分,token 更多)")
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--nhead", type=int, default=None)
    ap.add_argument("--n-layer", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    args = ap.parse_args()

    seed_everything()
    device = get_device()
    print(f"device = {device}")

    cfg = dict(PATCH_CFG)                        # 结构超参:命令行 > config 默认
    for key, val in (("patch_len", args.patch_len), ("stride", args.stride),
                     ("d_model", args.d_model), ("nhead", args.nhead),
                     ("n_layer", args.n_layer), ("d_ff", args.d_ff),
                     ("dropout", args.dropout)):
        if val is not None:
            cfg[key] = val
    diff = {k: cfg[k] for k in cfg if PATCH_CFG.get(k) != cfg[k]}
    try:
        plan = resolve_run("patchtst", PATCH_CKPT, final=args.final,
                           smoke=args.smoke, run_name=args.run_name)
    except ValueError as exc:
        ap.error(str(exc))
    if not plan.formal:
        print(f"非正式运行:只推 val,产物写入 {plan.ckpt_path.parent}")
    n_tok = (72 - cfg["patch_len"]) // cfg["stride"] + 1
    print(f"结构配置:{cfg}" + (f"(改动:{diff})" if diff else "(与默认一致)")
          + f" → {n_tok} 个 token")

    train_loader, val_loader, test_loader, scaler = get_loaders()
    model = PatchTST(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PatchTST 参数量:{n_params:,}(通道独立:实际批维 = batch × 7 通道)")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    plan.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best, patience, history = float("inf"), 0, []
    t0 = time.time()
    n_epochs = 1 if args.smoke else args.epochs
    for ep in range(1, n_epochs + 1):
        tr = run_epoch(model, train_loader, device, opt,
                       max_batches=30 if args.smoke else None)
        va = run_epoch(model, val_loader, device,
                       max_batches=10 if args.smoke else None)
        history.append({"epoch": ep, "train_mse": tr, "val_mse": va})
        mark = ""
        if va < best - 1e-5:
            best, patience, mark = va, 0, "  ← best,已保存"
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "val_mse_std": best, "epoch": ep}, plan.ckpt_path)
        else:
            patience += 1
        print(f"epoch {ep:02d} | train MSE {tr:.4f} | val MSE {va:.4f}{mark}")
        if not args.smoke and patience >= args.patience:
            print(f"早停:连续 {args.patience} 轮验证 MSE 无改善")
            break

    if best == float("inf"):     # 本轮没写出 checkpoint:不导出,也不覆盖旧正式产物
        raise RuntimeError("本轮未保存 checkpoint,未导出预测或覆盖训练历史")

    result = export("patch", data=(train_loader, val_loader, test_loader, scaler),
                    ckpt_path=plan.ckpt_path, pred_tag=plan.pred_tag,
                    splits=plan.splits, preds_dir=plan.preds_dir)
    out = {"model": "PatchTST", "cfg": cfg, "params": n_params,
           "run_name": args.run_name, "smoke": args.smoke, "final": plan.formal,
           "patience": args.patience,
           "epochs_ran": len(history), "best_val_mse_std": best,
           "train_seconds": round(time.time() - t0, 1),
           "history": history, "metrics_celsius": result}
    with open(plan.history_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"训练曲线与结果已存 {plan.history_path}")


if __name__ == "__main__":
    main()
