# -*- coding: utf-8 -*-
"""
train_enc.py —— 方案 A 主线:encoder-only Transformer 训练程序(Transformer计划.md §3/§5)。

用法:
    python train_enc.py --final        # 正式训练:≤50 epoch、早停 5,发布正式产物并推 test(仅一次)
    python train_enc.py --final --n-layer 3 --patience 8 --epochs 30
                                       # 按 val 选好的配置正式训练(同上,带选定超参)
    python train_enc.py                # 默认只推 val,产物进 outputs/v3/experiments/ 下独占子目录
    python train_enc.py --smoke        # 冒烟:1 epoch × 30 batch,只推 val
    python train_enc.py --run-name L3 --n-layer 3 --patience 8
                                       # 调参变体:产物独立目录,只推理 val
                                       # (测试集留给最终选定的模型,防泄漏 #3)
    --final 与 --smoke / --run-name 互斥。

流程:get_loaders() 拿标准化数据 → MSE(标准化空间)训练 → 验证集选型 + 早停
→ 存 checkpoint → export_preds 导出摄氏度 npy → evaluate() 报 MAE/RMSE。
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # code/ 目录 → common 包

from common.config import (EARLY_STOP, ENC_CFG, ENC_CKPT, FEATURE_COLUMNS, GRAD_CLIP, LR,
                           MAX_EPOCHS, SEED, TARGET_IDX, WEIGHT_DECAY, get_device)
from common.data import get_loaders, get_window_timestamps

from export_preds import export
from model_enc import TemporalEncoderTransformer
from run_artifacts import resolve_run


def seed_everything(seed=SEED):
    """固定随机源:Python / numpy / torch(含 CUDA)。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _time_features(ts: pd.DatetimeIndex) -> np.ndarray:
    """时刻(24h)与年积日(365d)的 sin-cos 编码 [T,4](与数据管线 口径一致)。"""
    tod = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    doy = ts.dayofyear.to_numpy() - 1
    return np.stack([
        np.sin(2 * np.pi * tod / 24.0), np.cos(2 * np.pi * tod / 24.0),
        np.sin(2 * np.pi * doy / 365.0), np.cos(2 * np.pi * doy / 365.0),
    ], axis=1)


def _add_time_features(*loaders, per_step=False):
    """给每个 Dataset 的 x 拼上 4 列时间特征(按窗口预测起点时刻计算)。

    窗口顺序 = get_window_timestamps() 返回顺序(都与 _split_windows 一致),
    train 打乱发生在 DataLoader 层,不影响 Dataset 行序,因此按行号对齐安全。
    per_step=False(基线):72 个时间步复制同一份"预测起点"编码;
    per_step=True(方案4):每条历史观测带自己的真实钟点/年积日 —— 预测起点
    往前推 71…0 小时,让模型看到"这次升温发生在早晨还是晚上"。
    """
    out = []
    for split, loader in zip(("train", "val", "test"), loaders):
        ts = get_window_timestamps(split)
        ds = loader.dataset
        x, y = ds.tensors
        assert len(x) == len(ts), f"{split}: 窗口数 {len(x)} 与时间戳 {len(ts)} 不齐"
        if per_step:
            # [N,72] 每列 = 起点 - (71…0) 小时
            offs = pd.timedelta_range(end="0h", periods=x.shape[1], freq="-1h")
            steps = ts.to_numpy()[:, None] + offs.to_numpy()[None, :]
            flat = pd.DatetimeIndex(steps.ravel())
            tf = _time_features(flat).reshape(len(ts), x.shape[1], 4)
        else:
            tf4 = _time_features(ts).astype(np.float32)                  # [N,4]
            tf = np.broadcast_to(tf4[:, None, :], (len(ts), x.shape[1], 4))
        tf = torch.from_numpy(np.ascontiguousarray(tf, dtype=np.float32))
        x2 = torch.cat([x, tf], dim=2)                                   # [N,72,11]
        out.append(DataLoader(TensorDataset(x2, y), batch_size=loader.batch_size,
                              shuffle=(loader.sampler is not None
                                       and isinstance(loader.sampler, RandomSampler)),
                              pin_memory=loader.pin_memory))
    return tuple(out)


def run_epoch(model, loader, device, opt=None, max_batches=None):
    """一个 epoch:opt 非 None 时训练,否则纯评估。
    返回按样本数加权的 MSE(标准化空间 —— 训练损失所在空间,README §九)。"""
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
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)   # 防早期注意力梯度爆炸
            opt.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="encoder-only Transformer 训练")
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
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--nhead", type=int, default=None)
    ap.add_argument("--n-layer", type=int, default=None)
    ap.add_argument("--d-ff", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--head", choices=["flatten", "pool"], default=None,
                    help="预测头:flatten=主线大头,pool=均值池化精简头")
    ap.add_argument("--residual", dest="residual", action="store_true", default=None,
                    help="残差预测:模型学相对最后观测的 ΔT(镜像 B 的残差变体)")
    ap.add_argument("--no-residual", dest="residual", action="store_false",
                    help="关闭残差预测(默认跟随 ENC_CFG)")
    ap.add_argument("--time-features", dest="time_features", action="store_true",
                    default=None,
                    help="输入注入 时刻/年积日 sin-cos 4 列(输入变 72x11,镜像 B;默认跟随 ENC_CFG 的 n_feat=11)")
    ap.add_argument("--no-time-features", dest="time_features", action="store_false",
                    help="关闭时间特征注入")
    ap.add_argument("--lr-schedule", action="store_true",
                    help="ReduceLROnPlateau(验证 MSE 不降 4 轮 → LR 减半,镜像 B)")
    # ---- 降误差四方案(独立开关,可叠加;各自消融) ----
    ap.add_argument("--select-metric", choices=["mse", "mae"], default=None,
                    help="方案1:模型选型/早停依据的验证指标(默认 mse=基线)。"
                         "mae 时同一次训练额外保存按 MAE 选出的 best_mae.pt")
    ap.add_argument("--loss", choices=["mse", "huber"], default=None,
                    help="方案2:训练损失(验证选型仍按 select-metric)。"
                         "huber 需配 --huber-delta,默认 0.23(≈2°C)")
    ap.add_argument("--huber-delta", type=float, default=0.23,
                    help="Huber 损失的 delta(标准化空间;0.23≈物理 2°C)")
    ap.add_argument("--ema", action="store_true",
                    help="方案3:训练全程维护 EMA 权重;训练结束用验证集比较"
                         "原始权重 vs EMA,导出验证集上更优的一份(只多一次前向)")
    ap.add_argument("--per-step-clock", action="store_true",
                    help="方案4:时间特征改逐时间步真实钟点(每条历史观测带自己的"
                         "小时/年积日 sin-cos),不再是预测起点编码复制 72 次;输入仍 11 维")
    variants = ap.add_mutually_exclusive_group()
    variants.add_argument('--dual-residual', action='store_true', help='逐时距学习昨日同刻的残差权重')
    variants.add_argument('--target-hour', action='store_true', help='预测头显式加入各目标小时的日周期')
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print(f"device = {device}")

    cfg = dict(ENC_CFG)                          # 结构超参:命令行 > config 默认
    for key, val in (("d_model", args.d_model), ("nhead", args.nhead),
                     ("n_layer", args.n_layer), ("d_ff", args.d_ff),
                     ("dropout", args.dropout), ("head", args.head)):
        if val is not None:
            cfg[key] = val
    # residual 用三态开关:默认跟随 ENC_CFG;--residual 显式开 / --no-residual 显式关。
    # (不能用 store_true:False 会把 ENC_CFG 里的 residual=True 无条件覆盖掉)
    if args.residual is not None:
        cfg["residual"] = args.residual
    cfg["target_idx"] = TARGET_IDX               # 契约常量,始终随 config
    use_tf = cfg['n_feat'] == 11 if args.time_features is None else args.time_features
    cfg['n_feat'] = len(FEATURE_COLUMNS) + 4 * use_tf
    # 方案4:逐时间步真实钟点编码 —— 模型侧换时间特征模式,数据侧注入方式随之切换
    cfg['per_step_clock'] = bool(args.per_step_clock)
    if args.dual_residual:
        cfg['dual_residual'] = True
    if args.target_hour:
        cfg['target_hour'] = True
    # 方案1/2/3 是训练与选型机制,记入 history 但不影响模型结构
    train_opts = {"select_metric": args.select_metric or "mse",
                  "loss": args.loss or "mse",
                  "huber_delta": args.huber_delta,
                  "ema": args.ema}
    if args.loss == "huber" and args.select_metric is None:
        print("提示:Huber 训练下选型指标未显式指定,沿用 val MSE(方案2 单独消融)")
    diff = {k: cfg[k] for k in cfg if ENC_CFG.get(k) != cfg[k]}
    try:
        plan = resolve_run("transformer", ENC_CKPT, final=args.final,
                           smoke=args.smoke, run_name=args.run_name)
    except ValueError as exc:
        ap.error(str(exc))
    if not plan.formal:
        print(f"非正式运行:只推 val,产物写入 {plan.ckpt_path.parent}")
    print(f"结构配置:{cfg}" + (f"(改动:{diff})" if diff else "(与默认一致)"))

    train_loader, val_loader, test_loader, scaler = get_loaders()
    # 时间特征开关:命令行 --time-features 显式给出,或 ENC_CFG 的 n_feat=11
    # (7 气象 + 4 时间)时自动启用 —— 保证模型输入维度与数据维度一致。
    if use_tf:
        # 注入"时刻 of day + 年积日"sin/cos 4 列(镜像 B 的 USE_TIME_FEATURES)。
        # 拼在 7 个气象特征后面,输入变 [B,72,11];per_step 模式每步用真实钟点。
        if plan.formal:
            train_loader, val_loader, test_loader = _add_time_features(
                train_loader, val_loader, test_loader, per_step=args.per_step_clock)
        else:
            train_loader, val_loader = _add_time_features(
                train_loader, val_loader, per_step=args.per_step_clock)
        mode = "逐时间步真实钟点" if args.per_step_clock else "预测起点复制"
        print(f"时间特征:已注入 tod_sin/tod_cos/doy_sin/doy_cos → 输入 11 维({mode})")
    model = TemporalEncoderTransformer(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"encoder-only 参数量:{n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4) if args.lr_schedule else None)
    # 方案3:EMA 权重影子(指数移动平均)。decay=0.999 ≈ 约 1000 步半衰期,
    # batch 128 下一个 epoch ≈ 382 步,约 2.6 个 epoch 平滑窗口。
    ema_model = (torch.optim.swa_utils.AveragedModel(
        model, avg_fn=lambda avg, new, n: 0.999 * avg + 0.001 * new)
        if args.ema else None)
    plan.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def _loss_fn(pred, target):
        if train_opts["loss"] == "huber":
            return nn.functional.huber_loss(pred, target,
                                            delta=train_opts["huber_delta"])
        return nn.functional.mse_loss(pred, target)

    def _mae_std(model_ref, loader, device_ref):
        """标准化空间验证 MAE(方案1 的选型指标;反标准化只改变缩放,排序不变)。"""
        model_ref.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                err = (model_ref(x.to(device_ref)) - y.to(device_ref)).abs()
                tot += err.sum().item()
                n += err.numel()
        return tot / max(n, 1)

    best, best_mae, patience, history = float("inf"), float("inf"), 0, []
    best_ep, best_mae_ep = 0, 0
    t0 = time.time()
    n_epochs = 1 if args.smoke else args.epochs
    for ep in range(1, n_epochs + 1):
        tr = run_epoch(model, train_loader, device, opt,
                       max_batches=30 if args.smoke else None)
        va = run_epoch(model, val_loader, device,
                       max_batches=10 if args.smoke else None)
        history.append({"epoch": ep, "train_mse": tr, "val_mse": va})
        mark = ""
        # 方案2:训练用(可能为)Huber;验证历史仍记 MSE 保持曲线可比
        if va < best - 1e-5:                     # 验证集选型(契约:模型选择只能用 val)
            best, patience, mark, best_ep = va, 0, "  ← best,已保存", ep
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "val_mse_std": best, "epoch": ep,
                        "time_features": use_tf,
                        "residual": cfg.get("residual", False),
                        **train_opts}, plan.ckpt_path)
        else:
            patience += 1
        # 方案1:同一次训练并行按验证 MAE 选轮次(不改动早停规则,零额外训练成本)
        if train_opts["select_metric"] == "mae" or args.ema:
            va_mae = _mae_std(model, val_loader, device)
            history[-1]["val_mae"] = va_mae
            if va_mae < best_mae - 1e-5:
                best_mae, best_mae_ep = va_mae, ep
                if train_opts["select_metric"] == "mae":
                    torch.save({"model": model.state_dict(), "cfg": cfg,
                                "val_mae_std": va_mae, "epoch": ep,
                                "time_features": use_tf,
                                "residual": cfg.get("residual", False),
                                **train_opts}, plan.ckpt_path)
                    mark += "  ← best(MAE)"
        if ema_model is not None:
            ema_model.update_parameters(model)
            history[-1]["ema"] = True
        if sched is not None:
            sched.step(va)
        print(f"epoch {ep:02d} | train MSE {tr:.4f} | val MSE {va:.4f}{mark}")
        if not args.smoke and patience >= args.patience:
            print(f"早停:连续 {args.patience} 轮验证 MSE 无改善")
            break

    if best == float("inf"):     # 本轮没写出 checkpoint:不导出,也不覆盖旧正式产物
        raise RuntimeError("本轮未保存 checkpoint,未导出预测或覆盖训练历史")

    # 方案3:训练结束用验证集比较原始 vs EMA,导出验证集上更优的一份
    # (只各做一次 val 前向,不是多模型融合;选哪个权重仍是验证集说了算)
    if ema_model is not None:
        ema_mse = run_epoch(ema_model.module, val_loader, device)
        print(f"[EMA] 原始 best val MSE {best:.4f}(epoch {best_ep}) vs "
              f"EMA val MSE {ema_mse:.4f} → 选"
              f"{'EMA' if ema_mse < best else '原始'}")
        if ema_mse < best:
            torch.save({"model": ema_model.module.state_dict(), "cfg": cfg,
                        "val_mse_std": ema_mse, "epoch": f"ema@{ep}",
                        "time_features": use_tf,
                        "residual": cfg.get("residual", False),
                        **train_opts}, plan.ckpt_path)

    # 正式运行:val+test 都推(测试集仅此一次);冒烟/调参:只推 val
    result = export("enc", data=(train_loader, val_loader, test_loader, scaler),
                    ckpt_path=plan.ckpt_path, pred_tag=plan.pred_tag,
                    splits=plan.splits, preds_dir=plan.preds_dir)
    out = {"model": "encoder-only Transformer", "cfg": cfg, "params": n_params,
           "run_name": args.run_name, "smoke": args.smoke, "final": plan.formal,
           "seed": args.seed, "patience": args.patience, "max_epochs": n_epochs,
           "lr_schedule": args.lr_schedule, **train_opts, "best_epoch_mse": best_ep,
           "best_epoch_mae": best_mae_ep, "best_val_mae_std": best_mae,
           "epochs_ran": len(history), "best_val_mse_std": best,
           "train_seconds": round(time.time() - t0, 1),
           "history": history, "metrics_celsius": result}
    with open(plan.history_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"训练曲线与结果已存 {plan.history_path}")


if __name__ == "__main__":
    main()
