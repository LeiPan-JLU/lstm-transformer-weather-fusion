# -*- coding: utf-8 -*-
"""
LSTM 训练脚本（数据管线）。契约产物：outputs/checkpoints/<tag>_best.pt

用法
----
    python code/lstm/train.py                       # 主实验（7 特征，契约配置）
    python code/lstm/train.py --tag lstm_time --time-features
    python code/lstm/train.py --tag lstm_res  --residual
    python code/lstm/train.py --tag lstm_full --time-features --residual --forget-bias 1.0

训练细节
--------
* 损失：标准化空间上的 MSE（数值稳定，与融合权重的 argmin 等价）；
* 早停：验证集**摄氏度 MAE**（对报告更直观，且与最终指标同口径）；
* 梯度裁剪 1.0；ReduceLROnPlateau 在验证损失上降 lr；
* 固定随机种子，训练集打乱、验证集不打乱（保证与 t_start 对齐）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from lstm_common import config as C          # noqa: E402
from lstm_common import data as D            # noqa: E402
from lstm_common import metrics as M         # noqa: E402
from lstm.model import build_model      # noqa: E402


# ---------------------------------------------------------------- 工具
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict_std(model: nn.Module, x: np.ndarray, device: torch.device,
                batch: int = 1024) -> np.ndarray:
    """批量推理，返回标准化空间预测 [N, 24]。"""
    model.eval()
    outs = []
    xt = torch.from_numpy(x)
    for i in range(0, len(xt), batch):
        xb = xt[i:i + batch].to(device, non_blocking=True)
        outs.append(model(xb).float().cpu())
    return torch.cat(outs, dim=0).numpy()[..., 0]


def evaluate_split(model, arr: dict, device: torch.device) -> tuple[dict, np.ndarray]:
    """在某个 split 上算摄氏度指标，返回 (metrics, 摄氏度预测)。"""
    p_std = predict_std(model, arr["x"], device)
    p_c = D.unscale_y(p_std, arr)
    return M.evaluate(arr["y_celsius"], p_c), p_c


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="lstm", help="实验名，决定 checkpoint 与日志文件名")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--hidden", type=int, default=C.LSTM_HIDDEN)
    ap.add_argument("--layers", type=int, default=C.LSTM_LAYERS)
    ap.add_argument("--dropout", type=float, default=C.LSTM_DROPOUT)
    ap.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY)
    ap.add_argument("--time-features", action="store_true", help="加入时刻/年积日周期特征")
    ap.add_argument("--residual", action="store_true", help="预测相对最后观测的增量")
    ap.add_argument("--forget-bias", type=float, default=C.FORGET_BIAS_INIT,
                    help="遗忘门偏置初值，1.0 为常用技巧")
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-cudnn", action="store_true",
                    help="关闭 cuDNN（慢 3~5 倍，但进程退出码干净；见 config.safe_shutdown）")
    args = ap.parse_args()

    if args.no_cudnn:
        torch.backends.cudnn.enabled = False

    set_seed(args.seed)
    device = pick_device(args.device)
    t0 = time.time()

    hd = D.build_hourly()
    bounds = D.split_bounds(len(hd.compact))
    scaler = D.fit_scaler(hd.compact, bounds["train"][1])
    arrs = {s: D.get_arrays(s, hd=hd, scaler=scaler,
                            use_time_features=args.time_features,
                            residual=args.residual)
            for s in ("train", "val", "test")}

    n_in = arrs["train"]["x"].shape[-1]
    model = build_model(
        input_size=n_in,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        pred_len=C.PRED_LEN,
        forget_bias_init=args.forget_bias,
    ).to(device)

    print(f"[{args.tag}] device={device} input={n_in} params={model.n_params():,}")
    print(f"  样本数 train={len(arrs['train']['x'])} val={len(arrs['val']['x'])} "
          f"test={len(arrs['test']['x'])}")

    train_ds = TensorDataset(torch.from_numpy(arrs["train"]["x"]),
                             torch.from_numpy(arrs["train"]["y"]))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, pin_memory=(device.type == "cuda"))

    x_val = torch.from_numpy(arrs["val"]["x"]).to(device)
    y_val = torch.from_numpy(arrs["val"]["y"]).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=C.LR_FACTOR, patience=C.LR_PATIENCE)
    loss_fn = nn.MSELoss()

    best_mae = float("inf")
    best_epoch = -1
    bad = 0
    history: list[dict] = []
    ckpt_path = C.CKPT_DIR / f"{args.tag}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            tot_loss += loss.item() * len(xb)
            n_seen += len(xb)
        train_loss = tot_loss / n_seen

        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).item())
        val_metrics, _ = evaluate_split(model, arrs["val"], device)
        sched.step(val_loss)

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_MAE": val_metrics["MAE"],
            "val_RMSE": val_metrics["RMSE"],
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(rec)
        improved = val_metrics["MAE"] < best_mae - 1e-6
        if improved:
            best_mae = val_metrics["MAE"]
            best_epoch = epoch
            bad = 0
            torch.save({
                "state_dict": model.state_dict(),
                "input_size": n_in,
                "hparams": vars(args),
                "scaler": scaler,
                "residual": bool(args.residual),
                "use_time_features": bool(args.time_features),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "config": {
                    "seq_len": C.SEQ_LEN, "pred_len": C.PRED_LEN,
                    "features": C.FEATURES, "target_idx": C.TARGET_IDX,
                    "split_ratio": C.SPLIT_RATIO,
                },
            }, ckpt_path)
        else:
            bad += 1

        if epoch == 1 or improved or epoch % 5 == 0:
            print(f"  ep{epoch:03d} train {train_loss:.4f} | val {val_loss:.4f} | "
                  f"MAE {val_metrics['MAE']:.4f} RMSE {val_metrics['RMSE']:.4f}"
                  f"{'  *' if improved else ''}")
        if bad >= args.patience:
            print(f"  早停：验证集 MAE 已 {args.patience} 个 epoch 未提升")
            break

    elapsed = time.time() - t0

    # 用最佳 checkpoint 重新评估
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    test_metrics, _ = evaluate_split(model, arrs["test"], device)
    val_metrics, _ = evaluate_split(model, arrs["val"], device)

    summary = {
        "tag": args.tag,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "elapsed_sec": round(elapsed, 1),
        "n_params": model.n_params(),
        "val": val_metrics,
        "test": test_metrics,
        "args": vars(args),
        "env": C.env_info(),
        "history": history,
    }
    out = C.LOG_DIR / f"{args.tag}_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {args.tag} ===")
    print(f"最佳 epoch {best_epoch}/{len(history)}，用时 {elapsed:.0f}s，参数 {model.n_params():,}")
    print(f"val  MAE {val_metrics['MAE']:.4f}  RMSE {val_metrics['RMSE']:.4f}")
    print(f"test MAE {test_metrics['MAE']:.4f}  RMSE {test_metrics['RMSE']:.4f}")
    print(f"checkpoint -> {ckpt_path}")
    print(f"summary    -> {out}")

    # 学习曲线
    try:
        C.set_plot_style()
        import matplotlib.pyplot as plt
        ep = [h["epoch"] for h in history]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(ep, [h["train_loss"] for h in history], label="训练损失")
        ax[0].plot(ep, [h["val_loss"] for h in history], label="验证损失")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE（标准化空间）"); ax[0].legend()
        ax[0].set_title("损失曲线")
        ax[1].plot(ep, [h["val_MAE"] for h in history], color="tab:red", label="验证 MAE")
        ax[1].axhline(y=best_mae, ls="--", lw=0.8, color="gray")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("MAE（℃）"); ax[1].legend()
        ax[1].set_title("验证集 MAE")
        fig.suptitle(f"LSTM 训练过程（{args.tag}）")
        fig.savefig(C.FIG_DIR / f"fig_learning_curve_{args.tag}.png")
        plt.close(fig)
        print(f"curve      -> {C.FIG_DIR / f'fig_learning_curve_{args.tag}.png'}")
    except Exception as e:  # 画图失败不影响训练结果
        print(f"（学习曲线绘制失败：{e}）")


if __name__ == "__main__":
    main()
    C.safe_shutdown()