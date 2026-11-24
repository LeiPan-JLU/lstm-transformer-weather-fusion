# -*- coding: utf-8 -*-
"""run_artifacts.py —— 一次训练运行的产物落点(正式 / 冒烟·调参)。

只有显式 --final 才发布正式产物并推 test;其余情况(含 --smoke、--run-name、
自定义超参)一律只推 val,每次运行在 outputs/v3/experiments/ 下新建一个独占
子目录(mkdtemp),重复调用/同名 run-name 不会互相覆盖。旧口径结果留在
outputs/{checkpoints,preds},不在这里。
"""
import re
import tempfile
from pathlib import Path

from common.config import CKPT_DIR, EXPERIMENTS_DIR, PREDS_DIR


class RunPlan:
    def __init__(self, *, formal, ckpt_path, history_path, preds_dir, pred_tag, splits):
        self.formal = formal
        self.ckpt_path = Path(ckpt_path)
        self.history_path = Path(history_path)
        self.preds_dir = Path(preds_dir)
        self.pred_tag = pred_tag
        self.splits = splits


def resolve_run(tag, formal_ckpt, *, final=False, smoke=False, run_name=None):
    """决定产物落点。tag = transformer / patchtst;formal_ckpt = 正式 checkpoint 文件名。

    --final 与 --smoke / --run-name 互斥(歧义直接报错);结构超参 / epochs /
    patience 可随 --final 一起给。非正式运行每次独占一个新目录。
    """
    if final and (smoke or run_name):
        raise ValueError("--final 只能用于正式训练,不能与 --smoke / --run-name 同时使用")
    if final:
        return RunPlan(formal=True, ckpt_path=CKPT_DIR / formal_ckpt,
                       history_path=CKPT_DIR / f"{tag}_history.json",
                       preds_dir=PREDS_DIR, pred_tag=None, splits=("val", "test"))

    name = re.sub(r'[\\/:*?"<>|]', "_", run_name) if run_name else ""
    stem = f"{tag}_{'smoke' if smoke else 'trial'}" + (f"_{name}" if name else "")
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(dir=EXPERIMENTS_DIR, prefix=f"{stem}_"))
    return RunPlan(formal=False, ckpt_path=out_dir / "best.pt",
                   history_path=out_dir / "history.json",
                   preds_dir=out_dir / "preds", pred_tag=stem, splits=("val",))
