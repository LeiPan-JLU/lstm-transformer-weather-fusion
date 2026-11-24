# -*- coding: utf-8 -*-
"""
LSTM 气温预测模型（数据管线）。

结构
----
    输入 [B, 72, F]
      -> 2 层 LSTM（hidden=64, dropout=0.1）
      -> 取最后一个时间步的隐状态 h_T  [B, 64]
      -> 全连接 [64 -> 24]
      -> 输出 [B, 24, 1]

为什么是"直接多输出（Direct / MIMO）"而不是递归
------------------------------------------------
递归策略要预测 24 次、每次把预测值喂回输入，误差会累积 24 次；
直接多输出只做一次前向，24 个预测互相独立，不存在误差累积。
两种策略的系统比较见 Taieb et al., "A review and comparison of strategies for
multi-step ahead time series forecasting"。

负责人：数据管线
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from lstm_common import config as C  # noqa: E402


class LSTMRegressor(nn.Module):
    """
    LSTM 多步回归器。

    参数
    ----
    input_size : 输入特征数（主实验 7；开启时间特征后为 11）
    hidden_size, num_layers, dropout : 见 config.py
    pred_len : 预测步数（24）
    forget_bias_init : 遗忘门偏置初值。LSTM 中遗忘门 b_f 的默认初始化是 0，
        初始时 sigmoid(0)=0.5，一半记忆被丢掉；把它初始化为 1.0 能让网络开局
        更倾向"记住"，对长序列任务常有帮助（Jozefowicz et al., 2015）。
    """

    def __init__(
        self,
        input_size: int = C.N_FEATURES,
        hidden_size: int = C.LSTM_HIDDEN,
        num_layers: int = C.LSTM_LAYERS,
        dropout: float = C.LSTM_DROPOUT,
        pred_len: int = C.PRED_LEN,
        forget_bias_init: float = C.FORGET_BIAS_INIT,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.pred_len = int(pred_len)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(self.hidden_size, self.pred_len)

        if forget_bias_init:
            self._init_forget_bias(forget_bias_init)

    def _init_forget_bias(self, value: float) -> None:
        """把每一层遗忘门的偏置设为 value。PyTorch 里 weight_ih 在前、weight_hh 在后。"""
        for layer in range(self.num_layers):
            b = getattr(self.lstm, f"bias_ih_l{layer}")
            h = self.hidden_size
            with torch.no_grad():
                b[h:2 * h].fill_(value)   # 遗忘门对应的那一段

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, F] -> [B, pred_len, 1]"""
        out, _ = self.lstm(x)                 # [B, T, H]
        h_last = out[:, -1, :]                # [B, H]
        y = self.head(h_last)                 # [B, pred_len]
        return y.unsqueeze(-1)                # [B, pred_len, 1]

    # 便于报告里写"参数量"
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(input_size: int, **kwargs) -> LSTMRegressor:
    """工厂函数，保证 train.py 与 export 脚本用同一份超参构造。"""
    return LSTMRegressor(input_size=input_size, **kwargs)


if __name__ == "__main__":
    m = build_model(C.N_FEATURES)
    x = torch.randn(4, C.SEQ_LEN, C.N_FEATURES)
    y = m(x)
    print("output shape:", tuple(y.shape))
    print("params:", m.n_params())
