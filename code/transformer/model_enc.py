# -*- coding: utf-8 -*-
"""
model_enc.py —— 方案 A 主线:encoder-only Transformer(README §4.2 配置)。

    Linear(7→64) 输入投影 → 正弦位置编码 → 2 层 pre-norm Encoder
    → Flatten 预测头(4608→256→24)

要点(lessons/0009):
  * 无因果掩码 —— 输入只有过去 72h,窗口内自注意力不涉及未来;
  * "解码"职责由预测头承担:回归任务把生成过程坍缩成一个映射;
  * 备用 head="pool":时间均值池化的精简头(约 2 万参数);
  * residual=True:预测相对"最后观测 T"的增量 ΔT(镜像 B 的残差变体),
    气温强持续性下水平位不再进入学习目标,只学变化 —— B 实测 MAE 1.80→1.73 的
    主要来源。对外输出仍是标准化 T,下游(损失/反标准化/导出)无需感知。
输入 [B,72,7](标准化空间)→ 输出 [B,24,1](标准化空间的 T)。
"""
import sys
from pathlib import Path

import math
import torch
import torch.nn as nn

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from common.config import SEQ_IN, SEQ_OUT, TARGET_IDX


class SinusoidalPE(nn.Module):
    """正弦位置编码:偶数下标 sin(pos/ω),奇数下标 cos;注册为 buffer,不参与训练。
    波长从 2π 到 10000·2π 几何递增,不同周期成分可被不同维度分辨(lessons/0004)。"""

    def __init__(self, seq_len, d_model):
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)       # [L, 1]
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))                   # [d/2]
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)      # d_model 需为偶数(本项目 64)
        self.register_buffer("pe", pe.unsqueeze(0))                         # [1, L, d]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TemporalEncoderTransformer(nn.Module):
    """encoder-only 时序预测模型(方案 A 主线)。"""

    def __init__(self, n_feat=7, d_model=64, nhead=4, n_layer=2, d_ff=128,
                 dropout=0.1, head="flatten", seq_in=SEQ_IN, seq_out=SEQ_OUT,
                 residual=False, target_idx=TARGET_IDX, dual_residual=False, target_hour=False,
                 per_step_clock=False):
        super().__init__()
        self.inp = nn.Linear(n_feat, d_model)       # 7 维气象特征 → d_model(逐时间步共享)
        self.pos = SinusoidalPE(seq_in, d_model)
        self.residual = residual
        self.target_idx = target_idx
        # 方案4标记:时间特征列是"每步真实钟点"(True)还是"预测起点复制"(False)。
        # 只影响数据注入与 checkpoint 元数据,前向计算两种模式完全一致 ——
        # per_step 模式下 x 的最后一步恰好就是预测起点,target_hour 分支语义不变。
        self.per_step_clock = per_step_clock

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        # pre-norm 结构最后补一次 LayerNorm,保证输出尺度稳定
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layer,
                                         norm=nn.LayerNorm(d_model),
                                         enable_nested_tensor=False)

        if head == "flatten":       # 主线头:全部时间步拼平 → MLP → 24 个输出
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_in * d_model, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, seq_out),
            )
        elif head == "pool":        # 精简头:时间均值池化(消融/参数量对照用)
            self.head = nn.Sequential(
                nn.Linear(d_model, 256), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(256, seq_out),
            )
        else:
            raise ValueError(f"未知 head 类型:{head}")
        self.head_kind = head
        self.day_logits = None
        if dual_residual:
            if not residual or seq_in < 24 or seq_out > 24:
                raise ValueError('dual_residual requires residual mode, 24h history and horizon <= 24')
            self.day_logits = nn.Parameter(torch.zeros(seq_out))
        self.target_hour = target_hour
        if target_hour:
            if n_feat != 11 or head != 'pool':
                raise ValueError('target_hour requires 11 features and the pool head')
            phase = torch.arange(seq_out) * (math.tau / 24)
            self.register_buffer('offset_sin', phase.sin(), persistent=False)
            self.register_buffer('offset_cos', phase.cos(), persistent=False)
            self.hour_coef = nn.Parameter(torch.zeros(d_model, 2))
            self.hour_bias = nn.Parameter(torch.zeros(2))

    def forward(self, x):                            # x: [B, 72, 7](标准化空间)
        h = self.enc(self.pos(self.inp(x)))          # [B, 72, d_model]
        if self.head_kind == "pool":
            h = h.mean(dim=1)                        # [B, d_model]
        out = self.head(h).unsqueeze(-1)             # [B, 24, 1]
        if self.target_hour:
            # 输入时间是首个预测小时，而不是最后观测小时；h=1 的相位偏移为零。
            sin0, cos0 = x[:, -1, 7:8], x[:, -1, 8:9]
            sin_h = sin0 * self.offset_cos + cos0 * self.offset_sin
            cos_h = cos0 * self.offset_cos - sin0 * self.offset_sin
            a, b = (h @ self.hour_coef + self.hour_bias).split(1, dim=-1)
            out = out + (a * sin_h + b * cos_h).unsqueeze(-1)
        if self.residual:
            # 残差模式:head 输出的是相对"最后观测 T"的增量 ΔT(标准化空间),
            # 加回窗口最后一步的目标列即得绝对 T。对外仍是标准化 T,契约不变;
            # MSE(绝对T) 与 MSE(ΔT) 只差一个与参数无关的常数项,梯度完全等价。
            last = x[:, -1:, self.target_idx: self.target_idx + 1]
            out = out + last
            if self.day_logits is not None:
                yesterday = x[:, -24:, self.target_idx: self.target_idx + 1][:, :out.size(1)]
                out = out + self.day_logits.sigmoid()[:, None] * (yesterday - last)
        return out


if __name__ == "__main__":
    # 冒烟:随机输入走一遍前向,核对输出形状与参数量
    model = TemporalEncoderTransformer()
    y = model(torch.randn(4, SEQ_IN, 7))
    n = sum(p.numel() for p in model.parameters())
    print(f"out: {tuple(y.shape)} | params: {n:,}")
