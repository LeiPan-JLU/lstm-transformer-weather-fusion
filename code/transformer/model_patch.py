# -*- coding: utf-8 -*-
"""
model_patch.py —— 方案 B 附加实验:PatchTST(Nie et al., 2023, ICLR)。
与 encoder-only 的三点差异(lessons/0010 精读):
  1. patch 化:每条通道切成 patch_len=12 / stride=12 的 6 个 patch(直切,精确铺满
     72 步;论文还补 S 个边缘值多出 1 个 patch,这里为公平对比用直切);
     patch 与预测起点右对齐:铺不满时丢弃**最早**的余数小时,保证窗口末尾
     (离预测起点最近的观测)一定进 patch;
  2. 通道独立:7 条通道 reshape 进批维(B*7),共享同一套 Transformer 权重,
     输出头对 7 个通道的结果取平均;
  3. 可学习位置编码 + patch 级 BatchNorm(Zerveas 2021)+ GELU + dropout 0.2。
输入 [B,72,7](标准化空间)→ 输出 [B,24,1](标准化空间的 T)。
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from common.config import SEQ_IN, SEQ_OUT


class PatchTST(nn.Module):
    """通道独立的 patch 化 Transformer(方案 B 附加实验)。"""

    def __init__(self, n_feat=7, d_model=64, nhead=4, n_layer=2, d_ff=128,
                 dropout=0.2, patch_len=12, stride=12,
                 seq_in=SEQ_IN, seq_out=SEQ_OUT):
        super().__init__()
        assert stride <= patch_len, "允许重叠 patch,但步长不能超过 patch 长度"
        self.patch_len, self.stride = patch_len, stride
        # 右端对齐预测起点:N 个 patch 尽量铺满序列末端,余数(excess)从头部丢掉。
        # 直切 (L−P) mod S = 0 时 excess=0,行为与原来一致;
        # 12/8 时 (72−12)=60 不是 8 的倍数,丢弃最早 4 小时、保留最后 4 小时。
        excess = (seq_in - patch_len) % stride
        self.offset = excess
        self.n_patches = (seq_in - excess - patch_len) // stride + 1

        self.embed = nn.Linear(patch_len, d_model)                       # 每个 patch → token
        self.bn = nn.BatchNorm1d(d_model)                                # patch 级 BN(Zerveas 2021)
        self.pos = nn.Parameter(                                         # 可学习位置编码
            torch.randn(1, self.n_patches, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layer,
                                         norm=nn.LayerNorm(d_model),
                                         enable_nested_tensor=False)

        # 每条通道各自:拼平 N 个 token → MLP → 24 个输出(7 条通道共享这套头)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_patches * d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, seq_out),
        )

    def forward(self, x):                            # x: [B, 72, 7](标准化空间)
        B, L, M = x.shape
        z = x[:, self.offset:, :].transpose(1, 2)    # [B, 7, L-offset] —— 每条通道单独成序列
        z = z.unfold(2, self.patch_len, self.stride) # [B, 7, N, patch_len]
        z = z.reshape(B * M, self.n_patches, self.patch_len)   # 7 通道共享权重
        h = self.embed(z)                            # [B*7, N, d_model]
        h = self.bn(h.transpose(1, 2)).transpose(1, 2)          # BatchNorm over 特征维
        h = self.enc(h + self.pos)                   # [B*7, N, d_model]
        out = self.head(h)                           # [B*7, 24]
        out = out.reshape(B, M, -1).mean(dim=1)      # 7 条通道预测取平均(通道独立的汇总)
        return out.unsqueeze(-1)                     # [B, 24, 1]


if __name__ == "__main__":
    # 冒烟:随机输入走一遍前向,核对输出形状与参数量
    model = PatchTST()
    y = model(torch.randn(4, SEQ_IN, 7))
    n = sum(p.numel() for p in model.parameters())
    print(f"out: {tuple(y.shape)} | n_patches: {model.n_patches} | params: {n:,}")
