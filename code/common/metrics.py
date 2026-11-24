# -*- coding: utf-8 -*-
"""
metrics.py —— 全项目唯一指标实现(README §九契约)。
输入摄氏度的 pred 与真值(形状 [N,24] 或 [N,24,1] 都接受,内部拉平),
输出 {"MAE":…, "RMSE":…}。LSTM 侧、Transformer 侧、fusion 侧都调用它,
禁止各自另写指标(README "evaluate():全项目唯一实现")。
"""
import numpy as np


def evaluate(pred_c, true_c):
    """pred_c / true_c:摄氏度预测与真值。返回 MAE、RMSE(单位 °C)。"""
    pred = np.asarray(pred_c, dtype=np.float64).reshape(-1)
    true = np.asarray(true_c, dtype=np.float64).reshape(-1)
    if pred.shape != true.shape:
        raise ValueError(f"pred/true 形状不一致:{pred.shape} vs {true.shape}")
    err = pred - true
    return {
        "MAE":  float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
    }


if __name__ == "__main__":
    # 自检:常数 2°C 偏差 → MAE = RMSE = 2
    a = np.zeros((3, 24))
    print(evaluate(a, a + 2.0))          # {'MAE': 2.0, 'RMSE': 2.0}
