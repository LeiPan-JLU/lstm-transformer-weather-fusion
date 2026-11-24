# Method

## Data protocol

The Jena Climate archive is filtered to integer-hour rows (`minute == 0`), giving
70,092 hourly observations. Seven variables are used:

`p (mbar), T (degC), VPmax (mbar), VPdef (mbar), sh (g/kg), rho (g/m**3), wv (m/s)`

The target is `T (degC)`. Wind sentinel `-9999` values are marked missing and
forward-filled from the most recent valid observation. Windows are kept only when
the 96 consecutive rows are separated by exactly one hour; this prevents a window
from crossing a real data gap. The chronological split is 70% / 10% / 20% for
training, validation, and test. A `StandardScaler` is fitted on the training
segment only.

Input tensors have shape `[batch, 72, 7]`; targets have shape `[batch, 24]` and are
stored in °C after inverse transformation.

## Models

**LSTM + time features + residual.** Two LSTM layers (hidden size 64, dropout 0.1)
consume 72 hourly steps. Four cyclical time features (time of day and day of year)
are appended, and the head predicts a residual relative to the last observed
temperature. The variant was selected on validation MAE.

**Transformer encoder.** The encoder-only model uses `d_model=128`, four attention
heads, three layers, a feed-forward width of 128, dropout 0.1, and a pooled
24-step prediction head. It uses the same seven weather variables plus the four
time features. PatchTST is included as an optional baseline but is not the main
model.

## Fusion

Given validation predictions from the two models, the equal-weight, scalar, and
per-horizon fusion rules are

$$
\hat Y_{\mathrm{eq}} = \frac{1}{2}\hat Y_L + \frac{1}{2}\hat Y_T
$$

$$
\hat Y_{\alpha} = \alpha \hat Y_L + (1-\alpha)\hat Y_T,\qquad
\alpha = \sigma(a)
$$

$$
\hat Y_{\alpha_t} = \alpha_t \hat Y_{L,t} + (1-\alpha_t)\hat Y_{T,t}
$$

The scalar and per-horizon weights are fitted by least squares on validation
predictions only. The equal-weight rule is selected by validation MAE and is used
once on the test set. The learned scalar is α = 0.540. A bootstrap robustness check
with 500 resamples is provided in `outputs/v3/fusion/fusion_robust_report.json`.

## Leakage controls

1. The scaler is fitted on the training segment only.
2. Fusion weights and the fusion variant are selected on validation data.
3. The test set is evaluated once, after selection.
4. Prediction files are stored in °C, so fusion and evaluation share one scale.
