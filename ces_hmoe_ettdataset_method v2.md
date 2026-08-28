# CES-HMoE for ETT Forecasting

当前实现：[ces_hmoe_ettdataset.py](E:\power2\ces_hmoe_ettdataset.py)

本文档以 `E:\power2` 当前代码和截至 2026-08-25 已完成的实验为准。旧日志复现、冻结专家后处理校准和 fresh train 必须分别标记，不能混为同一种实验。

## 1. 方法概述

CES-HMoE 保留趋势、周期和突变三个互补专家，并使用逐预测步的受限动态门控：

```text
TrendExpert-v2
PeriodicExpert-v2
RampExpert-v1
        ↓
历史/未来日历表示
        ↓
bounded horizon-wise gate
        ↓
三专家加权预测
```

对预测起点 `t` 和预测步 `h`：

$$
\hat y_{t,h}=\sum_{e\in\{trend,periodic,ramp\}}
g_{t,h}^{e}\hat y_{t,h}^{e},
\qquad \sum_e g_{t,h}^{e}=1.
$$

当前最优模型不是无约束 softmax MoE，而是围绕专家先验的弱动态路由：

$$
\mathbf g_{t,h}
=(1-\lambda)\mathbf p+
\lambda\widetilde{\mathbf g}_{t,h},
$$

其中：

```text
p = [0.853, 0.142, 0.004]
lambda = dynamic_blend = 0.20
```

因此模型仍然保留动态 horizon-wise routing，但限制动态范围，避免 RampExpert 或单一专家在测试集上塌缩。

ETT 的目标列是 `OT`，表示变压器油温。论文中应将 ETT 描述为电力系统时间序列预测基准，不应把 ETT 结果直接称为真实负荷泛化。

## 2. 数据和因果切分

标准 CSV 结构：

```text
date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
```

输入包含 7 个数值通道，代码将 `OT` 放在最后，使专家可以使用 `x[..., -1]`。

官方时间切分：

| 数据集 | 采样间隔 | 训练 | 验证 | 测试 |
|---|---:|---:|---:|---:|
| ETTh1/ETTh2 | 1 小时 | 8640 | 2880 | 2880 |
| ETTm1/ETTm2 | 15 分钟 | 34560 | 11520 | 11520 |

所有切分按时间顺序进行，不能随机划分。

对预测起点 `t`：

```text
x               = values[t-L:t]
target          = OT[t:t+H]
future_calendar = calendar[t:t+H]
entropy_state   = entropy(values[t-L:t])
```

scaler 只在训练段拟合；输入窗口、熵状态和未来日历均不读取预测区间内的实测值。

当前批量实验覆盖：

```text
seq_len  = [96, 168, 336, 720]
pred_len = [24, 96, 192, 336, 720]
```

对于 ETTm，`pred_len=96` 表示 15 分钟采样下未来 24 小时。

## 3. 可选因果多尺度熵状态

代码实现了以下因果状态：

```text
Permutation Entropy (PE)
Spectral Entropy (SpE)
Sample Entropy (SampEn)
Approximate Entropy (ApEn)
mean / std / median
slope / ramp ratio
```

ETTh 使用：

```python
scales = (24, 48, 168)
```

ETTm 使用：

```python
scales = (96, 192, 672)
```

每个尺度的状态维度为：

```text
5 * 7 + 4 = 39
```

三个尺度共 `117` 维。

熵 cache 刷新策略：

| 数据集 | PE/SpE/统计量 | SampEn/ApEn |
|---|---:|---:|
| ETTh | 每 1 步 | 每 24 步 |
| ETTm | 每 4 步 | 每 96 步 |

昂贵熵特征非锚点使用最近一次值前向保持，不使用未来填充。

### 熵状态实验结论

熵状态是可选机制，不是当前最佳配置的必要输入：

```text
当前最佳配置：entropy_state = disabled
```

已完成消融表明：

```text
全量熵状态强注入不稳定；
selective entropy adapter 没有稳定收益；
软专家路由监督没有稳定收益；
ETTm2 长预测中弱熵适配器与无熵配置基本一致。
```

因此论文中应将熵状态描述为可选的因果状态机制，并明确当前最佳稳定模型关闭了主路由熵输入。

## 4. 当前三个专家

### 4.1 TrendExpert-v2

TrendExpert-v2 只读取目标 `OT`：

```text
OT
→ moving average
→ Linear(L,H)
```

并加入平滑趋势残差：

```python
forecast = linear(trend)
forecast += 0.1 * residual_linear(target - trend)
```

它同时建模长期趋势和短期偏离趋势的残差。与 TrendExpert-v1 相比，这是当前收益最大的专家改动。

### 4.2 PeriodicExpert-v2

PeriodicExpert-v2 使用轻量 CNN：

```text
OT
→ Conv1d(kernel=25)
→ depthwise Conv1d(kernel=9)
→ AdaptiveAvgPool1d(8)
```

并显式加入：

```text
lag-24
lag-168
lag-336
```

季节锚点经过 Linear 映射后与 CNN 特征融合。

PeriodicExpert-v3 在 v2 上增加一层 depthwise convolution，但没有稳定优于 v2，因此不作为默认版本。

### 4.3 RampExpert-v1

RampExpert-v1 读取全部 7 个变量的一阶差分：

$$
\Delta x_i=x_i-x_{i-1}.
$$

使用 dilation 为 `1/2/4` 的 TCN，输出未来 `H` 步预测。

RampExpert-v2/v3 已完成消融，但没有稳定收益。因此 RampExpert 保留在三专家结构中，但先验质量限制在约 `0.004`。

## 5. Bounded horizon-wise gate

历史编码器：

```text
HistoryEncoder([B,L,7]) -> [B,64]
```

未来日历包含：

```text
hour sin/cos
day-of-week sin/cos
month sin/cos
```

未来日历经过 Linear 编码后与可学习 horizon embedding 相加。

当前最佳路由输入：

```text
历史表示
未来日历表示
horizon embedding
```

熵状态不进入主路由器。

当前配置：

```text
fusion_mode       = bounded_gate
gate_mode         = horizon
prior_weights     = 0.853,0.142,0.004
dynamic_blend     = 0.20
gate_init         = zero
entropy_state     = disabled
```

bounded gate 主要动态调整 Trend/Periodic 比例，Ramp 保持小权重。它不是静态融合，也不是三个专家完全自由竞争的无约束 MoE。

## 6. 分阶段训练和损失

当前正式配置：

```text
总 epochs = 100
Stage 1 = 60
Stage 2 = 30
Stage 3 = 10
```

Stage 1：独立预训练三个专家。

```text
stage1_fusion = independent
stage1_lr = 1e-4
```

Stage 2：冻结专家，只训练 gate。

```text
stage2_scope = gate_only
stage2_lr = 1e-4
```

Stage 3：gate-only 弱微调。

```text
stage3_scope = gate_only
finetune_lr = 1e-5
```

基础预测损失：

$$
L=L_{Huber}+0.2L_{slope}+0.3L_{ramp}.
$$

当前最佳模型使用：

```text
balance_weight = 0
horizon_balance_weight = 0
wtr_loss_weight = 0
route_loss_weight = 0
```

验证集 checkpoint 选择：

```text
selection_metric = weighted_wtr
selection_mse_tolerance = 0.01
```

代码支持 `--master_seed`，统一控制：

```text
seed = init_seed = gate_seed = loader_seed
```

旧的独立 seed 参数仍保留，仅用于历史实验复现。

## 7. 已完成消融实验

### 7.1 路由和熵消融

已测试：

```text
global gate
horizon gate
entropy enabled/disabled
selective entropy adapter
route supervision
dynamic_blend=0.5
不同趋势/周期先验
```

结论：

```text
bounded horizon-wise gate 最稳定；
全量熵输入没有稳定收益；
route supervision 没有稳定收益；
dynamic_blend=0.5 会放大验证-测试偏移；
concat 不稳定；
无约束 Ramp 路由会破坏长预测。
```

### 7.2 WTR loss 消融

测试了不同：

```text
wtr_loss_weight
wtr_temperature
uniform/early/late horizon weighting
```

H=24 和 H=192 的多个配置结果几乎一致，未形成稳定额外收益。因此 WTR 当前用于验证选择和报告，不作为主训练损失。

### 7.3 专家结构消融

测试了：

```text
Periodic-v2
Periodic-v3
Trend-v2
Ramp-v2
Ramp-v3
Periodic-v2 + Trend-v2
```

结论：

```text
Trend-v2：稳定有效；
Periodic-v2：有辅助收益；
Periodic-v3：没有稳定优于 v2；
Ramp-v2/v3：没有稳定收益；
Trend-v2 + Periodic-v2：当前推荐组合。
```

### 7.4 CARM

CARM 首轮曾因 Base 配置不一致产生不可比较结果。修正为严格加载 P0 bounded CES-HMoE checkpoint 后，ETTh2 H=192 的多组实验仍未降低测试 MSE。

因此当前 CARM 是失败增强分支，不纳入主模型。

## 8. 统一 seed 稳定性

ETTh2、`seq_len=96` 上，固定当前最佳架构，使用：

```text
master_seed = 2024/2025/2026
```

结果：

| H | MSE均值±标准差 | WTR均值±标准差 |
|---:|---:|---:|
| 24 | 0.0703±0.0005 | 69.568±0.289 |
| 96 | 0.1280±0.0022 | 57.331±0.436 |
| 192 | 0.1767±0.0012 | 50.725±0.058 |
| 336 | 0.2197±0.0017 | 45.700±0.179 |
| 720 | 0.3070±0.0064 | 36.058±0.406 |

严格配对的 v1/v1/v1 baseline 使用相同窗口、相同训练流程和相同 master seed 重新训练。v2/v2/v1 在五个预测长度上均优于 baseline。

## 9. 四数据集 80 组实验

批量入口：

```text
run_best_model_4datasets.ps1
```

截至 2026-08-25：

```text
ETTh1：20/20
ETTh2：20/20
ETTm1：20/20
ETTm2：20/20
总计：80/80，全部成功
```

结果：

```text
E:\power2\logs\best_model_4datasets_seed2024\summary.tsv
```

按测试 MSE 最优输入窗口的概览：

| 数据集 | H | 最优 SL | MSE | MAE | WTR |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 24 | 336 | 0.030 | 0.140 | 65.750 |
| ETTh1 | 96 | 336 | 0.060 | 0.190 | 54.290 |
| ETTh1 | 192 | 336 | 0.080 | 0.210 | 48.820 |
| ETTh1 | 336 | 336 | 0.100 | 0.250 | 42.230 |
| ETTh1 | 720 | 168 | 0.180 | 0.350 | 27.390 |
| ETTh2 | 24 | 96 | 0.070 | 0.200 | 69.640 |
| ETTh2 | 96 | 96 | 0.130 | 0.280 | 57.150 |
| ETTh2 | 192 | 336 | 0.170 | 0.330 | 49.310 |
| ETTh2 | 336 | 720 | 0.190 | 0.360 | 44.420 |
| ETTh2 | 720 | 720 | 0.270 | 0.420 | 38.840 |
| ETTm1 | 24 | 168 | 0.010 | 0.080 | 84.300 |
| ETTm1 | 96 | 720 | 0.030 | 0.130 | 68.440 |
| ETTm1 | 192 | 720 | 0.050 | 0.160 | 59.250 |
| ETTm1 | 336 | 720 | 0.070 | 0.200 | 52.090 |
| ETTm1 | 720 | 720 | 0.090 | 0.220 | 48.440 |
| ETTm2 | 24 | 336 | 0.030 | 0.110 | 87.270 |
| ETTm2 | 96 | 336 | 0.070 | 0.190 | 72.270 |
| ETTm2 | 192 | 336 | 0.090 | 0.230 | 64.050 |
| ETTm2 | 336 | 336 | 0.120 | 0.270 | 58.480 |
| ETTm2 | 720 | 336 | 0.170 | 0.320 | 51.690 |

窗口选择和模型比较必须在相同数据集、相同 `seq_len`、相同 `pred_len` 下进行。

## 10. 过拟合、欠拟合与效率

当前日志没有显示严重过拟合：

```text
Stage-2 early stopping；
Stage-3 仅 gate-only 弱微调；
验证集 checkpoint 选择；
统一 seed 下结果波动较小。
```

长预测存在优化平台和信息不足：

```text
验证 MSE 后期下降变慢；
趋势专家占主要权重；
H=720 的提升小于短预测。
```

当前模型较轻，GPU 利用率可能偏低。批量实验还可以优化：

```text
batch_size
num_workers
prefetch_factor
persistent_workers
```

## 11. 当前主模型

```text
TrendExpert-v2
PeriodicExpert-v2
RampExpert-v1
bounded horizon-wise gate
prior = 0.853,0.142,0.004
dynamic_blend = 0.20
entropy_state = disabled
```

该配置在 ETTh2 的统一三 seed 实验和四数据集 80 组批量实验中均已验证。

## 12. 后续可做消融

优先级：

1. 四数据集最佳窗口上的 `master_seed=2024/2025/2026` 稳定性；
2. `v1/v1/v1`、`v2/v1/v1`、`v1/v2/v1`、`v2/v2/v1` 的配对消融；
3. `dynamic_blend=0.05/0.10/0.20/0.30`；
4. global gate 与 horizon gate；
5. 高熵/低熵和高差分 Ramp 子集误差；
6. 参数量、训练时间、推理时间和 GPU 利用率。

不建议继续大规模投入：

```text
无约束 dynamic gate
全量熵强注入
高 Ramp 权重
concat
CARM 大参数网格
```

## 13. 可复现命令

单次最佳模型：

```powershell
python E:\power2\ces_hmoe_ettdataset.py `
  --dataset ETTh2 `
  --data_dir E:\power2\dataset `
  --seq_len 96 `
  --pred_len 192 `
  --master_seed 2024 `
  --trend_variant v2 `
  --periodic_variant v2 `
  --ramp_variant v1 `
  --fusion_mode bounded_gate `
  --gate_mode horizon `
  --prior_weights 0.853,0.142,0.004 `
  --dynamic_blend 0.20 `
  --disable_entropy_state `
  --stage1_fusion independent `
  --stage2_scope gate_only `
  --stage3_scope gate_only `
  --stage1_lr 1e-4 `
  --stage2_lr 1e-4 `
  --finetune_lr 1e-5 `
  --selection_metric weighted_wtr
```

四数据集 80 组：

```powershell
E:\power2\run_best_model_4datasets.ps1 `
  -Python E:\Anaconda\envs\T3Time\python.exe `
  -DataDir E:\power2\dataset `
  -LogRoot E:\power2\logs\best_model_4datasets_seed2024 `
  -MasterSeed 2024 `
  -GpuId 0
```

ETTh2 统一 seed：

```powershell
E:\power2\run_etth2_unified_seed_sl96.ps1 `
  -Python E:\Anaconda\envs\T3Time\python.exe `
  -DataDir E:\power2\dataset `
  -LogRoot E:\power2\logs\etth2_unified_seed_sl96_combo `
  -GpuId 0
```

## 14. 论文方法段

> We propose a causal horizon-wise mixture-of-experts model for ETT forecasting. The model contains three complementary experts: a residual-enhanced moving-average trend expert, a lightweight convolutional periodic expert augmented with explicit seasonal anchors, and a multivariate differenced TCN ramp expert. A bounded horizon-wise gate combines the expert forecasts using historical and future-calendar representations. The gate is regularized around a prior expert allocation of 0.853, 0.142, and 0.004, with a dynamic blend coefficient of 0.20, preventing unstable expert collapse while preserving horizon-dependent routing. The experts are independently pretrained, followed by frozen-expert gate training and gate-only fine-tuning. All windows, normalization statistics, calendar features, and optional entropy states are causal. Multi-scale entropy states are implemented as an optional mechanism and are disabled in the best stable configuration because the corresponding ablations did not provide consistent gains.

## 15. 报告规范

主表建议报告：

```text
MSE
MAE
RMSE
WTR5
WTR10
WTR15
Weighted-WTR
```

正式比较必须固定：

```text
数据切分
scaler
seq_len
pred_len
master_seed协议
训练预算
checkpoint选择规则
```

必须区分：

```text
fresh_train
posthoc_calibration
old_reproduction
```

不要将测试集用于反复选择结构或校准参数。

---

## 16. 当前代码的数据流与张量契约

本节按 `E:\power2\ces_hmoe_ettdataset.py` 的实际执行顺序说明代码。单个 batch 的核心张量为：

```text
x:               [B,L,7]
future_calendar: [B,H,6]
state:           [B,117]
y:               [B,H]
```

其中：

```text
B = batch size
L = seq_len
H = pred_len
7 = ETT数值通道数
6 = 未来日历特征数
117 = 三尺度熵状态维度
```

模型输出字典：

```python
{
    "prediction": [B,H],
    "weights":    [B,H,3],
    "experts":    [B,H,3],
}
```

`experts[...,0]`、`experts[...,1]`、`experts[...,2]` 分别对应：

```text
TrendExpert
PeriodicExpert
RampExpert
```

`weights` 的最后一维与专家顺序完全一致。

完整数据流：

```text
CSV
 -> 官方时间切分
 -> 训练段拟合 scaler
 -> 标准化全部数值通道
 -> 构建未来日历特征
 -> 构建因果熵缓存
 -> 生成滑动窗口 Dataset
 -> 三专家分别预测
 -> History/Calendar 编码
 -> horizon-wise gate
 -> bounded fusion
 -> prediction
 -> forecast loss / validation metrics
```

### 16.1 样本索引

`ETTWindowDataset` 保存窗口起点：

```python
self.origins = list(
    range(border1, border2 - seq_len - pred_len + 1)
)
```

读取第 `index` 个样本：

```python
origin = self.origins[index] + self.seq_len
x_start = origin - self.seq_len
x_end = origin
y_end = origin + self.pred_len
```

随后：

```python
x = values[x_start:x_end]
y = values[origin:y_end, target_idx]
future_calendar = calendar[origin:y_end]
state = entropy_cache[origin]
```

因此：

```text
x读取 [origin-L, origin)
y读取 [origin, origin+H)
```

输入与标签在时间上没有重叠。

### 16.2 验证和测试历史上下文

验证集使用：

```text
border1 = num_train - seq_len
border2 = num_train + num_val
```

测试集使用：

```text
border1 = num_train + num_val - seq_len
border2 = num_total
```

减去 `seq_len` 的目的不是泄漏，而是允许验证或测试的第一个预测原点读取其之前已经观测到的历史序列。

---

## 17. 数据预处理代码详解

### 17.1 数值列排序

代码先排除：

```text
date
target
```

然后把目标列追加到最后：

```python
numeric = [
    column for column in frame.columns
    if column not in {"date", target}
]
numeric.append(target)
```

这保证：

```python
target_idx = len(numeric) - 1
x[..., -1] == OT
```

### 17.2 缺失值

数值列先转换：

```python
pd.to_numeric(errors="coerce")
```

缺失值处理：

```python
train_mean = frame.loc[:num_train - 1, numeric].mean()
frame[numeric] = frame[numeric].ffill().fillna(train_mean)
```

含义：

1. 中间缺失值只使用过去值前向填充；
2. 序列开头无法前向填充的值使用训练段均值；
3. 不使用验证段或测试段均值。

### 17.3 标准化器

`NumpyScaler.fit()` 计算：

```python
mean = train_values.mean(axis=0)
std = train_values.std(axis=0)
```

并限制：

```python
std = np.maximum(std, 1e-6)
```

标准化：

$$
x'=\frac{x-\mu_{train}}{\sigma_{train}}.
$$

目标反标准化：

$$
\hat y_{raw}
=\hat y_{norm}\sigma_{OT}+\mu_{OT}.
$$

### 17.4 Ramp 阈值

训练段目标差分：

```python
diff = np.diff(train_OT)
```

阈值：

```python
ramp_threshold = quantile(abs(diff), 0.90)
```

该阈值同时用于：

```text
历史状态中的 ramp_ratio
训练损失中的 ramp_mask
```

并且固定由训练段计算。

---

## 18. 熵函数逐项实现

### 18.1 z-score

熵函数内部使用：

```python
z = (x - x.mean()) / (x.std() + EPS)
```

该 z-score 是针对单个历史片段的局部标准化，不是数据集 scaler。

### 18.2 模板构造

`_templates(x,m)` 使用：

```python
np.lib.stride_tricks.sliding_window_view(x, m)
```

例如长度为 `N`、模板长度为 `m` 时，输出：

```text
[N-m+1,m]
```

### 18.3 排列熵

排列熵使用：

```text
order = 3
delay = 1
```

构造局部向量：

$$
v_i=[x_i,x_{i+1},x_{i+2}].
$$

对每个向量稳定排序得到 ordinal pattern：

```python
patterns = np.argsort(vectors, axis=1, kind="stable")
```

归一化熵：

$$
PE=-\frac{\sum_\pi p(\pi)\log(p(\pi)+\epsilon)}
{\log(3!)}.
$$

输出范围通常位于：

```text
[0,1]
```

### 18.4 谱熵

先去均值：

```python
z = x - mean(x)
```

计算一侧功率谱：

```python
power = abs(rfft(z)) ** 2
```

去除 DC：

```python
probability = power[1:] / sum(power[1:])
```

谱熵：

$$
SpE=-\frac{\sum_kq_k\log(q_k+\epsilon)}
{\log K}.
$$

### 18.5 SampEn

SampEn 参数：

```text
m = 2
r_ratio = 0.2
```

对 `m` 和 `m+1` 模板分别计算 Chebyshev 距离：

```python
distance = max(abs(template_i-template_j))
```

排除 self-match 后：

$$
SampEn=-\log\frac{A+\epsilon}{B+\epsilon}.
$$

### 18.6 ApEn

ApEn 保留 self-match：

$$
ApEn=\Phi^m(r)-\Phi^{m+1}(r).
$$

代码对每个模板计算匹配比例，再计算：

```python
mean(log(match_rate + EPS))
```

### 18.7 状态拼接顺序

每个尺度先按数值通道拼接：

```text
channel0: PE,SpE,mean,std,median
channel1: PE,SpE,mean,std,median
...
channel6: PE,SpE,mean,std,median
```

然后追加目标特征：

```text
SampEn,ApEn,slope,ramp_ratio
```

三个尺度按配置顺序依次拼接。

---

## 19. 熵缓存逐步实现

缓存张量：

```text
cache: [N+1,117]
```

缓存索引直接对应预测原点：

```text
cache[origin]
```

每个原点只允许读取：

```text
values[origin-seq_len:origin]
```

刷新条件：

```python
origin == seq_len
or (origin-seq_len) % refresh_stride == 0
```

昂贵熵刷新条件：

```python
origin == seq_len
or (origin-seq_len) % expensive_stride == 0
```

如果当前原点不计算 SampEn/ApEn，代码先写入 `NaN`，再使用：

```python
fresh[np.isnan(fresh)] = current[np.isnan(fresh)]
```

即只继承过去已经计算的值。

缓存文件中同时保存：

```text
values
calendar
entropy_cache
columns
target_idx
borders
ramp_threshold
scaler_mean
scaler_std
```

因此命中缓存后不需要重新读取 CSV 和重新计算熵。

---

## 20. 三专家代码详解

### 20.1 TrendExpert-v2

输入：

```text
x: [B,L,7]
```

目标通道：

```python
target = x[..., -1:].transpose(1,2)
```

形状：

```text
[B,1,L]
```

移动平均核：

```python
kernel = min(
    25,
    seq_len if seq_len is odd else seq_len-1
)
```

这样保证卷积核为奇数，且不会超过输入长度。

趋势项：

```python
trend = AvgPool1d(
    kernel,
    stride=1,
    padding=kernel//2,
    count_include_pad=False,
)(target)
```

基础预测：

```python
base_forecast = Linear(L,H)(trend)
```

残差：

```python
residual = target - trend
```

残差预测：

```python
residual_forecast = Linear(L,H)(residual)
```

最终：

$$
\hat y^{trend}
=\hat y^{smooth}
+0.1\hat y^{residual}.
$$

该结构相对 v1 增加一个 `Linear(L,H)`，但仍然是轻量线性专家。

### 20.2 PeriodicExpert-v2

目标输入：

```text
[B,1,L]
```

CNN 路径：

```text
Conv1d(1,32,kernel=25,padding=12)
GELU
Depthwise Conv1d(32,32,kernel=9,padding=4)
GELU
AdaptiveAvgPool1d(8)
Flatten -> [B,256]
```

季节锚点：

```python
anchors = [
    target[-24],
    target[-168],
    target[-336],
]
```

如果历史窗口小于 lag，使用：

```python
target[:,0]
```

作为回退。

锚点映射：

```text
[B,3]
 -> Linear(3,64)
 -> [B,64]
```

拼接：

```text
[B,256] + [B,64] -> [B,320]
```

输出：

```text
Linear(320,H) -> [B,H]
```

### 20.3 RampExpert-v1

输入一阶差分：

```python
diff = torch.diff(
    x,
    dim=1,
    prepend=x[:,:1],
)
```

形状仍为：

```text
[B,L,7]
```

转置：

```text
[B,7,L]
```

TCN：

```text
Conv1d(7,48,kernel=3,dilation=1,padding=1)
GELU
Conv1d(48,48,kernel=3,dilation=2,padding=2)
GELU
Conv1d(48,48,kernel=3,dilation=4,padding=4)
GELU
```

代码读取最后一个时间位置：

```python
hidden = net(diff)[:, :, -1]
```

再使用：

```python
Linear(48,H)
```

输出完整 `H` 步预测。

---

## 21. HistoryEncoder 与门控输入

### 21.1 HistoryEncoder

```text
x [B,L,7]
 -> transpose [B,7,L]
 -> Conv1d(7,64,kernel=5)
 -> GELU
 -> Conv1d(64,64,kernel=5,dilation=2)
 -> GELU
 -> mean over time
 -> z_hist [B,64]
```

### 21.2 熵状态编码器

启用时：

```text
state [B,117]
 -> LayerNorm(117)
 -> Linear(117,64)
 -> GELU
 -> z_state [B,64]
```

当前最佳配置关闭该分支：

```text
use_entropy_state=False
```

### 21.3 日历编码器

```text
future_calendar [B,H,6]
 -> Linear(6,64)
 -> GELU
 -> [B,H,64]
```

与：

```text
horizon_embedding [H,64]
```

相加。

### 21.4 当前 route 张量

当前关闭熵状态：

```python
route = cat([
    z_hist[:,None,:].expand(-1,H,-1),
    z_calendar + horizon_embedding[None,:,:],
], dim=-1)
```

形状：

```text
[B,H,128]
```

Gate：

```text
Linear(128,64)
GELU
Linear(64,3)
```

输出：

```text
logits [B,H,3]
```

---

## 22. Bounded Gate 代码详解

初始化专家先验：

```python
prior = tensor([0.853,0.142,0.004])
prior = prior / prior.sum()
static_logits = log(prior)
```

前向传播时：

```python
prior = softmax(static_logits)
```

Trend/Periodic 先验：

```python
prior_pair = prior[:2] / prior[:2].sum()
```

动态二专家分配：

```python
dynamic_pair = softmax(
    log(prior_pair) + pair_logits / temperature
)
```

拼回 Ramp：

```python
dynamic_weights = cat([
    dynamic_pair * (1-prior_ramp),
    prior_ramp,
])
```

最终：

```python
weights = (
    (1-dynamic_blend)*prior
    + dynamic_blend*dynamic_weights
)
```

当前：

```text
dynamic_blend=0.20
```

最终预测：

```python
expert_predictions = stack([
    trend_pred,
    periodic_pred,
    ramp_pred,
], dim=-1)

prediction = (
    weights * expert_predictions
).sum(dim=-1)
```

形状：

```text
expert_predictions [B,H,3]
weights            [B,H,3]
prediction         [B,H]
```

### 22.1 动态路由的实际含义

当前测试日志中的：

```text
trend_weight
periodic_weight
ramp_weight
```

是对所有测试样本和 horizon 求平均后的比例。

真正的权重仍然是：

```text
weights[b,h,e]
```

因此即使平均值接近：

```text
0.70/0.296/0.004
```

不同样本和不同预测步仍可有差异。

---

## 23. 损失函数逐项说明

### 23.1 基础 Huber

```python
base = smooth_l1_loss(
    prediction,
    target,
)
```

### 23.2 斜率损失

```python
true_diff = target[:,1:] - target[:,:-1]
pred_diff = prediction[:,1:] - prediction[:,:-1]
slope = abs(true_diff-pred_diff).mean()
```

### 23.3 Ramp 加权损失

```python
ramp_mask = (
    abs(true_diff) >= ramp_threshold
).float()
```

```python
ramp = (
    smooth_l1_loss(
        pred_diff,
        true_diff,
        reduction="none",
    )
    * (1+ramp_mask)
).mean()
```

最终：

$$
L_{forecast}
=L_{Huber}
+0.2L_{slope}
+0.3L_{ramp}.
$$

### 23.4 独立专家损失

Stage 1 中：

```python
losses = [
    forecast_loss(experts[...,0], target),
    forecast_loss(experts[...,1], target),
    forecast_loss(experts[...,2], target),
]
loss = mean(stack(losses))
```

因此三个专家均直接面对真实目标，而不是只依赖融合输出。

### 23.5 当前关闭的附加损失

当前最佳配置：

```text
balance_weight = 0
horizon_balance_weight = 0
wtr_loss_weight = 0
route_loss_weight = 0
```

代码仍保留这些选项做消融，但正式主模型不使用。

---

## 24. 分阶段训练代码行为

### 24.1 Stage 1

```text
trainable:
TrendExpert
PeriodicExpert
RampExpert
```

冻结：

```text
HistoryEncoder
CalendarEncoder
Gate
```

当前：

```text
60 epochs
lr=1e-4
fusion=independent
```

### 24.2 Stage 2

```text
trainable:
Gate
```

冻结专家和其余路由编码器：

```text
30 epochs maximum
lr=1e-4
patience=5
```

Stage 2 会先评估先验初始状态：

```text
epoch=0 candidate
```

如果训练 gate 后变差，可以恢复先验候选。

### 24.3 Stage 3

```text
trainable:
Gate
```

```text
10 epochs
lr=1e-5
patience=10
```

当前 Stage 3 不解冻专家。

### 24.4 验证候选

每个非 Stage 1 epoch 保存：

```python
{
    "epoch": epoch,
    "mse": val_mse,
    "mae": val_mae,
    "weighted_wtr": val_weighted_wtr,
    "state": model_state,
}
```

最终按：

```text
selection_metric=weighted_wtr
```

选择 checkpoint。

测试集不参与候选选择。

---

## 25. 指标代码详解

标准化尺度：

```python
mse = mean(error**2)
rmse = sqrt(mse)
mae = mean(abs(error))
```

原始尺度误差：

```python
raw_error = error * target_std
raw_target = target*target_std + target_mean
```

MAPE 的分母下限：

```python
mape_floor = max(
    0.01*target_range,
    1e-6,
)
```

相对误差：

```python
relative_error = abs(raw_error) / target_range
```

命中率：

```python
wtr5  = mean(relative_error <= 0.05)*100
wtr10 = mean(relative_error <= 0.10)*100
wtr15 = mean(relative_error <= 0.15)*100
```

加权 WTR：

$$
WeightedWTR
=0.5WTR_5
+0.3WTR_{10}
+0.2WTR_{15}.
$$

### 25.1 专家权重统计

```python
trend_weight = mean(weights[...,0])
periodic_weight = mean(weights[...,1])
ramp_weight = mean(weights[...,2])
```

整体标准差：

```python
weights[...,e].std()
```

样本间标准差：

```python
weights[...,e].mean(dim=1).std()
```

horizon 间标准差：

```python
weights[...,e].mean(dim=0).std()
```

主导专家比例：

```python
dominant = weights.argmax(dim=-1)
mean(dominant == expert_id)
```

---

## 26. 当前代码参数与推荐值

| 参数 | 当前推荐值 | 作用 |
|---|---:|---|
| `trend_variant` | `v2` | 趋势残差增强 |
| `periodic_variant` | `v2` | CNN + 季节锚点 |
| `ramp_variant` | `v1` | 原始差分 TCN |
| `fusion_mode` | `bounded_gate` | 先验约束动态融合 |
| `gate_mode` | `horizon` | 逐预测步路由 |
| `prior_weights` | `0.853,0.142,0.004` | 专家先验 |
| `dynamic_blend` | `0.20` | 动态修正比例 |
| `disable_entropy_state` | `True` | 当前最佳不输入熵 |
| `gate_init` | `zero` | 从先验稳定起点开始 |
| `stage1_fusion` | `independent` | 独立专家预训练 |
| `stage2_scope` | `gate_only` | 冻结专家训练 gate |
| `stage3_scope` | `gate_only` | gate 弱微调 |
| `stage1_epochs` | `60` | 专家训练轮数 |
| `stage2_epochs` | `30` | gate 最大训练轮数 |
| `epochs` | `100` | 总轮数 |
| `stage1_lr` | `1e-4` | 专家学习率 |
| `stage2_lr` | `1e-4` | gate 学习率 |
| `finetune_lr` | `1e-5` | Stage 3 学习率 |
| `balance_weight` | `0` | 不强制平均专家使用 |
| `horizon_balance_weight` | `0` | 不强制 horizon 平均 |
| `wtr_loss_weight` | `0` | WTR 不直接参与训练 |
| `route_loss_weight` | `0` | 不使用 oracle 路由监督 |
| `selection_metric` | `weighted_wtr` | 验证 checkpoint 选择 |
| `master_seed` | 可配置 | 统一随机源 |

---

## 27. 完整防泄漏代码检查

### 数据层

- CSV 按时间顺序切分；
- scaler 只拟合训练段；
- 初始缺失值只使用训练均值；
- 验证和测试不参与 scaler；
- 输入窗口右端为预测原点，不含标签。

### 熵层

- `entropy_cache[origin]` 只读取 `values[:origin]`；
- 多尺度片段均以 `origin` 为右端；
- SampEn/ApEn 前向保持只继承过去值；
- ramp threshold 只由训练段计算。

### 路由层

- future calendar 只由 `date` 生成；
- 不输入未来实测值；
- 当前熵状态关闭，不影响因果性；
- horizon embedding 是可学习参数，不包含未来标签。

### 训练和评估层

- 验证集选择 checkpoint；
- 测试集只用于最终评估；
- 测试结果不能用于重新选择窗口和参数；
- old reproduction、fresh train、posthoc calibration 分开记录。

---

## 28. 当前代码验证

基础编译：

```powershell
& 'E:\Anaconda\envs\T3Time\python.exe' `
  -m py_compile `
  'E:\power2\ces_hmoe_ettdataset.py'
```

Smoke test：

```powershell
& 'E:\Anaconda\envs\T3Time\python.exe' `
  'E:\power2\ces_hmoe_ettdataset.py' `
  --smoke-test
```

检查内容：

```text
prediction.shape == [B,H]
weights.shape == [B,H,3]
experts.shape == [B,H,3]
sum(weights,dim=-1) == 1
bounded Ramp权重正确
各融合分支可反向传播
```

当前 80 组四数据集实验：

```text
ETTh1 20/20
ETTh2 20/20
ETTm1 20/20
ETTm2 20/20
失败 0
```
