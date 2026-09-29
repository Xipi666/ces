# CES-HMoE 8/20–8/26 实验记录复现报告

> 生成时间：2026-09-29
> 记录来源：Codex Desktop 会话历史（`~/.codex/sessions/2026/08/`）
> 代码基准：本仓库最新版本 `ces_hmoe_ettdataset.py`（commit `cc0aaea`，2026-08-29，即实验结束后的最终版）

---

## 一、Codex 聊天记录检索结果

在 2026-08-20 ~ 08-26 的 Codex 会话中，检索到以下与 ces 相关的会话（按相关性排序）：

| 日期 | 会话文件（`~/.codex/sessions/2026/08/` 下） | 内容 |
|---|---|---|
| 08-22 16:01 | `22/rollout-...-01a0287d-*.jsonl`（9.2 MB） | WTR 大实验计划（P0~P4）制定与启动、P0 基线矩阵 |
| 08-23 04:47 | `23/rollout-...-01a02b3a-*.jsonl`（7.5 MB） | P1/P2/P3/P4 消融、CARM 判定、统一 master_seed、启动 80 组扫描 |
| 08-21 05:33 | `21/rollout-...-01a02117-*.jsonl`（5.8 MB） | 上一轮 80 组 seq_len 搜索结果分析、下一轮优化方案 |
| 08-23 21:06 | `23/rollout-...-01a02eba-*.jsonl`（1.1 MB） | P2 多种子确认、专家种子确认、P4 CARM 收尾 |
| 08-25 17:28 | `25/rollout-...-01a0383f-*.jsonl`（948 KB） | **80 组四数据集最终结果汇总、导出报告与 Excel** |
| 08-26 13:13 | `26/rollout-...-01a03c7d-*.jsonl` | 结果表整理（pl/sl + 9 指标）、method v2 文档修改 |

原始工作目录为 `E:\power2`（Python：`E:\Anaconda\envs\T3Time\python.exe`），该目录已不存在；本复现全部使用仓库内最新代码。

---

## 二、8/20–8/26 实验时间线与结论（原始记录）

模型主线（TrendExpert + PeriodicExpert + RampExpert → CES 路由 → bounded horizon gate → 加权预测）不变，按阶段推进：

### P0（8/22）：重建干净基线
- ETTh2，`pred_len={24,96,192,336,720} × seq_len={96,168,336,720}`，seed 2024，20 组
- 固定配置：`bounded_gate` / `horizon` gate / 熵状态禁用 / `stage1_fusion=independent` / `stage2、stage3=gate_only` / `prior=0.853,0.142,0.004` / `dynamic_blend=0.20`

### P1（8/22）：WTR 专用训练目标
- `wtr_loss_weight ∈ {0,0.01,0.03,0.05,0.10,0.20}`、`wtr_temperature`、horizon 权重（uniform/early/late）
- **结论：WTR surrogate 损失没有带来稳定收益，最终主模型采用 `wtr_loss_weight=0`**

### P2（8/22–8/23）：专家结构消融（18 组）
| 变体 | H | MSE | WTR | 结论 |
|---|---:|---:|---:|---|
| Trend-v2 | 24 | 0.072 | 68.904 | **明显有效**（P0 同窗口 0.091 / 62.602） |
| Periodic-v2 + Trend-v2 | 24 | 0.072 | 69.072 | 当前最好 |
| Periodic-v2 / v3、Ramp-v2 / v3 | 24 | 0.090–0.092 | ~62.5 | 无效或退化 |
- P2 多种子确认（24 组）：gate/training seed 2024/2025/2026、专家种子 137/2025/2026 均复现改善，`TrendExpert-v2` 被确认为可靠改进

### P3（8/23）：融合结构（12 组）
- block gate / WTR utility gate 等与 P2 基本持平；**维持 bounded horizon-wise gate**

### P4（8/23–8/24）：CARM（因果残差记忆）
- 严格匹配 P0 的 `p4_carm_bounded`：H=192 全部 14/14 变差（最好 +0.001，最差 +0.010，alpha 越大越差）
- **结论：CARM 对 ETTh2 无收益，不进入主模型**（仓库中 `ces_hmoe_carm_ettdataset.py` 与 `logs/etth2_*` 即此分支的后续调试记录）
- 早先 `CARM MSE=1.321 vs Base 1.312` 属配置不匹配，不作正式证据

### 统一随机种子（8/24）
- 将 `seed / init_seed / gate_seed / loader_seed` 统一为单一 `--master_seed`，验证 2024/2025/2026 稳定可复现

### 最终 80 组四数据集扫描（8/24–8/25）
- 4 数据集 × 5 pred_len × 4 seq_len = 80 组，`MasterSeed=2024`，`80/80 全部成功`
- 最终模型：**TrendExpert-v2 + PeriodicExpert-v2 + RampExpert-v1 + bounded horizon-wise gate**

---

## 三、原始参考结果（8/25 会话汇总，MasterSeed=2024）

ETTh2（按最低 MSE 选窗口；括号内为该 H 的最高 WTR）：

| H | 最佳 SL | MSE | MAE | RMSE | WTR | 最高 WTR（SL=96） |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 96 | 0.070 | 0.200 | 0.265 | 69.640 | 69.640 |
| 96 | 96 | 0.129 | 0.276 | 0.359 | 57.146 | 57.146 |
| 192 | 336 | 0.173 | 0.329 | 0.416 | 49.314 | 50.645 |
| 336 | 720 | 0.193 | 0.358 | 0.440 | 44.424 | 45.812 |
| 720 | 720 | 0.270 | 0.420 | 0.519 | 38.842 | 38.842 |

四数据集平均（按每 H 最低 MSE）：

| 数据集 | 平均最佳 MSE | 平均 WTR |
|---|---:|---:|
| ETTh1 | 0.0904 | 47.695 |
| ETTh2 | 0.1670 | 51.873 |
| ETTm1 | 0.0480 | 62.506 |
| ETTm2 | 0.0958 | 66.753 |

---

## 四、本次复现配置

- **代码**：仓库最新 `ces_hmoe_ettdataset.py`（以最后版本为准）
- **环境**：`G:\Anaconda3\envs\wrj\python.exe`（Python 3.11.16，torch 2.14.0+cu126，RTX 3060 Laptop 6GB）
- **运行脚本**：`_repro/run_matrix.sh`（逐条复刻原 `run_best_model_4datasets.ps1` 的参数；支持 DONE 断点续跑，结果增量写入 `_repro/logs/<矩阵>/summary.tsv`）

单组命令（与原脚本一字不差）：

```bash
python ces_hmoe_ettdataset.py \
  --dataset ETTh2 --data_dir dataset \
  --seq_len $SL --pred_len $PL \
  --epochs 100 --stage1_epochs 60 --stage2_epochs 30 \
  --stage2_patience 5 --stage3_patience 10 \
  --batch_size 32 --lr 3e-5 \
  --stage1_lr 1e-4 --stage2_lr 1e-4 --finetune_lr 1e-5 \
  --stage1_fusion independent --stage2_scope gate_only --stage3_scope gate_only \
  --fusion_mode bounded_gate --gate_mode horizon \
  --prior_weights 0.853,0.142,0.004 --dynamic_blend 0.20 \
  --disable_entropy_state --gate_init zero \
  --balance_weight 0 --horizon_balance_weight 0 \
  --route_loss_weight 0 --wtr_loss_weight 0 \
  --selection_metric weighted_wtr --selection_mse_tolerance 0.01 \
  --min_epochs 15 --patience 15 \
  --master_seed $SEED --device auto \
  --trend_variant v2 --periodic_variant v2 --ramp_variant v1 \
  --save_best_checkpoint <run>/best.pt --suppress_horizon_weights
```

复现队列（后台顺序执行）：

1. `ETTh2 × {96,168,336,720} × {24,96,192,336,720}`，seed 2024（20 组，对应 P0 矩阵 + 最终主模型）
2. `ETTh2 × sl=96 × 5 个 pl`，seed 2025 / 2026（10 组，对应统一多种子稳定性实验）

> 注：原实验在 `E:\Anaconda\envs\T3Time`（torch 版本不详）上运行；本机使用 torch 2.14 + 3060，同一 seed 的浮点结果不保证逐位一致，比较看数量级与相对关系。

---

## 五、复现进度与对照

首轮复现（2026-09-29 晚）已完成并验证的组（数值取自 driver 日志；首轮因本机出现重复运行器进程导致部分日志互相覆盖，已清理，单实例队列重跑中）：

| run_id | sl | pl | seed | 原始 test_mse | 原始 wWTR | 复现 test_mse | 复现 wWTR | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh2_sl96_h24_seed2024 | 96 | 24 | 2024 | 0.070 | 69.640 | 0.07 | 69.64 | **完全一致** |
| ETTh2_sl96_h192_seed2024 | 96 | 192 | 2024 | 0.178 | 50.645 | 0.178 | 50.646 | **完全一致** |
| ETTh2_sl96_h96_seed2024 | 96 | 96 | 2024 | 0.129 | 57.146 | 状态 ok，日志被覆盖 | | 重跑中 |

> 首轮复现中 `ETTh2_sl96_h24_seed2024` 的 `Selected Val: mse=0.095545 weighted_wtr=69.605`、`Test: {'mse': 0.07, ..., 'wtr5': 52.873, 'wtr10': 81.427, 'wtr15': 93.878, 'weighted_wtr': 69.64}` 与 8/26 原始会话记录中该组日志（val mse 0.096、wWTR≈69.5、Test mse 0.07）逐项吻合——同代码同种子跨环境可复现。

当前后台队列（单实例、带互斥锁、DONE 断点续跑）按以下顺序重跑全部 30 组，完成后以 `_repro/logs/etth2_matrix_seed*/summary.tsv` 为准：

1. ETTh2 × sl=96 × 5 pl，seed 2024（含全部 WTR 最优窗口行）
2. ETTh2 × sl=336/720 × pl=192/336/720，seed 2024（长预测 MSE 最优窗口）
3. ETTh2 × sl=168/336/720 × pl=24/96，seed 2024
4. ETTh2 × sl=168/336/720 × pl=192/336/720，seed 2024（补全 20 组矩阵）
5. ETTh2 × sl=96 × 5 pl，seed 2025 / 2026（统一多种子稳定性）

---

## 六、如何继续

```bash
# 查看进度
tail -f _repro/logs/repro_queue_driver.log
cat _repro/logs/etth2_matrix_seed2024/summary.tsv

# 断点续跑（已完成的组自动跳过）
bash _repro/run_matrix.sh ETTh2 2024

# 之后扩展到其他数据集（对应原 80 组的剩余 60 组）
bash _repro/run_matrix.sh ETTh1 2024
bash _repro/run_matrix.sh ETTm1 2024
bash _repro/run_matrix.sh ETTm2 2024
```
