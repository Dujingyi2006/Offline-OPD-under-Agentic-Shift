# 进度交接 (handoff) — 2026/06/13

新对话开头先读这份 + `report.md`，即可无缝接上。持久记忆在
`C:\Users\25066\.claude\projects\c--Users-25066-Desktop-0---\memory\`
（`opd-multiturn-experiment.md`、`opd-toy-mdp-defense.md`、`MEMORY.md` 索引）。

---

## 一句话项目定位

无 GPU 的玩具实验，证明 **Lightning OPD**（离线在线策略蒸馏，`t2.pdf`）在它自己
附录 E 标注为开放问题的**多轮 agent 场景**下失效；对照 **Rang et al.** 离线全分布
KD（`对比论文.pdf`）。代码在 `opd_toy/`，结果在 `results/`，论文在 `report.md`。

核心机制：多轮里错误累积 → 进入 `π_ref` 没访问过的错误分支 → 优势信号
`A_t = log π_T − log π_θ` 在那里无定义，且补救动作 `ask_followup` 从不在黄金路径上。
这是**覆盖失败**（假设 3.2 被 agent 自己的错误违反），不是优化失败。

---

## 当前状态：✅ 全部完成并已用新数据重跑校验

最近一次完整 sweep 跑完时间约 2026/06/13 21:08，CPU 上约 16–18 分钟。

### 文件状态
- `env.py` — `AgenticTroubleshootMDP`，无改动。
- `policies.py` — 已加 `TabularPolicy`（per-state 查表，无共享权重，off-support 零梯度）。
- `methods.py` — 已加并验证：`train_{sft,offline_opd,rang_kd,online_opd}_tabular`
  + `_softmax_row` helper。离线循环已向量化。
- `run_experiments.py` — 已加 `run_representation()` + `_train_reference_tabular()`，
  `_row()` 多了 `representation` 列，已接入 `main()`。
- `plots.py` — 已加 `plot_representation` → `fig_representation.png`。
- `report.md` — **已完成**，含：§5.3 表示消融、**§6.1 DAgger 批判**、
  **§6.2 Rang 100% 压力测试**、§7 三点 takeaway。§3 主表 note 加了 ⚠ 标注。

### 产物（results/，均为最新）
`results_{main,coverage,deploy,representation}.csv`、`case_studies.txt`、
`fig_{main,coverage,deploy,representation}.png`。

### 关键数字（已校验，与 report 表一致）
主结果(5 seeds, collect=0.05, deploy=0.30):
teacher/online_rl/online_opd/**rang_kd = 1.000**;
**Lightning OPD = 0.694 ± 0.209**(退化到 SFT 0.699 且方差最大);
support-aware = 0.780;DAgger = 0.873。

表示消融(deploy=0.30, 按 collect_noise 0/0.05/0.15/0.30):
- linear rang_kd: 0.699 / 1.000 / 1.000 / 1.000
- tabular rang_kd: 0.697 / 0.780 / 0.903 / 0.919  ← **永不到 1.000**
- linear/tabular offline_opd: 都贴着 ~0.67–0.76
- 零覆盖(collect=0)时 tabular 下 sft=opd=rang 全部=0.697 完全相同

---

## 用户最近两个批判性思考点（已落地到 report，新对话要保持这个立场）

1. **DAgger 补丁(patch 2)是"伪装成离线的在线学习"，不算 OPD 的修复**(§6.1)。
   它中途调 `env.step` + 老师在线标注，花掉的正是 Lightning OPD 要省的环境访问。
   既然付了在线代价就该直接跑 Online RL/online OPD(都 1.000,碾压 DAgger 0.87)。
   所以 0.87 **支持**论点：闭合覆盖缺口必须删掉 Lightning OPD 的定义性前提。
   → 不要再把 DAgger 当合法离线修复。

2. **Rang 的 100% 不可疑、可解释、且脆弱**(§6.2):
   (a) collect=0 时掉回 SFT 地板 0.699;
   (b) 抽掉泛化(tabular)封顶 ≤0.92 永不到 1.000;
   (c) 机制 = 全分布 KL 把 `ask_followup` 尾部质量(~0.07,lam 加权)写进 target +
   共享权重广播到 off-support。100% = 全分布目标 × 泛化 × 有覆盖,缺一即退化。

诚实结论:no-env-access 侧前沿是 {support-aware, Rang},online 侧是
{online OPD, online RL},DAgger 严格被支配落在中间。

---

## 复现
```powershell
cd opd_toy
python run_experiments.py   # 写 ../results/*.csv + case_studies.txt(~16–18 min)
python plots.py             # 写 ../results/fig_*.png
```

## Windows 注意事项
- TaskStop **不回收子 python 进程**。跑前/杀进程后用 `Get-Process python` 检查,
  `Stop-Process -Id <pid> -Force` 清残留,否则 CPU 争用拖慢。
- PowerShell:用 `foreach($i in 1..3){...}` 不要用 `%%`;`$null` 不是 `/dev/null`。

---

## 可能的后续(用户尚未要求,仅备选)
- 若导师质疑玩具性:见 memory `opd-toy-mdp-defense.md` 的辩护话术。
- 可考虑把 report 译成对应语言/做成 PDF,或补一张 §6.1 的"诚实选项前沿"示意图。
- case_studies.txt 目前只覆盖 linear;若需要可加 tabular 的轨迹对照。
