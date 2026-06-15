# Offline On-Policy Distillation under Multi-Turn Distribution Shift

A small, GPU-free experiment that stress-tests the central assumption of
**Lightning OPD** (offline on-policy distillation, `t2.pdf`) against the
**multi-turn agentic** setting its own Limitations section flags as open, and
contrasts it with the offline on-policy KD of **Rang et al.** (openPangu
Embedded-1B, `对比论文.pdf`).

## The claim under test

Lightning OPD precomputes the teacher's per-token log-probs over rollouts from a
*frozen* reference policy `π_ref`, then trains the student with the
advantage-weighted policy gradient

```
A_t = log π_T(a_t | s_t) − log π_θ(a_t | s_t)
```

reusing the precomputed teacher signal. Its guarantees rest on **Assumption 3.2
(Support Coverage)** plus the argument that "dense teacher supervision leaves no
out-of-distribution region, so conservatism is unnecessary." Appendix E names
multi-turn agent interaction / multi-turn distribution shift as the open
problem.

In a multi-turn agent, errors *compound*: one wrong tool call lands the agent in
states `π_ref` never visits, so the precomputed teacher signal does not cover
them — and the recovery skill needed there was never distilled.

## Task

`AgenticTroubleshootMDP`: a multi-turn fault-diagnosis episode. The expert
follows a thin "golden path" (gather the discriminative tool, then commit). Under
deployment-time tool noise a discriminative tool can return a *false negative*
that rules the true fault out, pushing the agent into an **error branch** whose
only escape is the `ask_followup` recovery action — an action that never appears
on the clean golden path.

The student is a **feature-based linear-softmax policy** (shared weights), so it
*generalizes* its on-path behaviour into unseen states the way a real model
does — and there it misfires, rather than recovering by random chance.

## Methods (`methods.py`)

| method | family | env access | teacher signal |
|---|---|---|---|
| `train_sft` | imitation | clean demos | none |
| `train_online_rl` | online RL (REINFORCE) | full | none |
| `train_offline_opd` | **Lightning OPD** | none (frozen `π_ref`) | sampled-action log-probs |
| `train_online_opd` | on-policy distill (upper bound) | full | live log-probs |
| `train_rang_kd` | **Rang et al.** offline KD | none (frozen `π_ref`) | full top-k distribution |
| `train_offline_opd_support_aware` | patch 1 | none | + density weighting & conservative anchor |
| `train_offline_opd_dagger` | patch 2 | bounded refresh | + teacher-annotated student rollouts |

The `*_tabular` variants (`train_sft_tabular`, `train_offline_opd_tabular`,
`train_rang_kd_tabular`, `train_online_opd_tabular`) train a **per-state lookup
policy** (`policies.TabularPolicy`) instead of the shared-weight linear student.
They power the representation ablation below.

## Ablation axes

* **Axis A — collection coverage** (`collect_noise`): how often `π_ref`'s own
  rollouts surface recovery states into the precomputed dataset.
* **Axis B — deployment shift** (`deploy_noise`): how often the error branch is
  actually triggered at test time.
* **Axis C — student representation** (linear vs tabular): to separate the two
  ways an offline method can cover an error branch — *parameter generalization*
  (shared weights interpolate behaviour into unseen states) vs *literal support*
  (a per-state table only knows states it was trained on). The tabular student
  has no generalization, so an off-support state gets **zero gradient** and stays
  at its SFT value. This isolates Lightning OPD's Assumption 3.2 down to pure
  support coverage.

## Run

```bash
cd opd_toy
python run_experiments.py     # writes ../results/*.csv and case_studies.txt
python plots.py               # writes ../results/fig_*.png
```

Results, figures, and the full write-up: see [report.md](report.md) and the
[results/](../results/) directory.
