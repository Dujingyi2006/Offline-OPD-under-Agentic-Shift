"""Multi-seed experiment sweep.

Produces three result tables (written to ``results/``):

  * ``results_main.csv``      -- all methods at a fixed operating point
                                 (collect_noise=0.05, deploy_noise=0.30), the
                                 headline comparison.
  * ``results_coverage.csv``  -- ablation axis A: offline methods vs collection
                                 coverage (collect_noise sweep), deploy fixed.
  * ``results_deploy.csv``    -- ablation axis B: methods vs deployment
                                 distribution shift (deploy_noise sweep).

Also saves qualitative case-study trajectories to ``results/case_studies.txt``.

The toy task is tiny, so the whole sweep runs on a CPU in a couple of minutes.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from env import AgenticTroubleshootMDP
from evaluate import capture_case_studies, evaluate
from methods import (
    collect_dataset,
    train_offline_opd,
    train_offline_opd_active_query,
    train_offline_opd_branch_replay,
    train_offline_opd_full_adv,
    train_offline_opd_tabular,
    train_online_opd,
    train_online_opd_tabular,
    train_online_rl,
    train_rang_kd,
    train_rang_kd_tabular,
    train_sft,
    train_sft_tabular,
)
from policies import Teacher

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# -- experiment configuration ----------------------------------------------
N_SEEDS = 5
EVAL_EPISODES = 1500
TEACHER_TEMP = 0.25

# data sizes / training budget (kept modest so the sweep is fast)
SFT_EPISODES, SFT_EPOCHS = 200, 220
OPD_EPISODES, OPD_ITERS = 200, 350
ONLINE_RL_ITERS, ONLINE_OPD_ITERS = 2500, 350

# operating point for the headline table
MAIN_COLLECT_NOISE = 0.05
MAIN_DEPLOY_NOISE = 0.30

COVERAGE_GRID = [0.0, 0.05, 0.15, 0.30]   # collect_noise (axis A)
DEPLOY_GRID = [0.10, 0.20, 0.30, 0.45]    # deploy_noise (axis B)
REPR_GRID = [0.0, 0.05, 0.15, 0.30]       # collect_noise for the repr. ablation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _eval(policy, deploy_noise, seed, support=None, mode="greedy"):
    rng = np.random.default_rng(10_000 + seed)
    env = AgenticTroubleshootMDP(tool_noise=deploy_noise, rng=rng)
    return evaluate(policy, env, rng, n_episodes=EVAL_EPISODES, support=support,
                    mode=mode)


def _train_reference(teacher, seed):
    """Shared SFT reference trained on CLEAN teacher demos."""
    rng = np.random.default_rng(seed)
    demo_env = AgenticTroubleshootMDP(tool_noise=0.0, rng=rng)
    ref, _ = train_sft(teacher, demo_env, rng, n_episodes=SFT_EPISODES,
                       epochs=SFT_EPOCHS, clean_demos=True)
    return ref


def _train_reference_tabular(teacher, seed):
    """Per-state-table SFT reference on CLEAN teacher demos (repr. ablation)."""
    rng = np.random.default_rng(seed)
    demo_env = AgenticTroubleshootMDP(tool_noise=0.0, rng=rng)
    ref, _ = train_sft_tabular(teacher, demo_env, rng, n_episodes=SFT_EPISODES,
                               epochs=SFT_EPOCHS, clean_demos=True)
    return ref


def _row(method, seed, collect_noise, deploy_noise, metrics, representation="linear",
         sampled_success=None, env_steps=None, teacher_queries=None):
    return {
        "method": method,
        "seed": seed,
        "representation": representation,
        "collect_noise": collect_noise,
        "deploy_noise": deploy_noise,
        "success_rate": metrics["success_rate"],
        "avg_reward": metrics["avg_reward"],
        "avg_len": metrics["avg_len"],
        "off_support_state_ratio": metrics["off_support_state_ratio"],
        "sampled_success_rate": sampled_success,
        "train_env_steps": env_steps,
        "teacher_queries": teacher_queries,
    }


# ---------------------------------------------------------------------------
# Headline table: all methods at one operating point
# ---------------------------------------------------------------------------


def run_main():
    rows = []
    for seed in range(N_SEEDS):
        teacher = Teacher(temperature=TEACHER_TEMP)
        ref = _train_reference(teacher, seed)
        rng = np.random.default_rng(seed)
        coll = AgenticTroubleshootMDP(tool_noise=MAIN_COLLECT_NOISE, rng=rng)
        depl = AgenticTroubleshootMDP(tool_noise=MAIN_DEPLOY_NOISE, rng=rng)

        # A SHARED reference support set: the states the frozen pi_ref dataset
        # actually covers. Every no-env-access method is measured against the
        # SAME set so off_support_state_ratio is comparable across methods (the
        # quantity Assumption 3.2 is about), not method-specific.
        ref_data = collect_dataset(ref, coll, rng, OPD_EPISODES,
                                   explore_eps=MAIN_COLLECT_NOISE)
        ref_support = {rec["state"].key() for rec in ref_data}

        def add(name, pol, *, support=ref_support, env_steps=0, teacher_queries=None):
            """Eval greedy + sampled against the shared support; record both."""
            g = _eval(pol, MAIN_DEPLOY_NOISE, seed, support)
            s = _eval(pol, MAIN_DEPLOY_NOISE, seed, mode="sample")
            rows.append(_row(name, seed, MAIN_COLLECT_NOISE, MAIN_DEPLOY_NOISE, g,
                             sampled_success=s["success_rate"], env_steps=env_steps,
                             teacher_queries=teacher_queries))

        # teacher (skyline) + sft (floor) -- no training env access
        add("teacher", teacher, env_steps=0)
        add("sft", ref, env_steps=0)

        # offline OPD (Lightning) -- no env access in the training loop
        e0 = coll.n_env_steps
        pol, _ = train_offline_opd(ref, teacher, coll, rng,
                                   n_episodes=OPD_EPISODES, iters=OPD_ITERS)
        add("offline_opd", pol, env_steps=coll.n_env_steps - e0)

        # Rang et al. offline on-policy KD -- no env access in the training loop
        e0 = coll.n_env_steps
        pol, _ = train_rang_kd(ref, teacher, coll, rng,
                               n_episodes=OPD_EPISODES, iters=OPD_ITERS)
        add("rang_kd", pol, env_steps=coll.n_env_steps - e0)

        # patch 1: uncertainty-triggered teacher query (env-spending baseline)
        e0 = coll.n_env_steps
        pol, diag1 = train_offline_opd_active_query(ref, teacher, coll, rng,
                                                    n_episodes=OPD_EPISODES,
                                                    iters=OPD_ITERS)
        add("offline_opd_active_query", pol, env_steps=coll.n_env_steps - e0,
            teacher_queries=diag1.get("n_teacher_queries"))

        # patch 2: full-distribution advantage (no env access) -- in-contract fix
        e0 = coll.n_env_steps
        pol, _ = train_offline_opd_full_adv(ref, teacher, coll, rng,
                                            n_episodes=OPD_EPISODES, iters=OPD_ITERS)
        add("offline_opd_full_adv", pol, env_steps=coll.n_env_steps - e0)

        # patch 3: branch-aware replay (no env access) -- KL-reweighted full-dist
        e0 = coll.n_env_steps
        pol, _ = train_offline_opd_branch_replay(ref, teacher, coll, rng,
                                                 n_episodes=OPD_EPISODES,
                                                 iters=OPD_ITERS)
        add("offline_opd_branch_replay", pol, env_steps=coll.n_env_steps - e0)

        # online upper bounds -- full env access
        e0 = depl.n_env_steps
        pol, _ = train_online_opd(ref, teacher, depl, rng, iters=ONLINE_OPD_ITERS)
        add("online_opd", pol, env_steps=depl.n_env_steps - e0)

        e0 = depl.n_env_steps
        pol, _ = train_online_rl(depl, rng, iters=ONLINE_RL_ITERS)
        add("online_rl", pol, env_steps=depl.n_env_steps - e0)

        print(f"[main] seed {seed} done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Axis A: collection coverage
# ---------------------------------------------------------------------------


def run_coverage():
    rows = []
    for seed in range(N_SEEDS):
        teacher = Teacher(temperature=TEACHER_TEMP)
        ref = _train_reference(teacher, seed)
        for cn in COVERAGE_GRID:
            rng = np.random.default_rng(seed)
            coll = AgenticTroubleshootMDP(tool_noise=cn, rng=rng)
            pol, diag = train_offline_opd(ref, teacher, coll, rng,
                                          n_episodes=OPD_EPISODES, iters=OPD_ITERS)
            rows.append(_row("offline_opd", seed, cn, MAIN_DEPLOY_NOISE,
                             _eval(pol, MAIN_DEPLOY_NOISE, seed, diag["support"])))
            pol, _ = train_rang_kd(ref, teacher, coll, rng,
                                   n_episodes=OPD_EPISODES, iters=OPD_ITERS)
            rows.append(_row("rang_kd", seed, cn, MAIN_DEPLOY_NOISE,
                             _eval(pol, MAIN_DEPLOY_NOISE, seed)))
            pol, _ = train_offline_opd_full_adv(ref, teacher, coll, rng,
                                                n_episodes=OPD_EPISODES,
                                                iters=OPD_ITERS)
            rows.append(_row("offline_opd_full_adv", seed, cn,
                             MAIN_DEPLOY_NOISE, _eval(pol, MAIN_DEPLOY_NOISE, seed)))
            pol, _ = train_offline_opd_branch_replay(ref, teacher, coll, rng,
                                                     n_episodes=OPD_EPISODES,
                                                     iters=OPD_ITERS)
            rows.append(_row("offline_opd_branch_replay", seed, cn,
                             MAIN_DEPLOY_NOISE, _eval(pol, MAIN_DEPLOY_NOISE, seed)))
        print(f"[coverage] seed {seed} done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Axis B: deployment distribution shift
# ---------------------------------------------------------------------------


def run_deploy():
    rows = []
    for seed in range(N_SEEDS):
        teacher = Teacher(temperature=TEACHER_TEMP)
        ref = _train_reference(teacher, seed)
        rng = np.random.default_rng(seed)
        coll = AgenticTroubleshootMDP(tool_noise=MAIN_COLLECT_NOISE, rng=rng)
        # offline methods are trained ONCE (offline), then evaluated across
        # deployment noise levels -- the precomputed signal cannot adapt.
        pol_opd, diag = train_offline_opd(ref, teacher, coll, rng,
                                          n_episodes=OPD_EPISODES, iters=OPD_ITERS)
        pol_full, _ = train_offline_opd_full_adv(ref, teacher, coll, rng,
                                                  n_episodes=OPD_EPISODES,
                                                  iters=OPD_ITERS)
        pol_rang, _ = train_rang_kd(ref, teacher, coll, rng,
                                    n_episodes=OPD_EPISODES, iters=OPD_ITERS)
        for dn in DEPLOY_GRID:
            rows.append(_row("sft", seed, MAIN_COLLECT_NOISE, dn,
                             _eval(ref, dn, seed)))
            rows.append(_row("offline_opd", seed, MAIN_COLLECT_NOISE, dn,
                             _eval(pol_opd, dn, seed, diag["support"])))
            rows.append(_row("rang_kd", seed, MAIN_COLLECT_NOISE, dn,
                             _eval(pol_rang, dn, seed)))
            rows.append(_row("offline_opd_full_adv", seed,
                             MAIN_COLLECT_NOISE, dn, _eval(pol_full, dn, seed)))
            # online methods get to retrain at each deployment noise level
            depl = AgenticTroubleshootMDP(tool_noise=dn, rng=rng)
            pol, _ = train_online_opd(ref, teacher, depl, rng, iters=ONLINE_OPD_ITERS)
            rows.append(_row("online_opd", seed, MAIN_COLLECT_NOISE, dn,
                             _eval(pol, dn, seed)))
        print(f"[deploy] seed {seed} done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Representation ablation: generalization (linear) vs pure coverage (tabular)
# ---------------------------------------------------------------------------


def run_representation():
    """Cross the offline methods with the student REPRESENTATION over a coverage
    sweep, deploy fixed. The point: in the linear (shared-weight) student,
    distillation can rescue off-support behaviour via *parameter generalization*;
    in the tabular (per-state) student there is no generalization, so an
    off-support state receives no gradient and stays at the SFT floor. This
    isolates Lightning OPD's Assumption 3.2 to *pure support coverage*.
    """
    rows = []
    for seed in range(N_SEEDS):
        teacher = Teacher(temperature=TEACHER_TEMP)
        lin_ref = _train_reference(teacher, seed)
        tab_ref = _train_reference_tabular(teacher, seed)
        for cn in REPR_GRID:
            # --- linear (generalizing) student ---
            rng = np.random.default_rng(seed)
            coll = AgenticTroubleshootMDP(tool_noise=cn, rng=rng)
            rows.append(_row("sft", seed, cn, MAIN_DEPLOY_NOISE,
                             _eval(lin_ref, MAIN_DEPLOY_NOISE, seed), "linear"))
            pol, diag = train_offline_opd(lin_ref, teacher, coll, rng,
                                          n_episodes=OPD_EPISODES, iters=OPD_ITERS)
            rows.append(_row("offline_opd", seed, cn, MAIN_DEPLOY_NOISE,
                             _eval(pol, MAIN_DEPLOY_NOISE, seed, diag["support"]),
                             "linear"))
            pol, _ = train_rang_kd(lin_ref, teacher, coll, rng,
                                   n_episodes=OPD_EPISODES, iters=OPD_ITERS)
            rows.append(_row("rang_kd", seed, cn, MAIN_DEPLOY_NOISE,
                             _eval(pol, MAIN_DEPLOY_NOISE, seed), "linear"))
            depl = AgenticTroubleshootMDP(tool_noise=MAIN_DEPLOY_NOISE, rng=rng)
            pol, _ = train_online_opd(lin_ref, teacher, depl, rng,
                                      iters=ONLINE_OPD_ITERS)
            rows.append(_row("online_opd", seed, cn, MAIN_DEPLOY_NOISE,
                             _eval(pol, MAIN_DEPLOY_NOISE, seed), "linear"))

            # --- tabular (non-generalizing) student ---
            rng = np.random.default_rng(seed)
            coll = AgenticTroubleshootMDP(tool_noise=cn, rng=rng)
            rows.append(_row("sft", seed, cn, MAIN_DEPLOY_NOISE,
                             _eval(tab_ref, MAIN_DEPLOY_NOISE, seed), "tabular"))
            pol, diag = train_offline_opd_tabular(tab_ref, teacher, coll, rng,
                                                  n_episodes=OPD_EPISODES,
                                                  iters=OPD_ITERS)
            rows.append(_row("offline_opd", seed, cn, MAIN_DEPLOY_NOISE,
                             _eval(pol, MAIN_DEPLOY_NOISE, seed, diag["support"]),
                             "tabular"))
            pol, _ = train_rang_kd_tabular(tab_ref, teacher, coll, rng,
                                           n_episodes=OPD_EPISODES, iters=OPD_ITERS)
            rows.append(_row("rang_kd", seed, cn, MAIN_DEPLOY_NOISE,
                             _eval(pol, MAIN_DEPLOY_NOISE, seed), "tabular"))
            depl = AgenticTroubleshootMDP(tool_noise=MAIN_DEPLOY_NOISE, rng=rng)
            pol, _ = train_online_opd_tabular(tab_ref, teacher, depl, rng,
                                              iters=ONLINE_OPD_ITERS)
            rows.append(_row("online_opd", seed, cn, MAIN_DEPLOY_NOISE,
                             _eval(pol, MAIN_DEPLOY_NOISE, seed), "tabular"))
        print(f"[representation] seed {seed} done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Case studies
# ---------------------------------------------------------------------------


def run_case_studies():
    teacher = Teacher(temperature=TEACHER_TEMP)
    ref = _train_reference(teacher, 0)
    rng = np.random.default_rng(0)
    coll = AgenticTroubleshootMDP(tool_noise=0.0, rng=rng)
    deploy = AgenticTroubleshootMDP(tool_noise=0.30, rng=rng)

    pol_opd, _ = train_offline_opd(ref, teacher, coll, rng,
                                   n_episodes=OPD_EPISODES, iters=OPD_ITERS)
    pol_aq, _ = train_offline_opd_active_query(ref, teacher, coll, rng,
                                               n_episodes=OPD_EPISODES, iters=OPD_ITERS)
    pol_br, _ = train_offline_opd_branch_replay(ref, teacher, coll, rng,
                                                n_episodes=OPD_EPISODES, iters=OPD_ITERS)
    blocks = []
    # force a deployment episode that enters the error branch (noisy env)
    forced = np.random.default_rng(7)
    for name, pol in [("SFT", ref), ("offline_opd (zero-coverage)", pol_opd),
                      ("offline_opd_active_query (patch 1)", pol_aq),
                      ("offline_opd_branch_replay (patch 3)", pol_br),
                      ("teacher", teacher)]:
        env = AgenticTroubleshootMDP(tool_noise=0.30, rng=forced)
        blocks.append(f"===== {name} =====")
        blocks.extend(capture_case_studies(pol, env, forced, n=3))
        blocks.append("")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Patch 1 ablation: uncertainty-trigger threshold (env-spending baseline)
# ---------------------------------------------------------------------------


def run_active_query():
    """Sweep the confidence threshold of patch 1 (uncertainty-triggered teacher
    query). conf_thresh=0.0 -> plain Lightning OPD (no trigger); 1.0 -> full
    DAgger (every state). The interesting middle: DAgger-class success at a
    fraction of the live-teacher queries. Records greedy + sampled success,
    env-steps, and teacher_queries so the report can plot success vs query cost."""
    rows = []
    grid = [0.0, 0.5, 0.7, 0.9, 1.0]
    for seed in range(N_SEEDS):
        teacher = Teacher(temperature=TEACHER_TEMP)
        ref = _train_reference(teacher, seed)
        ref_data = collect_dataset(ref, AgenticTroubleshootMDP(
            tool_noise=MAIN_COLLECT_NOISE, rng=np.random.default_rng(seed)),
            np.random.default_rng(seed), OPD_EPISODES, explore_eps=MAIN_COLLECT_NOISE)
        support = {rec["state"].key() for rec in ref_data}
        for ct in grid:
            rng = np.random.default_rng(seed)
            coll = AgenticTroubleshootMDP(tool_noise=MAIN_COLLECT_NOISE, rng=rng)
            e0 = coll.n_env_steps
            pol, diag = train_offline_opd_active_query(
                ref, teacher, coll, rng, n_episodes=OPD_EPISODES,
                iters=OPD_ITERS, conf_thresh=ct)
            env_steps = coll.n_env_steps - e0
            g = _eval(pol, MAIN_DEPLOY_NOISE, seed, support)
            s = _eval(pol, MAIN_DEPLOY_NOISE, seed, mode="sample")
            rows.append(_row(f"active_query_ct{ct}", seed, MAIN_COLLECT_NOISE,
                             MAIN_DEPLOY_NOISE, g, sampled_success=s["success_rate"],
                             env_steps=env_steps,
                             teacher_queries=diag.get("n_teacher_queries")))
        print(f"[active_query] seed {seed} done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Patch 3 ablation: branch-replay reweighting strength gamma (no env access)
# ---------------------------------------------------------------------------


def run_patch3_ablation():
    """Sweep gamma in patch 3 (branch-aware replay). gamma=0 reproduces patch 2
    (full-distribution advantage) exactly. Larger gamma upweights rare
    error-branch / recovery records (alpha = 1 + gamma*KL(pi_T||pi_ref)). The
    thesis: greedy stays at 1.000 while sampled rises from ~0.864 toward ~0.99 --
    closing the mass gap with zero extra env access. Records greedy + sampled."""
    rows = []
    grid = [0.0, 1.0, 5.0, 20.0, 100.0]
    for seed in range(N_SEEDS):
        teacher = Teacher(temperature=TEACHER_TEMP)
        ref = _train_reference(teacher, seed)
        ref_data = collect_dataset(ref, AgenticTroubleshootMDP(
            tool_noise=MAIN_COLLECT_NOISE, rng=np.random.default_rng(seed)),
            np.random.default_rng(seed), OPD_EPISODES, explore_eps=MAIN_COLLECT_NOISE)
        support = {rec["state"].key() for rec in ref_data}
        for gamma in grid:
            rng = np.random.default_rng(seed)
            coll = AgenticTroubleshootMDP(tool_noise=MAIN_COLLECT_NOISE, rng=rng)
            e0 = coll.n_env_steps
            pol, _ = train_offline_opd_branch_replay(
                ref, teacher, coll, rng, n_episodes=OPD_EPISODES,
                iters=OPD_ITERS, gamma=gamma)
            env_steps = coll.n_env_steps - e0
            g = _eval(pol, MAIN_DEPLOY_NOISE, seed, support)
            s = _eval(pol, MAIN_DEPLOY_NOISE, seed, mode="sample")
            rows.append(_row(f"branch_replay_g{gamma}", seed, MAIN_COLLECT_NOISE,
                             MAIN_DEPLOY_NOISE, g, sampled_success=s["success_rate"],
                             env_steps=env_steps))
        print(f"[patch3] seed {seed} done")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Running main comparison ...")
    run_main().to_csv(os.path.join(RESULTS_DIR, "results_main.csv"), index=False)
    print("Running coverage ablation (axis A) ...")
    run_coverage().to_csv(os.path.join(RESULTS_DIR, "results_coverage.csv"),
                          index=False)
    print("Running deployment-shift ablation (axis B) ...")
    run_deploy().to_csv(os.path.join(RESULTS_DIR, "results_deploy.csv"),
                        index=False)
    print("Running representation ablation (linear vs tabular) ...")
    run_representation().to_csv(
        os.path.join(RESULTS_DIR, "results_representation.csv"), index=False)
    print("Running patch-1 active-query ablation ...")
    run_active_query().to_csv(os.path.join(RESULTS_DIR, "results_active.csv"),
                              index=False)
    print("Running patch-3 branch-replay ablation ...")
    run_patch3_ablation().to_csv(os.path.join(RESULTS_DIR, "results_patch3.csv"),
                                 index=False)
    print("Capturing case studies ...")
    with open(os.path.join(RESULTS_DIR, "case_studies.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(run_case_studies())
    print("Done. Results written to", os.path.abspath(RESULTS_DIR))


if __name__ == "__main__":
    main()
