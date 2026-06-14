"""Evaluation: roll out a trained policy and compute the reported metrics.

Metrics
-------
* success_rate            -- fraction of episodes ending in a correct diagnosis.
* avg_reward              -- mean episode return.
* avg_len                 -- mean number of steps per episode.
* off_support_state_ratio -- fraction of *visited* states that are absent from a
                             reference support set (the states present in the
                             offline dataset). This is the operational measure
                             of how often a policy leaves the region where the
                             offline OPD signal was valid.

We also capture a few full trajectories for qualitative case studies.
"""

from __future__ import annotations

import numpy as np

from env import ACTIONS, FAULTS, AgenticTroubleshootMDP


def _run_episode(policy, env, rng, fault, *, mode="greedy"):
    """Roll one episode. ``mode='greedy'`` takes argmax actions (the original
    metric); ``mode='sample'`` samples from the policy, which exposes how much of
    a method's success depends on argmax alignment vs the full action
    distribution -- a full-distribution KD that puts almost all greedy mass on
    the right action but only a thin tail on the recovery action will score lower
    under sampling. This is the train/eval-objective check the report needs."""
    state = env.reset(fault=fault)
    traj = []
    total = 0.0
    while not state.done:
        if mode == "sample":
            a = policy.sample(state, rng)
        else:
            a = policy.greedy(state)
        nxt, r, done = env.step(state, a)
        traj.append((state.key(), a, r))
        total += r
        state = nxt
    # success = last action was the correct final_* commit
    last_a = traj[-1][1]
    success = last_a in (4, 5, 6) and FAULTS[last_a - 4] == fault
    return traj, total, success


def _greedy_episode(policy, env, rng, fault):
    return _run_episode(policy, env, rng, fault, mode="greedy")


def evaluate(policy, env, rng, *, n_episodes=600, support=None, mode="greedy"):
    successes, returns, lengths = [], [], []
    visited, off_support = 0, 0
    for _ in range(n_episodes):
        fault = FAULTS[rng.integers(len(FAULTS))]
        traj, total, success = _run_episode(policy, env, rng, fault, mode=mode)
        successes.append(success)
        returns.append(total)
        lengths.append(len(traj))
        if support is not None:
            for (skey, _a, _r) in traj:
                visited += 1
                if skey not in support:
                    off_support += 1
    metrics = {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(returns)),
        "avg_len": float(np.mean(lengths)),
    }
    if support is not None:
        metrics["off_support_state_ratio"] = off_support / max(visited, 1)
    else:
        metrics["off_support_state_ratio"] = float("nan")
    return metrics


def format_trajectory(traj, fault):
    """Human-readable case study of one greedy episode."""
    lines = [f"  hidden_fault = {fault}"]
    for (skey, a, r) in traj:
        beliefs, step, misled = skey
        lines.append(
            f"    step {step}: beliefs={beliefs} misled={misled} "
            f"-> {ACTIONS[a]:<12} (r={r:+.2f})"
        )
    return "\n".join(lines)


def capture_case_studies(policy, env, rng, n=3):
    """Return a few formatted greedy trajectories, one per fault when possible."""
    studies = []
    for fault in FAULTS[:n]:
        traj, total, success = _greedy_episode(policy, env, rng, fault)
        header = f"[{'SUCCESS' if success else 'FAIL'}] return={total:+.2f}"
        studies.append(header + "\n" + format_trajectory(traj, fault))
    return studies
