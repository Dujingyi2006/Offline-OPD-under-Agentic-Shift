"""Training methods compared in the study.

All methods train the same feature-based linear-softmax student
(``policies.LinearPolicy``) on the same toy MDP, so differences in outcome come
from the *learning signal*, not from capacity. The teacher is a fixed tabular
near-optimal policy reused for both SFT-data generation and OPD targets, which
enforces Lightning OPD's teacher-consistency condition (so any failure is
attributable to support coverage / distribution shift, not teacher mismatch).

Methods
-------
* ``train_sft``            -- behaviour cloning on (clean) teacher trajectories.
* ``train_online_rl``      -- REINFORCE with a baseline (sees its own mistakes).
* ``train_offline_opd``    -- Lightning OPD: precompute teacher log-probs over
                              FROZEN pi_ref rollouts, then PG with advantage
                              A_t = log pi_T - log pi_theta (stop-grad, clipped).
* ``train_online_opd``     -- same advantage but rollouts refreshed from the
                              current student + live teacher (on-policy bound).
* ``train_rang_kd``        -- Rang et al. composite (1-lam)*CE + lam*KL(P||Q) on
                              frozen student rollouts (comparison paper's method).
* ``train_offline_opd_active_query``  -- patch 1 (env-spending baseline):
                              uncertainty-triggered teacher query (SafeDAgger);
                              queries teacher only at low-confidence states.
* ``train_offline_opd_full_adv``      -- patch 2 (no-env, minimal fix):
                              full-distribution advantage; recovers the mode.
* ``train_offline_opd_branch_replay`` -- patch 3 (no-env, mass recovery):
                              branch-aware replay; KL-reweights rare records to
                              close the greedy-vs-sampled gap.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from env import N_ACTIONS, AgenticTroubleshootMDP
from policies import LinearPolicy, Teacher

ADV_CLIP = 5.0  # advantage clip tau (Assumption 3.1 / Algorithm 1 line 13)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


def rollout(policy, env, rng, fault=None, explore_eps=0.0):
    """Run one episode under ``policy``; return per-step records + return.

    ``explore_eps`` mixes uniform-random actions into data *collection* only.
    It is the coverage knob (ablation axis A): higher epsilon makes the
    reference rollouts wander into error-branch / recovery states, widening the
    support of the offline dataset.
    """
    state = env.reset(fault=fault)
    steps, total = [], 0.0
    while not state.done:
        if explore_eps > 0.0 and rng.random() < explore_eps:
            a = int(rng.integers(N_ACTIONS))
        else:
            a = policy.sample(state, rng)
        nxt, r, done = env.step(state, a)
        steps.append({"state": state, "action": a, "reward": r})
        total += r
        state = nxt
    return steps, total, state


def collect_dataset(policy, env, rng, n_episodes, teacher=None, explore_eps=0.0):
    """Collect a fixed offline dataset; optionally annotate with teacher signal."""
    data = []
    for _ in range(n_episodes):
        steps, total, _ = rollout(policy, env, rng, explore_eps=explore_eps)
        for st in steps:
            rec = dict(st)
            if teacher is not None:
                rec["teacher_logp"] = teacher.logprobs(st["state"])
                rec["teacher_p"] = teacher.probs(st["state"])
            data.append(rec)
    return data


def state_key_set(data):
    return {rec["state"].key() for rec in data}


# ---------------------------------------------------------------------------
# 1. SFT (behaviour cloning on teacher trajectories)
# ---------------------------------------------------------------------------


def train_sft(teacher, env, rng, *, n_episodes=400, lr=0.3, epochs=300, init=0.0,
              clean_demos=True):
    """Max-likelihood imitation of teacher rollouts (Eq. 6 in Lightning OPD).

    With ``clean_demos=True`` demos are collected from a NOISE-FREE environment:
    the realistic agentic setting where expert demonstrations show the golden
    path and rarely contain error-branch / recovery states. The SFT reference
    therefore never learns to recover -- the gap on-policy methods must close.
    """
    demo_env = AgenticTroubleshootMDP(tool_noise=0.0, rng=rng) if clean_demos else env
    data = collect_dataset(teacher, demo_env, rng, n_episodes)
    policy = LinearPolicy(init=init)
    for _ in range(epochs):
        grad = np.zeros_like(policy.w)
        for rec in data:
            grad += policy.grad_logp(rec["state"], rec["action"])
        policy.w += lr * grad / len(data)
    return policy, {"n_demo_steps": len(data)}


# ---------------------------------------------------------------------------
# 2. Online RL (REINFORCE with baseline)
# ---------------------------------------------------------------------------


def train_online_rl(env, rng, *, iters=4000, batch=16, lr=0.2, gamma=1.0,
                    init=0.1, seed=0):
    """Tabular-free REINFORCE. The policy rolls out in the env and learns from
    its OWN trajectories, so it directly experiences error branches."""
    policy = LinearPolicy(init=init, seed=seed)
    baseline = 0.0
    for _ in range(iters):
        grad = np.zeros_like(policy.w)
        returns = []
        for _ in range(batch):
            steps, total, _ = rollout(policy, env, rng)
            G, rtg = 0.0, []
            for st in reversed(steps):
                G = st["reward"] + gamma * G
                rtg.append(G)
            rtg.reverse()
            returns.append(total)
            for st, g_t in zip(steps, rtg):
                grad += (g_t - baseline) * policy.grad_logp(st["state"], st["action"])
        policy.w += lr * grad / batch
        baseline = 0.9 * baseline + 0.1 * float(np.mean(returns))
    return policy, {"final_baseline": baseline}


# ---------------------------------------------------------------------------
# 3. Offline OPD (Lightning OPD)
# ---------------------------------------------------------------------------


def train_offline_opd(ref_policy, teacher, env, rng, *, n_episodes=400,
                      iters=600, lr=0.2, explore_eps=0.0):
    """Lightning OPD. Preprocess: sample rollouts ONCE from the frozen reference
    and store teacher log-probs. Train: J_off = E_{x~pi_ref}[sum_t A_t] with
    A_t = log pi_T(a|s) - log pi_theta(a|s), stop-grad + clipped. No env access."""
    data = collect_dataset(ref_policy, env, rng, n_episodes, teacher=teacher,
                           explore_eps=explore_eps)
    policy = ref_policy.copy()
    support = state_key_set(data)
    for rec in data:
        rec["_F"] = _features(rec["state"])
    for _ in range(iters):
        grad = np.zeros_like(policy.w)
        for rec in data:
            F, a = rec["_F"], rec["action"]
            logits = F @ policy.w
            z = logits - logits.max()
            p = np.exp(z); p /= p.sum()
            lp_a = np.log(max(p[a], 1e-12))
            adv = float(np.clip(rec["teacher_logp"][a] - lp_a, -ADV_CLIP, ADV_CLIP))
            grad += adv * (F[a] - p @ F)
        policy.w += lr * grad / len(data)
    return policy, {"support": support, "n_support_states": len(support)}


# ---------------------------------------------------------------------------
# 4. Online OPD (on-policy upper bound)
# ---------------------------------------------------------------------------


def train_online_opd(ref_policy, teacher, env, rng, *, iters=600, batch=24, lr=0.2):
    """On-policy distillation with a LIVE teacher: rollouts refreshed from the
    current student every step, teacher queried online. Same advantage as
    offline OPD; the on-policy upper bound."""
    policy = ref_policy.copy()
    for _ in range(iters):
        grad = np.zeros_like(policy.w)
        n = 0
        for _ in range(batch):
            steps, _, _ = rollout(policy, env, rng)
            for st in steps:
                s, a = st["state"], st["action"]
                adv = float(np.clip(teacher.logprobs(s)[a] - policy.logprobs(s)[a],
                                    -ADV_CLIP, ADV_CLIP))
                grad += adv * policy.grad_logp(s, a)
                n += 1
        policy.w += lr * grad / max(n, 1)
    return policy, {}


# ---------------------------------------------------------------------------
# 5. Rang et al. offline on-policy KD (composite CE + forward-KL)
# ---------------------------------------------------------------------------


def train_rang_kd(ref_policy, teacher, env, rng, *, n_episodes=400, iters=600,
                lr=0.2, lam=0.7, explore_eps=0.0):
    """Comparison paper (Rang et al.). Frozen student rollouts annotated with the
    teacher's full distribution; train L = (1-lam)*CE + lam*KL(P||Q), P=teacher,
    Q=student. CE target is the rollout action (student-generated label)."""
    data = collect_dataset(ref_policy, env, rng, n_episodes, teacher=teacher,
                           explore_eps=explore_eps)
    policy = ref_policy.copy()
    for rec in data:
        rec["_F"] = _features(rec["state"])
    for _ in range(iters):
        grad = np.zeros_like(policy.w)
        for rec in data:
            F, a = rec["_F"], rec["action"]
            logits = F @ policy.w
            z = logits - logits.max()
            q = np.exp(z); q /= q.sum()
            p = rec["teacher_p"]
            qF = q @ F
            # grad_w CE(a)    = phi[a] - E_q phi
            g_ce = F[a] - qF
            # grad_w KL(p||q) = E_q phi - E_p phi  (minimizing KL wrt student)
            g_kl = qF - p @ F
            grad += (1 - lam) * g_ce - lam * g_kl
        policy.w += lr * grad / len(data)
    return policy, {}


# ---------------------------------------------------------------------------
# 6. Uncertainty-triggered teacher query (patch 1 -- active DAgger / SafeDAgger)
# ---------------------------------------------------------------------------


def train_offline_opd_active_query(ref_policy, teacher, env, rng, *,
                                   n_episodes=400, iters=600, lr=0.2,
                                   refresh_every=60, refresh_episodes=60,
                                   conf_thresh=0.7):
    """Patch 1 (env-spending baseline): uncertainty-triggered teacher query.

    This is active imitation learning (SafeDAgger / active DAgger). The professor's
    literal hint. It periodically rolls the current student out in the env and
    appends teacher-annotated fresh states, but queries the teacher ONLY where the
    student is uncertain:

        max_a pi_theta(a|s) < conf_thresh.

    Confident states are skipped (no teacher query spent there). The trigger
    concentrates the live-teacher budget on the error branch -- the off-support
    states where the student is unsure -- instead of spending it uniformly.

    ``conf_thresh`` interpolates endpoints: ``0.0`` never triggers ->
    plain Lightning OPD; ``1.0`` always triggers -> full DAgger.

    SCOPE: this still spends env access and live-teacher queries. It is NOT a
    no-env fix -- it is the env-spending baseline that patch 3 (branch-aware
    replay) is meant to surpass without any env access.

    Diagnostics: ``n_refresh_steps`` (states appended) and
    ``n_teacher_queries`` (live teacher annotations spent).
    """
    data = collect_dataset(ref_policy, env, rng, n_episodes, teacher=teacher,
                           explore_eps=0.0)
    for rec in data:
        rec["_F"] = _features(rec["state"])
    policy = ref_policy.copy()
    n_refresh = 0
    n_queries = 0
    for it in range(iters):
        if refresh_every and it > 0 and it % refresh_every == 0:
            # roll the CURRENT student out in the env (env.step counted)
            for _ in range(refresh_episodes):
                steps, _, _ = rollout(policy, env, rng)
                for st in steps:
                    state = st["state"]
                    # uncertainty trigger: query the teacher only where the
                    # student's top action is below the confidence threshold.
                    conf = float(policy.probs(state).max())
                    if conf < conf_thresh:
                        n_queries += 1
                        rec = dict(st)
                        rec["teacher_logp"] = teacher.logprobs(state)
                        rec["teacher_p"] = teacher.probs(state)
                        rec["_F"] = _features(state)
                        data.append(rec)
                        n_refresh += 1
        grad = np.zeros_like(policy.w)
        for rec in data:
            F, a = rec["_F"], rec["action"]
            logits = F @ policy.w
            z = logits - logits.max()
            p = np.exp(z); p /= p.sum()
            lp_a = np.log(max(p[a], 1e-12))
            adv = float(np.clip(rec["teacher_logp"][a] - lp_a, -ADV_CLIP, ADV_CLIP))
            grad += adv * (F[a] - p @ F)
        policy.w += lr * grad / len(data)
    return policy, {"n_refresh_steps": n_refresh, "n_teacher_queries": n_queries}


# ---------------------------------------------------------------------------
# 7. Full-distribution-advantage Lightning OPD (patch 2)
# ---------------------------------------------------------------------------


def train_offline_opd_full_adv(ref_policy, teacher, env, rng, *, n_episodes=400,
                               iters=600, lr=0.2, explore_eps=0.0):
    """Patch 2: full-distribution advantage, the minimal in-contract repair.

    Lightning OPD's gradient at a state uses the SINGLE sampled action ``a``:

        A_t = log pi_T(a|s) - log pi_theta(a|s)   ->   A_t * grad log pi(a|s)

    On a golden-path rollout that action is never ``ask_followup``, so the
    recovery mass is literally absent from the gradient even where the teacher
    target carries it. This patch keeps everything else identical -- still NO
    env access, still NO live teacher, still the advantage-PG form -- and changes
    ONE thing: it takes the expectation of the per-action advantage under the
    teacher's full distribution (already precomputed in ``teacher_p``):

        sum_a pi_T(a|s) * clip(log pi_T(a|s) - log pi_theta(a|s)) * grad log pi(a|s)

    This isolates the single variable separating Lightning OPD from Rang et al.
    -- *sampled-action* vs *full-distribution* target -- without spending the
    env access that DAgger (patch 2) quietly reintroduces. If this lifts the
    score toward Rang's, the gap was the sampled-action restriction, not the
    no-env-access contract.
    """
    data = collect_dataset(ref_policy, env, rng, n_episodes, teacher=teacher,
                           explore_eps=explore_eps)
    policy = ref_policy.copy()
    support = state_key_set(data)
    for rec in data:
        rec["_F"] = _features(rec["state"])
    for _ in range(iters):
        grad = np.zeros_like(policy.w)
        for rec in data:
            F = rec["_F"]
            tp = rec["teacher_p"]
            tlp = rec["teacher_logp"]
            logits = F @ policy.w
            z = logits - logits.max()
            p = np.exp(z); p /= p.sum()
            lp = np.log(np.clip(p, 1e-12, 1.0))
            pF = p @ F
            # per-action advantage, clipped exactly like the sampled version
            adv = np.clip(tlp - lp, -ADV_CLIP, ADV_CLIP)          # (N_ACTIONS,)
            # expectation under the teacher distribution of adv * grad log pi(a)
            #   grad log pi(a) = F[a] - pF  ->  sum_a tp[a]*adv[a]*(F[a]-pF)
            coeff = tp * adv                                       # (N_ACTIONS,)
            grad += coeff @ F - coeff.sum() * pF
        policy.w += lr * grad / len(data)
    return policy, {"support": support, "n_support_states": len(support)}


# ---------------------------------------------------------------------------
# 8. Branch-aware replay (patch 3 -- no-env, KL-reweighted full-dist)
# ---------------------------------------------------------------------------


def train_offline_opd_branch_replay(ref_policy, teacher, env, rng, *,
                                    n_episodes=400, iters=600, lr=0.2,
                                    gamma=5.0, explore_eps=0.0):
    """Patch 3: branch-aware replay.

    Diagnosis: patch 2 (full-distribution advantage) hits greedy 1.000 but
    only sampled ~0.864.  The reason is frequency dilution: the offline dataset
    has ~32 rare error-branch / recovery records vs ~600 golden-path records.
    In the equally-weighted gradient sum the on-path majority drowns out the
    recovery minority.  The student therefore puts only thin mass on
    ``ask_followup`` even though it is the argmax -- hence the greedy/sampled
    gap.  Online OPD reaches sampled ~0.990 on the same linear features because
    it visits the error states repeatedly and reinforces recovery to dominance.

    Fix: reweight each record by

        alpha(s) = 1 + gamma * KL(pi_T(.|s) || pi_ref(.|s))

    High at error-branch states -- where teacher and pi_ref maximally disagree
    (KL ~ tens of nats, pi_T puts ~1.0 on recovery, pi_ref puts ~0) -- low on
    the golden path where they agree (KL ~ 0).  All quantities are already
    precomputed offline in the frozen dataset (teacher_p, ref logprobs).  Zero
    extra env interaction, zero extra teacher query.

    The gradient is patch 2's full-distribution advantage, scaled per record:

        alpha(s) * sum_a pi_T(a|s) * clip(log pi_T - log pi_theta) * grad log pi(a|s)

    gamma=0 reproduces patch 2 exactly.  Expected: greedy stays at 1.000,
    sampled rises from 0.864 toward 0.99.
    """
    data = collect_dataset(ref_policy, env, rng, n_episodes, teacher=teacher,
                           explore_eps=explore_eps)
    policy = ref_policy.copy()
    support = state_key_set(data)
    for rec in data:
        rec["_F"] = _features(rec["state"])
        # KL(pi_T || pi_ref) at this state, clamped to [0, inf)
        tp = rec["teacher_p"]                            # (N_ACTIONS,)
        ref_lp = ref_policy.logprobs(rec["state"])       # (N_ACTIONS,)
        tlp = rec["teacher_logp"]                        # (N_ACTIONS,)
        # KL = sum_a pi_T(a) * (log pi_T(a) - log pi_ref(a)), skip zero-mass actions
        kl = float(np.sum(tp * np.clip(tlp - ref_lp, -50.0, 50.0) *
                          (tp > 1e-30)))
        rec["_alpha"] = 1.0 + gamma * max(kl, 0.0)
    for _ in range(iters):
        grad = np.zeros_like(policy.w)
        for rec in data:
            F = rec["_F"]
            tp = rec["teacher_p"]
            tlp = rec["teacher_logp"]
            alpha = rec["_alpha"]
            logits = F @ policy.w
            z = logits - logits.max()
            p = np.exp(z); p /= p.sum()
            lp = np.log(np.clip(p, 1e-12, 1.0))
            pF = p @ F
            adv = np.clip(tlp - lp, -ADV_CLIP, ADV_CLIP)   # (N_ACTIONS,)
            coeff = tp * adv                                  # (N_ACTIONS,)
            grad += alpha * (coeff @ F - coeff.sum() * pF)
        policy.w += lr * grad / len(data)
    return policy, {"support": support, "n_support_states": len(support)}


# import here to avoid a cycle at module top (features live in policies)
from policies import state_features as _features  # noqa: E402
from policies import TabularPolicy  # noqa: E402


# ===========================================================================
# Tabular variants (representation ablation)
# ===========================================================================
# These mirror sft / offline_opd / rang_kd / online_opd but train a per-state
# lookup policy (policies.TabularPolicy) instead of the shared-weight linear
# student. With no shared parameters there is NO generalization: an off-support
# state never receives a gradient and keeps its (uninformative) initial logits.
# This isolates pure Support-Coverage (Assumption 3.2) from parameter
# generalization. The gradient of log pi(a|s) wrt the state's own logits is
# simply (e_a - p): it touches only state s.


def _softmax_row(theta):
    z = theta - theta.max()
    e = np.exp(z)
    return e / e.sum()


def train_sft_tabular(teacher, env, rng, *, n_episodes=400, lr=0.5, epochs=300,
                      init=0.0, clean_demos=True):
    """Behaviour cloning into a per-state table. Only states that appear in the
    clean demos get logits; every other state stays at init (-> greedy action 0,
    never recovery). The tabular analogue of train_sft."""
    demo_env = AgenticTroubleshootMDP(tool_noise=0.0, rng=rng) if clean_demos else env
    data = collect_dataset(teacher, demo_env, rng, n_episodes)
    policy = TabularPolicy(init=init)
    # group records by state so each table entry is fit independently
    by_state: dict = {}
    for rec in data:
        by_state.setdefault(rec["state"].key(), []).append(rec["action"])
    for key, actions in by_state.items():
        theta = policy._logits_for(key)
        target = np.bincount(actions, minlength=N_ACTIONS) / len(actions)
        for _ in range(epochs):
            theta += lr * (target - _softmax_row(theta))
    return policy, {"n_demo_states": len(by_state)}


def train_offline_opd_tabular(ref_policy, teacher, env, rng, *, n_episodes=400,
                              iters=600, lr=0.5, explore_eps=0.0):
    """Lightning OPD on a per-state table. Precompute teacher log-probs over
    FROZEN pi_ref rollouts; advantage A_t = log pi_T - log pi_theta updates only
    the table entries the dataset covers. Off-support error-branch states get no
    update -> the recovery skill is never installed there."""
    data = collect_dataset(ref_policy, env, rng, n_episodes, teacher=teacher,
                           explore_eps=explore_eps)
    policy = ref_policy.copy()
    support = state_key_set(data)
    # aggregate per (state, action) the teacher target; train per state
    by_state: dict = {}
    for rec in data:
        by_state.setdefault(rec["state"].key(), []).append(rec)
    for _ in range(iters):
        for key, recs in by_state.items():
            theta = policy._logits_for(key)
            p = _softmax_row(theta)
            grad = np.zeros(N_ACTIONS)
            for rec in recs:
                a = rec["action"]
                lp_a = np.log(max(p[a], 1e-12))
                adv = float(np.clip(rec["teacher_logp"][a] - lp_a,
                                    -ADV_CLIP, ADV_CLIP))
                g = -p.copy(); g[a] += 1.0          # grad log pi(a|.) = e_a - p
                grad += adv * g
            theta += lr * grad / len(recs)
    return policy, {"support": support, "n_support_states": len(support)}


def train_rang_kd_tabular(ref_policy, teacher, env, rng, *, n_episodes=400,
                          iters=600, lr=0.5, lam=0.7, explore_eps=0.0):
    """Rang et al. composite (1-lam)*CE + lam*KL(P||Q) on a per-state table. Even
    the full-distribution target only reaches states present in the frozen
    rollouts: with no shared weights it CANNOT transfer recovery mass into
    off-support states. This is the key contrast with the linear setting, where
    rang_kd recovered via generalization."""
    data = collect_dataset(ref_policy, env, rng, n_episodes, teacher=teacher,
                           explore_eps=explore_eps)
    policy = ref_policy.copy()
    by_state: dict = {}
    for rec in data:
        by_state.setdefault(rec["state"].key(), []).append(rec)
    for _ in range(iters):
        for key, recs in by_state.items():
            theta = policy._logits_for(key)
            q = _softmax_row(theta)
            grad = np.zeros(N_ACTIONS)
            for rec in recs:
                a = rec["action"]
                p = rec["teacher_p"]
                g_ce = -q.copy(); g_ce[a] += 1.0    # grad CE(a) = e_a - q
                g_kl = q - p                         # grad KL(p||q) wrt theta = q - p
                grad += (1 - lam) * g_ce - lam * g_kl
            theta += lr * grad / len(recs)
    return policy, {}


def train_online_opd_tabular(ref_policy, teacher, env, rng, *, iters=600,
                             batch=24, lr=0.5):
    """On-policy distillation into a table with a LIVE teacher. Rollouts are
    refreshed from the current student, so error-branch states the student
    actually reaches DO get visited and updated -- the upper bound that shows
    the tabular collapse is about coverage, not representation."""
    policy = ref_policy.copy()
    for _ in range(iters):
        batch_recs: dict = {}
        for _ in range(batch):
            steps, _, _ = rollout(policy, env, rng)
            for st in steps:
                batch_recs.setdefault(st["state"].key(),
                                      []).append((st["state"], st["action"]))
        for key, items in batch_recs.items():
            theta = policy._logits_for(key)
            p = _softmax_row(theta)
            grad = np.zeros(N_ACTIONS)
            for state, a in items:
                lp_a = np.log(max(p[a], 1e-12))
                adv = float(np.clip(teacher.logprobs(state)[a] - lp_a,
                                    -ADV_CLIP, ADV_CLIP))
                g = -p.copy(); g[a] += 1.0
                grad += adv * g
            theta += lr * grad / len(items)
    return policy, {}
