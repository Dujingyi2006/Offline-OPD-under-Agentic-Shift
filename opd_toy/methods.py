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
* ``train_offline_opd_support_aware`` -- patch 1: down-weight the OPD update by
                              a per-state support score (visitation density).
* ``train_offline_opd_dagger``        -- patch 2: periodically refresh the
                              dataset with teacher-annotated student rollouts.
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
# 6. Support-aware Lightning OPD (patch 1)
# ---------------------------------------------------------------------------


def train_offline_opd_support_aware(ref_policy, teacher, env, rng, *,
                                    n_episodes=400, iters=600, lr=0.2,
                                    kappa=8.0, beta=1.0, beta0=0.3,
                                    explore_eps=0.0):
    """Patch 1: support-aware *conservative* offline OPD.

    Two coupled mechanisms, both keyed on a per-state support score
    ``w(s) = n(s) / (n(s) + kappa)`` (visitation density in the precomputed set):

      1. The OPD advantage update is scaled by ``w(s)`` -- thinly covered states
         emit weak updates, so the student is not driven by confident-but-
         unverifiable advantages where the offline approximation is unreliable.
      2. A conservative anchor pulls the student back toward the SFT reference
         with strength ``beta * (1 - w(s))``: on thinly covered states the update
         is dominated by an L2-style pull to pi_ref, which reduces variance and
         gives a do-no-harm floor at the SFT level.

    NOTE on scope: this anchor is NOT a fix for the missing recovery skill. It
    pulls the student toward pi_ref, and pi_ref (a clean-demo SFT policy) never
    learned ``ask_followup`` either -- so on a genuine error branch the anchor
    can only stabilise the student around the same non-recovering behaviour, not
    invent the recovery action. What it buys is variance reduction and a
    conservative floor on well-vs-poorly covered states; it cannot manufacture a
    signal that the sampled-action advantage never contained. (This is weaker
    than, and should not be conflated with, the *implicit* regularisation
    Lightning OPD derives in-distribution; that result is about staying near
    pi_ref where the teacher signal is dense, not about covering off-support
    error branches.)

    Together these target the *variance* side of the failure, not the *coverage*
    side -- contrast patch 3 (full-distribution advantage), which attacks the
    coverage side directly while keeping the same no-env-access contract.
    """
    data = collect_dataset(ref_policy, env, rng, n_episodes, teacher=teacher,
                           explore_eps=explore_eps)
    policy = ref_policy.copy()
    counts = Counter(rec["state"].key() for rec in data)
    support_w = {k: c / (c + kappa) for k, c in counts.items()}
    # precompute per-record feature matrix + frozen reference log-probs
    for rec in data:
        rec["_F"] = _features(rec["state"])
        rec["_w"] = support_w[rec["state"].key()]
        rec["_ref_logp"] = ref_policy.logprobs(rec["state"])
    for _ in range(iters):
        grad = np.zeros_like(policy.w)
        for rec in data:
            F, a, w = rec["_F"], rec["action"], rec["_w"]
            logits = F @ policy.w
            z = logits - logits.max()
            p = np.exp(z); p /= p.sum()
            lp = np.log(np.clip(p, 1e-12, 1.0))
            pF = p @ F
            # (1) density-scaled OPD push on the sampled action
            adv = float(np.clip(rec["teacher_logp"][a] - lp[a], -ADV_CLIP, ADV_CLIP))
            grad += w * adv * (F[a] - pF)
            # (2) conservative anchor toward pi_ref (grad of -KL(pi_theta||pi_ref)),
            #     vectorized: sum_a coeff[a]*(F[a]-pF) = coeff@F - coeff.sum()*pF
            anchor = beta0 + beta * (1.0 - w)
            coeff = p * (rec["_ref_logp"] - lp)
            grad += anchor * (coeff @ F - coeff.sum() * pF)
        policy.w += lr * grad / len(data)
    return policy, {"support_w": support_w}


# ---------------------------------------------------------------------------
# 7. DAgger-refresh Lightning OPD (patch 2)
# ---------------------------------------------------------------------------


def train_offline_opd_dagger(ref_policy, teacher, env, rng, *, n_episodes=400,
                             iters=600, lr=0.2, refresh_every=60,
                             refresh_episodes=60, explore_eps=0.0):
    """Patch 2. Periodically the current student rolls out a small batch in the
    env; the teacher annotates those fresh student-visited states and they are
    appended to the dataset. Minimal DAgger-style refresh: adds bounded online
    interaction so error-branch states the student actually reaches get covered.
    Trades a little 'no env access' purity for robustness to coverage gaps."""
    data = collect_dataset(ref_policy, env, rng, n_episodes, teacher=teacher,
                           explore_eps=explore_eps)
    for rec in data:
        rec["_F"] = _features(rec["state"])
    policy = ref_policy.copy()
    n_refresh = 0
    for it in range(iters):
        if refresh_every and it > 0 and it % refresh_every == 0:
            fresh = collect_dataset(policy, env, rng, refresh_episodes, teacher=teacher)
            for rec in fresh:
                rec["_F"] = _features(rec["state"])
            data = data + fresh
            n_refresh += len(fresh)
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
    return policy, {"n_refresh_steps": n_refresh}


# ---------------------------------------------------------------------------
# 8. Full-distribution-advantage Lightning OPD (patch 3)
# ---------------------------------------------------------------------------


def train_offline_opd_full_adv(ref_policy, teacher, env, rng, *, n_episodes=400,
                               iters=600, lr=0.2, explore_eps=0.0):
    """Patch 3: full-distribution advantage, the minimal in-contract repair.

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
