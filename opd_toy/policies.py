"""Policies: a feature-based linear-softmax student and a tabular optimal teacher.

Why a *feature-based* student (not a lookup table)?
--------------------------------------------------
A per-state lookup policy has no generalization: in a state it never saw during
training it would act uniformly at random, which would *accidentally* recover
~1/N of the time. Real language-model agents instead **generalize** their
on-distribution behaviour into novel states -- often wrongly. We reproduce this
with a linear-softmax policy over hand-crafted state-action features with shared
weights. The student that learns the golden path from clean demonstrations
carries those same weights into deployment-time error states it never trained
on, and there the weights misfire (it never learned the recovery action). This
is exactly the off-support failure Lightning OPD's Support-Coverage assumption
(Assumption 3.2) glosses over in multi-turn settings.

The teacher remains a tabular near-optimal policy: it only needs to supply
log pi_T(a|s) targets and demonstrations, and being tabular keeps it exact.
"""

from __future__ import annotations

import numpy as np

from env import (
    ACTIONS,
    DISCRIMINATIVE_TOOL,
    EV_AGAINST,
    EV_STRONG,
    EV_UNKNOWN,
    EV_WEAK,
    FAULTS,
    GATHER_ACTIONS,
    MAX_STEPS,
    N_ACTIONS,
    N_FAULTS,
    State,
    all_state_keys,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Feature map  phi(s) -> (N_ACTIONS, D)
# ---------------------------------------------------------------------------
# Weights are SHARED across the three gather tools and across the three commit
# actions (the per-action bias columns let them differentiate). This sharing is
# what produces generalization -- and mis-generalization into unseen states.

# action 0 inspect_log -> network, 1 query_db -> db, 2 check_config -> config
_TOOL_TARGET = {a: f for f, a in DISCRIMINATIVE_TOOL.items()}

# column layout
_C_BIAS = 0                 # [0:7] one-hot action bias
_C_GATHER = 7               # [7:12] gather block: unknown,against,strong,has_pos,stepfrac
_C_FOLLOW = 12              # [12:17] ask_followup: has_against,all_ruled,exhausted,no_pos,stepfrac
_C_FINAL = 17               # [17:23] final: strong,weak,against,others_against_frac,has_pos,stepfrac
FEATURE_DIM = 23


def state_features(state: State) -> np.ndarray:
    """Return the (N_ACTIONS, FEATURE_DIM) feature matrix for ``state``.

    Cached by observable state key: the state space is small and enumerable, so
    every distinct feature matrix is built at most once.
    """
    key = state.key()
    cached = _FEATURE_CACHE.get(key)
    if cached is not None:
        return cached
    F = _build_features(state)
    _FEATURE_CACHE[key] = F
    return F


_FEATURE_CACHE: dict = {}


def _build_features(state: State) -> np.ndarray:
    b = state.beliefs
    F = np.zeros((N_ACTIONS, FEATURE_DIM))

    n_against = sum(1 for x in b if x == EV_AGAINST)
    n_unknown = sum(1 for x in b if x == EV_UNKNOWN)
    n_strong = sum(1 for x in b if x == EV_STRONG)
    has_positive = any(x in (EV_WEAK, EV_STRONG) for x in b)
    step_frac = state.step / MAX_STEPS

    for a in range(N_ACTIONS):
        F[a, _C_BIAS + a] = 1.0

        if a in (0, 1, 2):  # gather tool with a fault target
            f = _TOOL_TARGET[a]
            i = FAULTS.index(f)
            F[a, _C_GATHER + 0] = 1.0 if b[i] == EV_UNKNOWN else 0.0
            F[a, _C_GATHER + 1] = 1.0 if b[i] == EV_AGAINST else 0.0
            F[a, _C_GATHER + 2] = 1.0 if b[i] == EV_STRONG else 0.0
            F[a, _C_GATHER + 3] = 1.0 if has_positive else 0.0
            F[a, _C_GATHER + 4] = step_frac

        elif a == 3:  # ask_followup (the recovery action)
            # The ONLY feature that is positive for ask_followup is the
            # "all hypotheses ruled out" indicator -- a state that occurs only
            # AFTER a false-negative reading, i.e. never on the clean golden
            # path. Its weight therefore receives gradient ONLY from data that
            # actually contains recovery states. On the golden path ask_followup
            # carries just its (shared) bias column, which SFT drives negative
            # because the teacher never uses it there. Result: a student trained
            # only on golden-path data cannot recover -- a genuine off-support
            # failure, not an artifact of random tie-breaking.
            F[a, _C_FOLLOW + 0] = 1.0 if n_against == N_FAULTS else 0.0
            F[a, _C_FOLLOW + 1] = step_frac

        else:  # final_db / final_config / final_network
            i = a - 4
            others_against = sum(1 for j, x in enumerate(b)
                                 if j != i and x == EV_AGAINST)
            F[a, _C_FINAL + 0] = 1.0 if b[i] == EV_STRONG else 0.0
            F[a, _C_FINAL + 1] = 1.0 if b[i] == EV_WEAK else 0.0
            F[a, _C_FINAL + 2] = 1.0 if b[i] == EV_AGAINST else 0.0
            F[a, _C_FINAL + 3] = others_against / max(N_FAULTS - 1, 1)
            F[a, _C_FINAL + 4] = 1.0 if has_positive else 0.0
            F[a, _C_FINAL + 5] = step_frac

    return F


class LinearPolicy:
    """Linear-softmax policy:  pi(a|s) = softmax_a( w . phi(s)[a] )."""

    def __init__(self, init: float = 0.0, seed: int | None = None):
        rng = np.random.default_rng(seed)
        if init == 0.0:
            self.w = np.zeros(FEATURE_DIM)
        else:
            self.w = rng.normal(0.0, init, size=FEATURE_DIM)

    def logits(self, state: State) -> np.ndarray:
        return state_features(state) @ self.w

    def probs(self, state: State) -> np.ndarray:
        return softmax(self.logits(state))

    def logprobs(self, state: State) -> np.ndarray:
        return np.log(np.clip(self.probs(state), 1e-12, 1.0))

    def sample(self, state: State, rng: np.random.Generator) -> int:
        return int(rng.choice(N_ACTIONS, p=self.probs(state)))

    def greedy(self, state: State) -> int:
        return int(np.argmax(self.logits(state)))

    def grad_logp(self, state: State, action: int) -> np.ndarray:
        """grad_w log pi(action|state) = phi[action] - E_{a~pi} phi[a]."""
        F = state_features(state)
        p = softmax(F @ self.w)
        return F[action] - p @ F

    def copy(self) -> "LinearPolicy":
        new = LinearPolicy.__new__(LinearPolicy)
        new.w = self.w.copy()
        return new


class TabularPolicy:
    """Per-state lookup softmax:  pi(a|s) = softmax_a( theta_s[a] ).

    Each observable state owns an INDEPENDENT logit vector. There is no shared
    parameter and no feature map, so there is **zero generalization**: a gradient
    computed at state s touches only theta_s. States absent from the training
    data are never updated -- they stay frozen at the initialization. Under
    greedy evaluation an all-zero (uninitialised) entry deterministically picks
    action 0, never the recovery action, so an off-support error branch fails by
    construction rather than by chance.

    This is the foil to ``LinearPolicy``: it isolates *pure Support-Coverage*
    (Assumption 3.2) from the parameter-generalization channel. Whatever offline
    signal we distill, if a state is off-support it receives no gradient at all.
    """

    def __init__(self, init: float = 0.0, seed: int | None = None):
        self.table: dict = {}
        self.init = init
        self._rng = np.random.default_rng(seed)

    def _logits_for(self, key: tuple) -> np.ndarray:
        v = self.table.get(key)
        if v is None:
            v = (np.zeros(N_ACTIONS) if self.init == 0.0
                 else self._rng.normal(0.0, self.init, size=N_ACTIONS))
            self.table[key] = v
        return v

    def logits(self, state: State) -> np.ndarray:
        return self._logits_for(state.key())

    def probs(self, state: State) -> np.ndarray:
        return softmax(self.logits(state))

    def logprobs(self, state: State) -> np.ndarray:
        return np.log(np.clip(self.probs(state), 1e-12, 1.0))

    def sample(self, state: State, rng: np.random.Generator) -> int:
        return int(rng.choice(N_ACTIONS, p=self.probs(state)))

    def greedy(self, state: State) -> int:
        return int(np.argmax(self.logits(state)))

    def copy(self) -> "TabularPolicy":
        new = TabularPolicy.__new__(TabularPolicy)
        new.table = {k: v.copy() for k, v in self.table.items()}
        new.init = self.init
        new._rng = np.random.default_rng()
        return new


# ---------------------------------------------------------------------------
# Teacher (tabular near-optimal)
# ---------------------------------------------------------------------------


def _teacher_action_scores(state: State) -> np.ndarray:
    """Near-optimal action preferences (reasoning + recovery)."""
    scores = np.full(N_ACTIONS, -4.0)
    b = state.beliefs
    strong = [i for i, x in enumerate(b) if x == EV_STRONG]
    weak = [i for i, x in enumerate(b) if x == EV_WEAK]
    against = [i for i, x in enumerate(b) if x == EV_AGAINST]
    unknown = [i for i, x in enumerate(b) if x == EV_UNKNOWN]

    if strong:                                   # confirmed -> commit
        scores[4 + strong[0]] = 6.0
        return scores
    if len(against) == N_FAULTS and not weak:    # all ruled out -> recover
        scores[3] = 6.0
        return scores
    if weak:                                     # recovered weak signal -> commit
        scores[4 + weak[0]] = 6.0
        return scores
    if unknown:                                  # keep investigating untested fault
        scores[DISCRIMINATIVE_TOOL[FAULTS[unknown[0]]]] = 5.0
        scores[3] = 0.5
        for a in GATHER_ACTIONS:
            if scores[a] < 0:
                scores[a] = 0.2
        return scores
    scores[3] = 5.0                              # nothing left to test -> recover
    return scores


class Teacher:
    """Fixed tabular near-optimal softmax teacher (temperature-controlled)."""

    def __init__(self, temperature: float = 0.25):
        self.temperature = temperature
        self.keys = all_state_keys()
        self.index = {k: i for i, k in enumerate(self.keys)}
        self._logits = np.zeros((len(self.keys), N_ACTIONS))
        for k in self.keys:
            beliefs, step, misled = k
            st = State(beliefs=beliefs, step=step, misled=misled)
            self._logits[self.index[k]] = _teacher_action_scores(st) / temperature

    def logits(self, state: State) -> np.ndarray:
        return self._logits[self.index[state.key()]]

    def probs(self, state: State) -> np.ndarray:
        return softmax(self.logits(state))

    def logprobs(self, state: State) -> np.ndarray:
        return np.log(np.clip(self.probs(state), 1e-12, 1.0))

    def sample(self, state: State, rng: np.random.Generator) -> int:
        return int(rng.choice(N_ACTIONS, p=self.probs(state)))

    def greedy(self, state: State) -> int:
        return int(np.argmax(self.logits(state)))


if __name__ == "__main__":
    t = Teacher()
    print("empty:", ACTIONS[t.greedy(State())])
    print("db-strong:", ACTIONS[t.greedy(State(beliefs=(EV_STRONG, 0, 0)))])
    print("all-against:", ACTIONS[t.greedy(State(beliefs=(EV_AGAINST,)*3, step=3))])
