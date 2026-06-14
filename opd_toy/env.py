"""AgenticTroubleshootMDP: a small multi-turn agentic troubleshooting task.

The episode is a fault-diagnosis dialogue. A hidden ``fault_type`` must be
identified by gathering the *discriminative* piece of evidence and then
committing to a diagnosis. The dynamics are deliberately *branching*:

  * The expert / teacher follows a thin "golden path" -- gather the one tool
    observation that discriminates the true fault, then ``final_answer``.
  * Wrong early actions push the agent into *error branches*: querying the
    wrong source returns misleading evidence that biases later decisions, and a
    premature ``final_answer`` ends the episode in failure.

These error-recovery states are rarely (or never) visited by the reference
policy used to collect offline data, which is exactly where Lightning OPD's
Support-Coverage assumption (Assumption 3.2) is stressed.

No GPU / no large model: everything is tabular and runs in milliseconds.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------------

FAULTS = ("db", "config", "network")          # hidden fault types
N_FAULTS = len(FAULTS)

# Actions. The first three are *information-gathering* tools; ask_followup is a
# generic probe; the final_answer_* actions commit to a diagnosis and end the
# episode.
ACTIONS = (
    "inspect_log",     # 0  discriminates db vs others
    "query_db",        # 1  discriminates db; wrong source for config/network
    "check_config",    # 2  discriminates config
    "ask_followup",    # 3  cheap generic probe, weakly informative
    "final_db",        # 4  commit: fault == db
    "final_config",    # 5  commit: fault == config
    "final_network",   # 6  commit: fault == network
)
N_ACTIONS = len(ACTIONS)
FINAL_ACTIONS = {4: "db", 5: "config", 6: "network"}
GATHER_ACTIONS = (0, 1, 2, 3)

# The single tool whose observation cleanly discriminates each fault. This is
# what the golden path uses.
DISCRIMINATIVE_TOOL = {"db": 1, "config": 2, "network": 0}

# Reward shaping
R_CORRECT = 1.0
R_WRONG = -1.0
R_STEP = -0.02
R_PREMATURE = -0.5     # extra penalty for committing with no evidence gathered
MAX_STEPS = 6


# ---------------------------------------------------------------------------
# Belief / observation model
# ---------------------------------------------------------------------------

# Evidence levels stored per fault hypothesis in the agent's "belief" slots.
#   0 = unknown, 1 = weak-for, 2 = strong-for, 3 = strong-against (misleading)
# We keep the belief vector tiny (one slot per fault) so the state space stays
# enumerable and tabular methods are exact.

EV_UNKNOWN, EV_WEAK, EV_STRONG, EV_AGAINST = 0, 1, 2, 3
EV_LEVELS = (EV_UNKNOWN, EV_WEAK, EV_STRONG, EV_AGAINST)


@dataclass
class State:
    """Observable agent state (the hidden fault is *not* part of it)."""

    beliefs: tuple = (EV_UNKNOWN,) * N_FAULTS  # evidence per fault hypothesis
    step: int = 0
    misled: int = 0                            # 1 if an error branch was entered
    done: bool = False

    def key(self) -> tuple:
        return (self.beliefs, self.step, self.misled)


def all_state_keys() -> list:
    """Enumerate every reachable observable state key (for tabular tables)."""
    keys = []
    for beliefs in itertools.product(EV_LEVELS, repeat=N_FAULTS):
        for step in range(MAX_STEPS + 1):
            for misled in (0, 1):
                keys.append((beliefs, step, misled))
    return keys


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@dataclass
class AgenticTroubleshootMDP:
    """A branching multi-turn diagnosis MDP.

    Parameters
    ----------
    tool_noise:
        Probability that an information-gathering tool returns a *misleading*
        observation. Higher noise => precomputed offline signals go stale
        (ablation axis B).
    rng:
        numpy Generator for reproducibility.
    """

    tool_noise: float = 0.0
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    fault: str = field(default="db", init=False)
    n_env_steps: int = field(default=0, init=False)  # live env.step() call counter

    # -- episode lifecycle --------------------------------------------------

    def reset(self, fault: str | None = None) -> State:
        self.fault = fault if fault is not None else FAULTS[self.rng.integers(N_FAULTS)]
        return State()

    def _set_belief(self, beliefs: tuple, idx: int, level: int) -> tuple:
        b = list(beliefs)
        # keep the *strongest* signal; AGAINST overrides nothing but is recorded
        if level == EV_AGAINST:
            b[idx] = EV_AGAINST
        else:
            b[idx] = max(b[idx], level)
        return tuple(b)

    def step(self, state: State, action: int) -> tuple[State, float, bool]:
        """Apply ``action`` in ``state``; return (next_state, reward, done)."""
        if state.done:
            raise RuntimeError("step() called on a terminal state")

        self.n_env_steps += 1
        reward = R_STEP
        beliefs = state.beliefs
        misled = state.misled
        step = state.step + 1
        done = False

        if action in FINAL_ACTIONS:
            dx = FINAL_ACTIONS[action]
            # premature commit: no investigation at all (every belief unknown)
            gathered = any(b != EV_UNKNOWN for b in beliefs)
            if not gathered:
                reward += R_PREMATURE
            reward += R_CORRECT if dx == self.fault else R_WRONG
            done = True
            nxt = State(beliefs=beliefs, step=step, misled=misled, done=True)
            return nxt, reward, done

        # --- information-gathering action -------------------------------
        if action == 3:  # ask_followup: the RECOVERY action.
            # Returns weak (but correct) evidence for the true fault and, if a
            # prior reading had wrongly ruled the true fault out, clears that
            # false negative. Never noisy. This is the skill needed to escape an
            # error branch; the teacher demonstrates it, a drifting student must
            # learn it.
            idx = FAULTS.index(self.fault)
            b = list(beliefs)
            b[idx] = EV_WEAK if b[idx] == EV_AGAINST else max(b[idx], EV_WEAK)
            beliefs = tuple(b)
            if misled and all(x != EV_AGAINST for x in b):
                misled = 0
        else:
            disc_fault = [f for f, t in DISCRIMINATIVE_TOOL.items() if t == action][0]
            idx = FAULTS.index(disc_fault)
            if disc_fault == self.fault:
                # Correct discriminative tool. Normally confirms (STRONG), but
                # with prob tool_noise it returns a FALSE NEGATIVE (rules the
                # true fault out) -- the trap that opens an error branch.
                if self.rng.random() < self.tool_noise:
                    beliefs = self._set_belief(beliefs, idx, EV_AGAINST)
                    misled = 1
                else:
                    beliefs = self._set_belief(beliefs, idx, EV_STRONG)
            else:
                # Wrong source for this fault: correctly rules it out.
                beliefs = self._set_belief(beliefs, idx, EV_AGAINST)

        if step >= MAX_STEPS:
            # forced termination without a commit -> treated as wrong answer
            reward += R_WRONG
            done = True

        nxt = State(beliefs=beliefs, step=step, misled=misled, done=done)
        return nxt, reward, done

    # -- helpers ------------------------------------------------------------

    def optimal_final_action(self, state: State) -> int:
        """Greedy diagnosis implied by current beliefs (used by the teacher)."""
        scores = []
        for i, _f in enumerate(FAULTS):
            b = state.beliefs[i]
            s = {EV_UNKNOWN: 0.0, EV_WEAK: 1.0, EV_STRONG: 3.0, EV_AGAINST: -2.0}[b]
            scores.append(s)
        return 4 + int(np.argmax(scores))
