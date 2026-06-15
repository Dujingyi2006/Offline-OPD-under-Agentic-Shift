# Offline On-Policy Distillation breaks under multi-turn distribution shift

**A controlled, GPU-free study.** All numbers below come from `run_experiments.py`
(5 seeds, 1500 eval episodes per point); figures from `plots.py`. Raw tables live
in [results/](../results/).

---

## 1. What we test, and the claim we attack

**Lightning OPD** (`t2.pdf`) makes on-policy distillation cheap by precomputing
the teacher's log-probs over rollouts sampled *once* from a frozen reference
policy `π_ref`, then training the student with the advantage-weighted gradient

```
A_t = log π_T(a_t | s_t) − log π_θ(a_t | s_t)   (stop-grad, clipped)
```

reusing that frozen signal — no teacher and no environment in the training loop.

**The claim we attack — the paper's own central argument, not its admitted open
problem.** It is easy (and weak) to attack Assumption 3.2 (Support Coverage): the
authors themselves flag multi-turn agents as an open case in Appendix E, so beating
on coverage is beating on a concession. We aim at the *load-bearing* claim instead.
The paper's theory says the offline approximation is sound **whenever the SFT and
OPD teachers are the same model** — its *teacher-consistency* condition. Formally:

- **Theorem 3.5** bounds the online–offline gap by drift alone,
  `‖∇J_on − ∇J_off‖ ≤ G·σ_A·√χ²(π_θ‖π_ref)`, with `χ² = 0` (the gap is *exactly*
  zero) at initialization `π_θ = π_ref`.
- **Theorem 3.6** gives a shared fixed point when the teacher is representable
  (`π_T ∈ Π_Θ`).
- The paper then asserts (§3, just after Thm 3.6) that when the teacher is *not*
  representable the offline update still "stays close to the online one as long as
  drift remains small, **suggesting comparable capacity-limited behavior in
  practice**."

Read together, this is the暴论 we target: **given teacher consistency, the residual
online–offline gap is a *capacity* (representability) story — add capacity and the
offline method tracks the online one. Coverage never enters.** Our experiment is
engineered to make that claim falsifiable on its own terms:

1. **Teacher consistency holds by construction.** A single tabular oracle teacher
   generates the SFT demonstrations *and* supplies every `log π_T` target — so the
   mismatch variance is `σ_Δ² = 0` exactly. We are inside the regime where the
   paper's soundness guarantee is supposed to bite. (This neutralizes the obvious
   referee escape, "you just violated consistency.")
2. **Capacity is not the bottleneck.** The recovery action *is* representable in the
   student's hypothesis class: the **same linear student**, fed the **full**
   teacher distribution (patch 3, §6.3), reaches **1.000** at zero extra env cost.
   The capacity to recover is there; the sampled-action signal simply never points
   at it.

With consistency satisfied *and* capacity sufficient, the paper's argument predicts
near-online behavior. Instead Lightning OPD lands **at/below the SFT floor** (§3).
The gap that survives is therefore neither a consistency bias nor a capacity limit —
it is a **coverage-of-the-distilled-signal** failure that the Thm 3.5 bound does not
forbid, because off-support the drift `χ²(π_θ‖π_ref)` *diverges* and the bound goes
vacuous. We are not refuting Theorem 3.5; we are refuting the informal extrapolation
that the surviving gap is "capacity-limited." The mechanism is multi-turn
**error compounding**: one bad tool call lands the agent in states `π_ref` never
visits, where (a) the precomputed teacher advantage is undefined on the *sampled*
action and (b) the recovery skill was never on the golden path to sample.

We contrast Lightning OPD against the offline top-k KD of **Rang et al.**
(openPangu Embedded-1B, the comparison paper), which distills the teacher's full
next-token distribution rather than just the sampled-action advantage.

## 2. Task — `AgenticTroubleshootMDP`

A fault-diagnosis episode. The expert walks a thin **golden path**: gather the
discriminative tool result, then commit to the matching fix. Under deployment
tool-noise a discriminative tool can return a **false negative** that rules out
the true fault, pushing the agent onto an **error branch**. The only escape is
`ask_followup` — a recovery action that *never appears on the clean golden path*,
so it is absent from any frozen-`π_ref` dataset.

The student is a **feature-based linear-softmax policy** with shared weights, so
it *generalizes* its on-path behaviour into unseen states the way a real model
does — and there it misfires deterministically, rather than escaping by chance.

## 3. Headline result

Fixed operating point: `collect_noise = 0.05`, `deploy_noise = 0.30`. Success rate,
mean ± std over 5 seeds ([results_main.csv](../results/results_main.csv),
[fig_main.png](../results/fig_main.png)):

| method | success | note |
|---|---|---|
| teacher | 1.000 ± .000 | oracle |
| online RL | 1.000 ± .000 | full env access (~155k env steps) |
| online OPD | 1.000 ± .000 | live teacher — the OPD upper bound (~30k env steps) |
| **Rang et al. offline KD** | **1.000 ± .000** | full-distribution target, no env (see §6.2) |
| **offline OPD + full-dist advantage (patch 3)** | **1.000 ± .000** | no env access — the minimal in-contract fix (see §6.3) |
| offline OPD + DAgger refresh (patch 2) | 0.919 ± .111 | ⚠ uses env access — really online (see §6.1) |
| offline OPD + support-aware (patch 1) | 0.747 ± .168 | no env access |
| SFT | 0.699 ± .015 | imitation baseline |
| **Lightning OPD** | **0.632 ± .301** | no env access |

**Lightning OPD lands at the SFT baseline and below** (0.632 vs 0.699) and is by far
the most *unstable* method in the table (std 0.301 — an order of magnitude above
SFT's 0.015). It buys nothing over plain imitation in this regime, and the large
variance is the signature of the failure: whether a seed's frozen dataset happened
to brush the error branch decides the run. Its `off_support_state_ratio` averages
**0.147** — ~15% of states the deployed student visits were never in the frozen
dataset — against **0.000** for SFT measured on the same support set.

The striking contrast is the **three methods at 1.000 from the same frozen,
no-env-access budget**: Rang's full-distribution KD, and — decisively — *Lightning
OPD's own gradient with one line changed* (patch 3, §6.3). Taking the advantage
expectation under the teacher's **full** distribution instead of the single sampled
action transfers the recovery action's probability mass into states the student
reaches by generalization; distilling only the sampled-action advantage does not,
because the golden-path demonstrations never sample `ask_followup`.

## 4. Why — the mechanism, from the case studies

[case_studies.txt](../results/case_studies.txt) shows a forced error-branch
episode (true fault `db`, a discriminative tool misfires at step 1):

- **Lightning OPD** — after the false negative it thrashes
  `inspect_log → check_config → query_db → query_db → query_db` and times out
  at **−1.12**. Across the headline runs its `off_support_state_ratio` averages
  **0.147**: ~15% of visited states were never in the frozen dataset, and
  behaviour there is undefined.
- **Teacher** — hits the same false negative, then plays `ask_followup` at the
  all-against state `beliefs=(3,3,3) misled=1`, recovers the belief, and commits
  `final_db` for **+0.98**. The recovery skill exists; OPD just never saw it.
- **DAgger refresh (patch 2)** — recovers the easy faults but still fails one
  branch (**−1.10**): bounded refresh narrows the coverage gap without closing it.

This is a coverage failure, not an optimization failure. The advantage signal
`log π_T − log π_θ` is only defined where the frozen rollouts went; the error
branch is off-support **by construction**, and Lightning OPD has no term that
pulls the student toward the teacher there.

## 5. Ablations

### Axis A — collection coverage (`collect_noise`, deploy fixed)
[results_coverage.csv](../results/results_coverage.csv) ·
[fig_coverage.png](../results/fig_coverage.png)

| collect_noise | offline_opd | support-aware | full-dist adv | DAgger | rang_kd |
|---|---|---|---|---|---|
| 0.00 | 0.673 | 0.683 | 0.699 | 0.780 | 0.699 |
| 0.05 | 0.694 | 0.780 | 1.000 | 0.941 | 1.000 |
| 0.15 | 0.601 | 0.699 | 1.000 | 0.919 | 1.000 |
| 0.30 | 0.732 | 0.699 | 1.000 | 0.960 | 1.000 |

Plain Lightning OPD is **flat across coverage** (~0.6–0.73): broadening the frozen
dataset does not help, because the advantage gradient still ignores the actions
the reference never sampled. DAgger climbs with coverage (it injects
student-visited states). **Both full-distribution methods (full-dist adv, rang_kd)
sit at the 0.699 SFT floor at `collect_noise = 0` and jump to 1.000 the moment any
collection noise surfaces the error states** — they only need the recovery states
to *appear* in the data, not to be acted out. That the in-contract patch 3
(full-dist adv) tracks Rang cell-for-cell is the cleanest evidence that *what you
distill* (full distribution vs sampled action), not *who distills it*, is the
operative variable.

### Axis B — deployment shift (`deploy_noise`)
[results_deploy.csv](../results/results_deploy.csv) ·
[fig_deploy.png](../results/fig_deploy.png)

| deploy_noise | offline_opd | support-aware | rang_kd | online_opd | sft |
|---|---|---|---|---|---|
| 0.10 | 0.804 | 0.927 | 1.000 | 1.000 | 0.900 |
| 0.20 | 0.751 | 0.847 | 1.000 | 1.000 | 0.795 |
| 0.30 | 0.694 | 0.778 | 1.000 | 1.000 | 0.699 |
| 0.45 | 0.608 | 0.674 | 1.000 | 1.000 | 0.554 |

Lightning OPD **degrades monotonically** as the error branch is triggered more
often, tracking SFT closely — confirming it inherits imitation's brittleness
under shift. The support-aware patch keeps a consistent margin over it. The full
distribution methods (rang_kd, online_opd) are flat at 1.000: shift only matters
if the recovery skill was never distilled.

### Axis C — student representation: generalization vs pure coverage
[results_representation.csv](../results/results_representation.csv) ·
[fig_representation.png](../results/fig_representation.png)

There are two distinct ways an offline method can cover an error branch it never
acted out: **parameter generalization** (a shared-weight student interpolates
on-path behaviour into unseen states) and **literal support** (a per-state table
only knows states present in its data). To separate them we re-run the offline
methods with a **tabular per-state student** (`TabularPolicy`) whose gradient at
state *s* touches only *s*'s own logits — off-support states get **zero gradient**
and stay frozen at their SFT value, by construction.

Success rate at deploy 0.30, over the collection-coverage sweep:

| | collect 0.00 | 0.05 | 0.15 | 0.30 |
|---|---|---|---|---|
| **linear** SFT | 0.699 | 0.699 | 0.699 | 0.699 |
| **linear** Lightning OPD | 0.673 | 0.694 | 0.601 | 0.732 |
| **linear** Rang KD | 0.699 | **1.000** | **1.000** | **1.000** |
| **tabular** SFT | 0.697 | 0.697 | 0.697 | 0.697 |
| **tabular** Lightning OPD | 0.697 | 0.725 | 0.719 | 0.758 |
| **tabular** Rang KD | 0.697 | 0.780 | 0.903 | 0.919 |
| online OPD (both) | 1.000 | 1.000 | 1.000 | 1.000 |

Two things the tabular column makes undeniable:

1. **At zero coverage, every offline method collapses to *exactly* the SFT floor
   (0.697 = 0.697 = 0.697).** With no shared weights there is no gradient at the
   off-support error-branch states, so neither the advantage signal nor the full
   teacher distribution can move the student there. **The entire distillation gain
   over imitation, in the off-support region, is a generalization effect** — pull
   generalization out and Lightning OPD provably adds nothing over SFT. This is
   the sharpest possible statement of Assumption 3.2's fragility: the assumption
   does not just *degrade* off-support, it leaves the method with **no signal at
   all**.

   **This is also the direct refutation of the "capacity is the remedy" claim
   (§1).** The tabular student is the *highest-capacity* student in the study — one
   independent logit vector per state, zero approximation error, the teacher is
   exactly representable. The paper's argument predicts this is the *easy* regime
   (representable teacher ⇒ Thm 3.6 shared fixed point). Yet at zero coverage it is
   the *worst*: adding capacity this way makes the result **collapse to SFT**, not
   approach online. Capacity without coverage of the distilled signal buys exactly
   nothing. The remedy the paper names is the one axis that provably does not move
   the off-support number.

2. **Rang's headline win was carried by generalization.** In the linear student,
   full-distribution KD jumps to 1.000 as soon as recovery states *appear* in the
   data — because shared weights propagate that mass into the (many) error states
   the student actually reaches at deploy time. Strip generalization away and the
   same method, same data, tops out at ~0.92 and climbs only as fast as the
   tabular dataset literally enumerates error states. Coverage is doing the work;
   generalization is the amplifier.

`online_opd` stays at 1.000 in **both** representations — because it visits the
error states itself, so coverage is never the bottleneck. That pins the blame
precisely: the tabular collapse is a **coverage** failure, not a capacity or
optimization one.

> Note on framing: the off-support success rate does not crash to *zero* — the
> on-path golden behaviour the table did learn still solves the no-error episodes
> (~0.70). What crashes to zero is the **distillation gain over SFT** in the
> off-support region, which is the quantity Assumption 3.2 is actually about.

## 6. Do the proposed fixes work? (and a fix that *looks* like one but isn't)

- **Patch 1 — support-aware** (density weighting + a conservative KL anchor to
  `π_ref`, *no env access*): a real but partial gain (0.63 → 0.75 headline; holds
  its margin across both axes). It stabilizes off-support behaviour but cannot
  invent a recovery action that was never in the signal. This is an *honest* fix:
  it stays inside Lightning OPD's no-env-access contract and buys what that
  contract allows — variance reduction and a conservative anchor — and no more.

  **Correction to an earlier draft (the Thm 3.7 over-reading).** An earlier version
  of this report justified the anchor by appeal to the paper's **Theorem 3.7**,
  whose covariance term `−Cov_{π_ref}[w, f]` "empirically acts as a trust-region
  effect," and claimed the patch inherits an *implicit* regularization toward
  `π_ref` that helps off-support. That was an over-reading and we retract it. The
  implicit trust region in Thm 3.7 pulls the student **toward `π_ref`** — and on
  the error branch *that is exactly the wrong direction*. `π_ref` is a clean-demo
  SFT policy that **never learned `ask_followup`**; it is itself a non-recovering
  agent. Anchoring to it does not supply the missing skill, it *reinforces its
  absence*: in the off-support region the anchor actively competes with the (rare,
  correct) gradient that would raise recovery mass. This is not a footnote — it is
  precisely **why patch 1 stalls at 0.747 and cannot climb further** (§6.1 table,
  §axis B): the regularizer the paper sells as a stabilizer is, in the multi-turn
  off-support regime, a force pulling the student back toward the very policy whose
  blind spot caused the failure. The honest reading of Thm 3.7 is narrow: it
  stabilizes drift *where the teacher signal is dense* (on-path), and says nothing
  helpful off-support.
- **Patch 3 — full-distribution advantage** (§6.3): the minimal in-contract fix.
  Same data, same no-env-access, same advantage-PG form; the *only* change is
  taking the advantage's expectation under the teacher's full distribution instead
  of the single sampled action. It hits **1.000**, tying Rang. This is the fix the
  paper should have shipped.

### 6.1 Why patch 2 (DAgger refresh) is a trap, not a fix

Patch 2 periodically stops training, rolls the *current* student out **in the
environment**, has the teacher annotate the fresh (error-branch) states, and
appends them to the dataset. It posts 0.92 headline (up to 0.96 with coverage) —
but note it no longer even posts the best no-full-distribution number, since the
in-contract patch 3 beats it at 1.000 with zero env cost. We deliberately leave it
in the results, but we argue it should **not** be counted as a fix for Lightning
OPD, and the reason is the whole point of the paper:

> Lightning OPD's entire reason to exist is to **remove the environment (and the
> live teacher) from the training loop** — that is the cost it "sells the pots and
> pans" to save. The moment patch 2 calls `env.step` on student rollouts mid-
> training, it has spent exactly that resource back. And once you are paying for
> live environment access, the honest question is: *why keep the frozen offline
> advantage formula at all?* You are already standing in the regime where
> **Online RL** and **online OPD** live — and both of those hit **1.000** here
> (§3), cleanly beating DAgger's 0.92.

So patch 2 is not "offline OPD, fixed." It is "online learning, wearing offline
OPD's clothes," and a strictly worse version of it (it inherits the frozen
advantage's blind spots while paying the online price). We therefore read its
number the other way around — **as evidence *for* the thesis, not against it**:
the only way we found to close the coverage gap while keeping the sampled-action
advantage was to re-introduce env access, i.e. to break the one premise that
distinguishes Lightning OPD from plain online RL. A method whose only repair
requires deleting its defining assumption has a problem with that assumption.
The legitimate ways to close the gap are therefore: distill the **full teacher
distribution** offline (Rang, or our in-contract patch 3 — both keep the
no-env-access contract and hit 1.000), or admit you need env access and **just run
Online RL / online OPD** (also 1.000). DAgger is the dominated middle.

| | env access? | live teacher? | headline | honest verdict |
|---|---|---|---|---|
| Lightning OPD | no | no | 0.63 | fails (the result) |
| support-aware (patch 1) | no | no | 0.75 | honest partial fix |
| **DAgger refresh (patch 2)** | **yes** | yes | 0.92 | **mislabeled — this is online learning** |
| full-dist advantage (patch 3) | no | no | **1.00** | honest fix (full distribution) |
| Rang offline KD | no | no | **1.00** | honest fix (full distribution) |
| online OPD | yes | yes | 1.00 | the right tool once env is on the table |
| online RL | yes | no | 1.00 | the right tool once env is on the table |

Read top to bottom: there is no row that is *both* no-env-access *and* beats the
full-distribution offline methods. The frontier of honest options is
{support-aware, full-dist adv, Rang} offline and {online OPD, online RL} online —
patch 2 sits strictly inside it, paying online cost (~1.5k env steps) for a number
the in-contract patch 3 beats at zero env cost.

**The budget table, in full (no method gets to hide its cost).** Reviewers rightly
distrust comparisons that quietly let one method spend 100× the environment
interaction of another. So we report the *measured* training-time environment cost
of every method, mean over 5 seeds (`train_env_steps` in `results_main.csv`). The
column that matters is not the raw count but **when** the steps are spent:

| method | env access | training env-steps | env-steps spent… | headline |
|---|---|---:|---|---|
| teacher / SFT | none | **0** | — | 1.000 / 0.699 |
| **Lightning OPD** | none | **633** | once, at collection | 0.632 |
| support-aware (patch 1) | none | 642 | once, at collection | 0.747 |
| **full-dist adv (patch 3)** | none | **635** | once, at collection | **1.000** |
| Rang offline KD | none | 642 | once, at collection | **1.000** |
| DAgger refresh (patch 2) | **yes** | **1 538** | once + **mid-training rollouts** | 0.919 |
| online OPD | yes | **30 303** | **every gradient step** | 1.000 |
| online RL | yes | **154 526** | **every gradient step** | 1.000 |

Two things this makes honest and undeniable:

1. **The five no-env-access methods share one budget.** Lightning OPD, both its
   patches that keep the contract, and Rang all pay ≈ **630–640** env-steps — a
   *one-time* dataset collection (350 frozen rollouts) that 350 training iterations
   then reuse with **zero** further `env.step`. That collection could even be a
   pre-existing log. On this identical budget the score ranges from 0.632 (Lightning
   OPD) to 1.000 (patch 3, Rang). **The comparison is apples-to-apples; what
   separates them is the loss, not the budget.**
2. **DAgger and the online methods are a different cost class.** DAgger's 1 538
   steps are not a bigger one-time collection — they are *recurring* mid-training
   rollouts (`env.step` on the student every `refresh_every=60` iters), the exact
   resource Lightning OPD exists to remove. Online OPD/RL spend env-steps on *every*
   gradient step, 48× / 244× the offline budget. They reach 1.000 — but they are
   answering a different, more expensive question. **Once a method is in this
   column, the honest baseline is online OPD/RL (both 1.000), and DAgger's 0.919 is
   strictly dominated within its own cost class.** Reporting the budget is what
   turns "DAgger fixes Lightning OPD" into "DAgger left the no-env-access regime and
   underperformed the methods that legitimately live in the regime it entered."

### 6.2 Is Rang's 100% too good to be true? Stress-testing the headline

A perfect, zero-variance 1.000 should make a referee suspicious, so we tried to
break it rather than celebrate it. Three checks, all already in the data above:

1. **It is not 1.000 unconditionally.** At `collect_noise = 0` (§5 axis A, and the
   `linear / rang_kd / 0.00` cell of §5.3) Rang sits at **0.699 — the SFT floor.**
   With a truly clean frozen dataset the recovery states *never appear at all*,
   and full-distribution KD has nothing to transfer. Rang's 1.000 is **contingent
   on the collection process brushing the error branch at least occasionally**
   (any `collect_noise > 0`), not on the method being magic. That single 0.699
   cell is the tell: the win is a coverage-times-generalization effect, not a
   property of the loss alone.
2. **Strip generalization and the 1.000 disappears.** This is what the tabular
   column of §5.3 is *for*. Same loss, same data, per-state student with no shared
   weights → Rang tops out at **0.78 / 0.90 / 0.92** (rising only as fast as the
   tabular dataset literally enumerates error states) and never reaches 1.000.
   The headline 1.000 was **generalization spending coverage efficiently**, not the
   composite loss being inherently complete. Pull either ingredient (coverage *or*
   generalization) and it degrades gracefully and predictably — exactly what a real
   effect should do, and exactly what a too-good-to-be-true artifact would not.
3. **Why full-distribution KD uses coverage so much better than the advantage.**
   Lightning OPD's signal at a state is `log π_T(a) − log π_θ(a)` for the *single
   sampled action a*; on a clean golden-path rollout that action is never
   `ask_followup`, so the recovery mass is literally absent from the gradient. Rang
   matches the **whole** distribution `KL(P‖Q)`, so wherever the teacher target
   puts mass on `ask_followup`, that mass is present in the gradient and shared
   weights carry it into the error states the student reaches at deploy time. The
   one subtlety — and a place an earlier draft of this report was simply wrong —
   is *where* that recovery mass lives. We measured it directly: on golden-path
   states the teacher puts ≈`1e-8` on `ask_followup` (effectively zero), **not** a
   small constant tail. The recovery mass (mean ≈0.6, up to 1.0) sits **entirely
   on the error-branch states** — the all-against `misled` states — and those
   states only enter a frozen dataset when collection noise triggers them
   (0 such states at `collect_noise=0`; 32 / 56 / 99 records at 0.05 / 0.15 / 0.30).
   So full-distribution KD is not winning because the teacher leaks a little
   recovery probability everywhere; it is winning because, *once an error state
   appears at all*, the full-distribution target there is ≈1.0 on the recovery
   action while the sampled-action advantage at that same state is whatever
   non-recovery action the rollout happened to take. This is exactly why both
   full-distribution methods read 0.699 at zero coverage and 1.000 above it — and
   it is mechanistic, not lucky.

So the 100% is real but **fragile and fully explained**: it is the joint product of
(a) the full-distribution target carrying recovery mass and (b) parameter
generalization broadcasting it off-support. Remove (a) → Lightning OPD (0.63).
Remove (b) → tabular Rang (≤0.92). Remove the coverage that feeds both → 0.699
SFT floor. We report it as a ceiling that is *conditional on its two enabling
mechanisms*, not as an unqualified "Rang wins."

### 6.3 The minimal in-contract fix: full-distribution advantage (patch 3)

Rang's 1.000 leaves an obvious objection: maybe it wins because it is a *different
method* (a CE+KL loss), not because of the full-distribution target specifically.
Patch 3 is the control that isolates the variable. It keeps Lightning OPD's
gradient **exactly** — same frozen dataset, no env access, no live teacher, same
advantage-PG form, same clip — and changes one thing: instead of the sampled
action's advantage `A(a) · ∇log π(a)`, it takes the expectation under the
teacher's full distribution,

```
Σ_a π_T(a|s) · clip(log π_T(a|s) − log π_θ(a|s)) · ∇log π_θ(a|s)
```

That is the single edit separating "sampled-action advantage" from
"full-distribution advantage." The result is **1.000 ± .000**, tying Rang and
tracking it cell-for-cell across the entire coverage sweep (§5 axis A: 0.699 at
zero coverage, 1.000 above). It runs at the same ~635 env-step budget as plain
Lightning OPD — i.e. zero additional env interaction.

**Why this is not a variance argument — the two gradients have different
expectations.** It is tempting to read patch 3 as "the same gradient, lower
variance" (average over actions instead of sampling one). That is wrong, and the
distinction is the crux of the whole diagnosis. Write the per-state advantage-PG
contribution. Lightning OPD evaluates it at the **single rollout action** `a ~ b(·|s)`,
where `b` is the *behaviour* policy that generated the frozen dataset (here `π_ref`):

```
g_LOPD(s) = A(a, s) · ∇log π_θ(a|s),         a ~ b(·|s)
          = [log π_T(a|s) − log π_θ(a|s)] · ∇log π_θ(a|s)
```

Patch 3 replaces the sampled action by the expectation under the **teacher's** full
distribution:

```
g_FULL(s) = Σ_a π_T(a|s) · [log π_T(a|s) − log π_θ(a|s)] · ∇log π_θ(a|s)
          = E_{a ~ π_T(·|s)} [ A(a, s) · ∇log π_θ(a|s) ]
```

The catch is the sampling distribution. A Monte-Carlo estimate of `g_FULL` would
require `a ~ π_T`, but Lightning OPD's single sample is drawn from `b = π_ref`, not
`π_T`. So `g_LOPD` is an unbiased one-sample estimator of a **different** quantity,

```
E_{a ~ π_ref} [ A(a, s) · ∇log π_θ(a|s) ]   ≠   E_{a ~ π_T} [ A(a, s) · ∇log π_θ(a|s) ] = g_FULL(s),
```

and the two expectations coincide only when `π_ref ≈ π_T` on `supp(π_T(·|s))`.
**That is exactly the support-coverage assumption, re-derived as an importance-
sampling condition.** Off-support it fails maximally: at an error-branch state the
recovery action `r = ask_followup` has `π_T(r|s) ≈ 1.0` but `π_ref(r|s) ≈ 0` (the
clean-demo behaviour never plays it — we measured ≈`1e-8`, §6.2). The term that
dominates `g_FULL`,

```
π_T(r|s) · A(r, s) · ∇log π_θ(r|s)     with π_T(r|s) ≈ 1.0 and A(r,s) ≫ 0,
```

is sampled by `g_LOPD` with probability `π_ref(r|s) ≈ 0`. **No amount of averaging
over more rollouts recovers it**, because every rollout is drawn from the same
`π_ref` that puts zero mass on `r`: the missing term is not high-variance, it is
*systematically absent* from the support of the estimator. Increasing the dataset
(more rollouts) is the wrong axis — it shrinks variance on the terms `π_ref` already
samples and does nothing for the term it never can. This is the formal content of
"flat across coverage" in §5 axis A.

Two clarifications keep this honest. **(i)** `g_FULL` is *not* the exact gradient of
`KL(π_T‖π_θ)` — that gradient is `Σ_a π_T(a)∇log π_θ(a)`, without the advantage
weight. `g_FULL` is the advantage-weighted policy gradient with the **teacher** as
the sampling distribution; it keeps Lightning OPD's stop-grad advantage `A(a,s)`
verbatim and changes *only* the measure the per-action term is summed against, from
`π_ref` (one sample) to `π_T` (closed-form). What the two share is the fixed point:
at `π_θ = π_T` every `A(a,s) = 0`, so both `g_FULL` and the true on-policy gradient
vanish there. **(ii)** Patch 3 is therefore not a variance-reduced estimator of
`g_LOPD`'s target; it computes a **different** target exactly, in closed form, using
the one piece of information Lightning OPD already precomputed but only ever read at
the sampled index — the full row `π_T(·|s)`. It costs no env access because the
recovery mass lives in `π_T` (precomputed offline), not in `π_ref` (which did the
sampling).

This pins the diagnosis with no room left to argue:

- It is **not** the no-env-access contract that dooms Lightning OPD (patch 3 keeps
  it and hits ceiling).
- It is **not** that Rang's specific CE+KL composite is magic (a one-line change to
  Lightning OPD's own gradient reproduces the win).
- It **is** the sampled-action restriction — formally, estimating `g_FULL` by a
  single draw from `π_ref` rather than evaluating the `π_T`-expectation in closed
  form. The recovery mass lives at the error-branch states (§6.2), the teacher
  target there is ≈1.0 on `ask_followup`, and the full-distribution expectation puts
  that mass in the gradient while the sampled action — drawn from a near-
  deterministic `π_ref` — is `r` with probability ≈0. Same coverage, same
  generalization, opposite outcome, on the strength of one expectation.

Patch 3 is therefore the honest fix the paper should have shipped: it stays inside
the exact contract Lightning OPD sells ("no teacher, no env in the loop") and still
closes the gap, because the gap was never about the contract — it was about
distilling one action instead of the whole distribution.

### 6.4 What the 1.000 hides: greedy vs sampled evaluation

The headline metric takes **greedy (argmax)** actions. That is the standard
protocol, but it flatters offline distillation specifically, and a careful referee
should ask what the *distribution* the student actually learned looks like — not
just its mode. We re-evaluated every method by **sampling** from the policy
(`sampled_success_rate` in `results_main.csv`), which scores a method down whenever
it puts the recovery action on top by only a thin margin. The contrast is sharp:

| method | greedy | sampled | Δ (sampled − greedy) | env access |
|---|---|---|---|---|
| teacher | 1.000 | 1.000 | 0.00 | — |
| online RL | 1.000 | 0.997 | −0.00 | yes |
| online OPD | 1.000 | 0.990 | −0.01 | yes |
| **Rang offline KD** | **1.000** | **0.869** | **−0.13** | no |
| **full-dist adv (patch 3)** | **1.000** | **0.864** | **−0.14** | no |
| support-aware (patch 1) | 0.747 | 0.743 | −0.00 | no |
| SFT | 0.699 | 0.749 | +0.05 | no |
| Lightning OPD | 0.632 | 0.665 | +0.03 | no |

Two honest admissions, both of which *sharpen* rather than soften the thesis:

1. **Our own 1.000s are greedy-conditional.** Both full-distribution offline methods
   drop to ≈0.86 under sampling — a **−0.13/−0.14** gap. So patch 3 does not learn a
   policy that is *confidently* recovering; it learns one where `ask_followup` is
   merely the **plurality** action at the error states, with ~13% of the mass still
   leaking onto wrong actions. Greedy evaluation rounds that plurality up to a clean
   win. We report the 1.000 as **"argmax-aligned," not "distribution-complete."**
2. **The greedy–sampled gap is itself a coverage thermometer.** The two *online*
   methods barely move (−0.00/−0.01): having visited and been rewarded at the error
   states, they drive the recovery mass to ≈1.0, so sampling and argmax agree. The
   two *offline* methods lose 0.13 precisely because off-support they only ever saw
   the recovery action *once it appeared in data*, never reinforced it to dominance.
   The size of the greedy→sampled drop measures **how much of the win is real mass
   vs argmax luck** — and it cleanly separates "visited the states" (online, ~0)
   from "only distilled the states" (offline, ~0.13). This is the same coverage
   axis as the main result, read off a different metric.

So even our fix carries the fingerprint of the disease. The honest framing is not
"patch 3 solves it" but "patch 3 recovers the *mode* off-support at zero env cost,
and closing the remaining mass gap is exactly what costs env access (online OPD/RL,
sampled ≈0.99)."

Together §6.1, §6.2, §6.3 and §6.4 say the diagnosis is the same from every
direction: the gap is **missing recovery supervision off-support**, and the only
honest ways to supply it are a full-distribution offline target (Rang or patch 3,
when collection has any coverage) or live environment access (online OPD / RL). The
sampled-action frozen advantage can do neither — and the "fix" that appears to
rescue it (DAgger) does so only by quietly becoming online learning.

## 7. Takeaway

We do not attack Lightning OPD's admitted open problem (multi-turn coverage); we
attack its **central, load-bearing claim**: that under *teacher consistency* the
online–offline gap is governed by drift (Thm 3.5) and any residual is a
**capacity-limited** matter (§1). We satisfy that claim's premises exactly —
consistency holds by construction (`σ_Δ² = 0`) and capacity is sufficient (the same
linear student reaches 1.000 once fed the full distribution) — and the method still
lands **at or below SFT (0.63 vs 0.699)** with an order of magnitude more variance.
The surviving gap is therefore neither a consistency bias nor a capacity limit. It
is a coverage failure *of the distilled signal* that Thm 3.5 does not forbid:
off-support the drift `χ²(π_θ‖π_ref)` diverges and the bound goes vacuous. Meanwhile
offline *full-distribution* targets — Rang et al. *and* a one-line change to
Lightning OPD's own gradient (patch 3) — stay at ceiling on the identical
no-env-access budget.

The representation ablation sharpens this to its root cause and **refutes the
capacity remedy head-on**: the *highest-capacity* student (per-state tabular, teacher
exactly representable) collapses to **exactly** the SFT floor at zero coverage,
because off-support states receive no gradient at all. Adding capacity that way
makes the result worse, not online-like. Whatever rescue we saw in the linear case
was generalization amplifying coverage, never a substitute for it. So three messages,
in order of force:

1. The failure is **not** what the paper's theory locates it in. With consistency
   satisfied and capacity sufficient, the method still fails — because off-support
   it has **no signal**, and only literal support (or live env access) can supply
   one. The "capacity is the remedy" extrapolation is falsified by the tabular
   column, where maximal capacity at zero coverage *is* the worst case.
2. *Given* generalization, **what you distill** (full distribution vs
   sampled-action advantage) decides whether that coverage is enough. We isolate
   this variable with patch 3 (§6.3): hold the dataset, the no-env-access contract,
   and the advantage-PG form fixed, swap *only* the sampled action for the teacher's
   full distribution, and the score goes 0.63 → 1.000 — one line, in-contract. But
   we do not oversell it: under *sampled* evaluation that 1.000 is ≈0.86 (§6.4), so
   patch 3 recovers the recovery action's **mode**, not its full mass; closing the
   remaining gap is exactly what costs env access (online OPD/RL, sampled ≈0.99).
3. The results that look like they cut against the thesis don't. DAgger's 0.92 is
   **online learning in disguise** (§6.1): it spends the env access Lightning OPD
   exists to avoid (~1.5k steps vs the offline ~635 one-time collection), is
   dominated by the online methods it has become, and is beaten at zero marginal env
   cost by patch 3. Rang's (and patch 3's) 1.000 is **not unconditional** (§6.2): it
   falls to the SFT floor at zero collection coverage and to ≤0.92 once
   generalization is removed — a coverage×generalization ceiling, not evidence that
   any offline loss is safe off-support. And the proposed conservative anchor is a
   *misuse* of Thm 3.7's implicit trust region (§6, patch 1): it pulls the student
   toward `π_ref`, the non-recovering policy whose blind spot caused the failure,
   which is why it stalls at 0.747. Every "surprise" reduces to the same mechanism
   as the main failure.

## Reproduce

```bash
cd opd_toy
python run_experiments.py     # writes ../results/*.csv + case_studies.txt
python plots.py               # writes ../results/fig_*.png
```

Artifacts: `results_main.csv`, `results_coverage.csv`, `results_deploy.csv`,
`results_representation.csv`, `case_studies.txt`, and `fig_{main,coverage,deploy,representation}.png`.
