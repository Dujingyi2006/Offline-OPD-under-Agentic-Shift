"""Paper-ready figures for the Offline OPD study.

Two figures, each rendered to vector PDF (camera-ready) and high-res PNG:
  Fig A  Performance-vs-budget Pareto front   -> fig_pareto.{pdf,png}
  Fig B  2x2 faceted CI-band sweeps           -> fig_sweeps.{pdf,png}

Statistics: every point/line aggregates 5 seeds. Shaded bands and error bars
are 95% confidence intervals (Student-t, df = n-1). No seaborn dependency.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ---- shared house style -----------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.7,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,   # editable text in vector PDF
    "ps.fonttype": 42,
})

# semantic palette (echoes the architecture figure)
C_BASELINE = "#9aa3ad"      # sft / teacher
C_NAIVE = "#cf6b6b"         # bare offline_opd (fragile)
C_RANG = "#b07aa1"          # rang et al. KD
C_INCONTRACT1 = "#5a9e6f"   # full-dist adv
C_INCONTRACT2 = "#3f8f8a"   # branch-aware replay
C_BREAKS = "#d9a441"        # active query (breaks contract)
C_ONLINE = "#4e79a7"        # online opd / rl


def ci95(series: pd.Series):
    """Mean and half-width of the 95% t-CI over seeds."""
    vals = series.dropna().values
    n = len(vals)
    m = float(np.mean(vals)) if n else np.nan
    if n < 2:
        return m, 0.0
    sem = float(np.std(vals, ddof=1) / np.sqrt(n))
    h = sem * stats.t.ppf(0.975, n - 1)
    return m, h


def agg_ci(df, xcol, ycol):
    """Return sorted x, mean(y), ci-halfwidth(y) grouped over seeds."""
    xs, ms, hs = [], [], []
    for xv, g in df.groupby(xcol):
        m, h = ci95(g[ycol])
        xs.append(xv)
        ms.append(m)
        hs.append(h)
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ms = np.array(ms)[order]
    hs = np.array(hs)[order]
    return xs, ms, hs


# =============================================================================
# Figure A: Pareto front  (success rate vs interaction budget)
# =============================================================================
def fig_pareto(df: pd.DataFrame, out_base: str):
    spec = [
        ("sft",                       "SFT",                    C_BASELINE,    "o", "zero"),
        ("teacher",                   "Teacher",                "#333333",     "*", "zero"),
        ("offline_opd",               "Lightning OPD",          C_NAIVE,       "X", "in"),
        ("rang_kd",                   "Rang et al. KD",         C_RANG,        "D", "in"),
        ("offline_opd_full_adv",      "+ full-dist adv (P2)",   C_INCONTRACT1, "o", "in"),
        ("offline_opd_branch_replay", "+ branch replay (P3)",   C_INCONTRACT2, "s", "in"),
        ("offline_opd_active_query",  "+ active query (P1)",    C_BREAKS,      "^", "breaks"),
        ("online_opd",                "Online OPD",             C_ONLINE,      "P", "online"),
        ("online_rl",                 "Online RL",              "#8ca9c9",     "v", "online"),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    pts = {}
    for key, label, color, marker, klass in spec:
        sub = df[df.method == key]
        if sub.empty:
            continue
        y, yh = ci95(sub["sampled_success_rate"])
        steps = sub["train_env_steps"].fillna(0)
        x = float(steps.mean())
        if key == "offline_opd_active_query":
            x += float(sub["teacher_queries"].fillna(0).mean())
        pts[key] = (x, y)

        face = color
        edge = "white"
        if klass == "breaks":
            face = "white"
            edge = color
        ax.errorbar(x, y, yerr=yh, fmt=marker, ms=11, color=color,
                    markerfacecolor=face, markeredgecolor=edge,
                    markeredgewidth=1.6, capsize=4, elinewidth=1.3,
                    ecolor=color, zorder=5, label=label)

    frontier_keys = ["sft", "offline_opd_full_adv", "offline_opd_branch_replay",
                     "online_opd", "online_rl"]
    fx = np.array([pts[k][0] for k in frontier_keys if k in pts])
    fy = np.array([pts[k][1] for k in frontier_keys if k in pts])
    order = np.argsort(fx)
    ax.plot(fx[order], fy[order], ls="--", lw=1.3, color="#9aa3ad", zorder=1)

    ax.set_xscale("symlog", linthresh=300)
    ax.set_xlim(-80, 4e5)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlabel("Interaction budget  —  environment steps (+ teacher queries),  symlog")
    ax.set_ylabel("Deployment success rate  (sampled policy, 95% CI)")
    ax.set_title("Performance–budget Pareto front\n"
                 "in-contract patches reach the online ceiling at ~50× lower budget")

    if "offline_opd_branch_replay" in pts and "online_opd" in pts:
        xb, yb = pts["offline_opd_branch_replay"]
        ax.annotate("same ceiling,\n~50× less interaction",
                    xy=(xb, yb), xytext=(1800, 0.64),
                    fontsize=8.5, color="#3f8f8a", ha="left",
                    arrowprops=dict(arrowstyle="->", color="#3f8f8a", lw=1.1))
    ax.axvline(0, color="#cccccc", lw=0.8, ls=":", zorder=0)
    ax.text(0, 0.515, " zero-interaction", fontsize=7.5, color="#999999",
            rotation=90, va="bottom", ha="right")

    ax.legend(loc="lower right", ncol=2, framealpha=0.92, handletextpad=0.3,
              columnspacing=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=220)
    plt.close(fig)
    return out_base


# =============================================================================
# Figure B: 2x2 faceted CI-band sweeps
# =============================================================================
def fig_sweeps(cov, dep, patch3, active, out_base: str):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), sharey=True)

    # ---- (a) collection coverage sweep ----
    ax = axes[0, 0]
    for key, label, color in [
        ("offline_opd", "Lightning OPD", C_NAIVE),
        ("offline_opd_full_adv", "+ full-dist adv", C_INCONTRACT1),
        ("offline_opd_branch_replay", "+ branch replay", C_INCONTRACT2),
        ("rang_kd", "Rang et al. KD", C_RANG),
    ]:
        sub = cov[cov.method == key]
        if sub.empty:
            continue
        x, m, h = agg_ci(sub, "collect_noise", "success_rate")
        ax.plot(x, m, marker="o", ms=5, color=color, lw=1.6, label=label)
        ax.fill_between(x, m - h, m + h, color=color, alpha=0.16, lw=0)
    ax.set_title("(a) Collection coverage")
    ax.set_xlabel("Collect-time tool noise")
    ax.set_ylabel("Deployment success rate")
    ax.legend(loc="lower right")

    # ---- (b) deployment shift sweep ----
    ax = axes[0, 1]
    for key, label, color in [
        ("sft", "SFT", C_BASELINE),
        ("offline_opd", "Lightning OPD", C_NAIVE),
        ("offline_opd_full_adv", "+ full-dist adv", C_INCONTRACT1),
        ("online_opd", "Online OPD", C_ONLINE),
        ("rang_kd", "Rang et al. KD", C_RANG),
    ]:
        sub = dep[dep.method == key]
        if sub.empty:
            continue
        x, m, h = agg_ci(sub, "deploy_noise", "success_rate")
        ax.plot(x, m, marker="s", ms=5, color=color, lw=1.6, label=label)
        ax.fill_between(x, m - h, m + h, color=color, alpha=0.16, lw=0)
    ax.set_title("(b) Deployment distribution shift")
    ax.set_xlabel("Deploy-time tool noise")
    ax.legend(loc="lower left")

    # ---- (c) patch-3 gamma sweep ----
    ax = axes[1, 0]
    p3 = patch3.copy()
    p3["gamma"] = p3["method"].str.replace("branch_replay_g", "", regex=False).astype(float)
    x, m, h = agg_ci(p3, "gamma", "sampled_success_rate")
    ax.plot(x, m, marker="o", ms=5, color=C_INCONTRACT2, lw=1.8,
            label="branch replay (sampled)")
    ax.fill_between(x, m - h, m + h, color=C_INCONTRACT2, alpha=0.18, lw=0)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_title("(c) Patch 3: replay strength γ")
    ax.set_xlabel("Reweighting strength  γ   (symlog)")
    ax.set_ylabel("Deployment success rate")
    ax.legend(loc="lower right")

    # ---- (d) active-query tau sweep ----
    ax = axes[1, 1]
    aq = active.copy()
    aq["tau"] = aq["method"].str.replace("active_query_ct", "", regex=False).astype(float)
    x, m, h = agg_ci(aq, "tau", "success_rate")
    ax.plot(x, m, marker="^", ms=6, color=C_BREAKS, lw=1.8,
            label="active query (greedy)")
    ax.fill_between(x, m - h, m + h, color=C_BREAKS, alpha=0.18, lw=0)
    # secondary axis: teacher-query cost
    ax2 = ax.twinx()
    xq, mq, hq = agg_ci(aq, "tau", "teacher_queries")
    ax2.plot(xq, mq, marker="x", ms=6, color="#b06a1f", lw=1.2, ls="--",
             label="teacher queries (cost)")
    ax2.set_ylabel("Teacher queries", color="#b06a1f")
    ax2.tick_params(axis="y", labelcolor="#b06a1f")
    ax2.grid(False)
    ax.set_title("(d) Patch 1: confidence threshold τ")
    ax.set_xlabel("Query threshold  τ")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")

    for a in axes.ravel():
        a.set_ylim(0.4, 1.05)

    fig.suptitle("Sweep ablations  (5 seeds, shaded = 95% CI)", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_base}.{ext}", dpi=220)
    plt.close(fig)
    return out_base


def main():
    rd = RESULTS_DIR
    main_df = pd.read_csv(os.path.join(rd, "results_main.csv"))
    cov = pd.read_csv(os.path.join(rd, "results_coverage.csv"))
    dep = pd.read_csv(os.path.join(rd, "results_deploy.csv"))
    p3 = pd.read_csv(os.path.join(rd, "results_patch3.csv"))
    aq = pd.read_csv(os.path.join(rd, "results_active.csv"))

    a = fig_pareto(main_df, os.path.join(rd, "fig_pareto"))
    b = fig_sweeps(cov, dep, p3, aq, os.path.join(rd, "fig_sweeps"))
    print("wrote", a + ".{pdf,png}")
    print("wrote", b + ".{pdf,png}")


if __name__ == "__main__":
    main()
