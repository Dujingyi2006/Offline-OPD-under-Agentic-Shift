"""Plots from the experiment CSVs (matplotlib, no seaborn dependency)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# stable display order + labels + colors
METHOD_ORDER = [
    "sft", "offline_opd", "offline_opd_support_aware", "offline_opd_full_adv",
    "offline_opd_dagger", "rang_kd", "online_opd", "online_rl", "teacher",
]
LABELS = {
    "sft": "SFT",
    "rang_kd": "Rang et al. KD",
    "offline_opd": "Lightning OPD\n(offline)",
    "offline_opd_support_aware": "+ support-aware\n(patch 1)",
    "offline_opd_full_adv": "+ full-dist adv\n(patch 3)",
    "offline_opd_dagger": "+ DAgger refresh\n(patch 2)",
    "online_opd": "Online OPD",
    "online_rl": "Online RL",
    "teacher": "Teacher",
}
COLORS = {
    "sft": "#9e9e9e",
    "rang_kd": "#b07aa1",
    "offline_opd": "#e15759",
    "offline_opd_support_aware": "#f28e2b",
    "offline_opd_full_adv": "#edc948",
    "offline_opd_dagger": "#59a14f",
    "online_opd": "#4e79a7",
    "online_rl": "#76b7b2",
    "teacher": "#333333",
}


def _agg(df, group_cols, value="success_rate"):
    g = df.groupby(group_cols)[value]
    return g.mean(), g.std()


def plot_main(df):
    means, stds = _agg(df, ["method"])
    methods = [m for m in METHOD_ORDER if m in means.index]
    y = [means[m] for m in methods]
    err = [stds[m] if not np.isnan(stds[m]) else 0.0 for m in methods]
    colors = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(methods)), y, yerr=err, capsize=4, color=colors)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([LABELS[m] for m in methods], fontsize=9)
    ax.set_ylabel("Deployment success rate")
    ax.set_ylim(0, 1.05)
    ax.axhline(means.get("sft", 0), ls="--", c="#9e9e9e", lw=1, zorder=0)
    ax.set_title("Multi-turn agentic deployment (collect noise 0.05, deploy 0.30)\n"
                 "Offline OPD fails to recover from errors it never saw offline")
    for b, v in zip(bars, y):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_main.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_coverage(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ["offline_opd", "rang_kd", "offline_opd_support_aware",
              "offline_opd_full_adv", "offline_opd_dagger"]:
        sub = df[df.method == m]
        if sub.empty:
            continue
        means, stds = _agg(sub, ["collect_noise"])
        x = means.index.values
        ax.errorbar(x, means.values, yerr=stds.values, marker="o",
                    capsize=3, color=COLORS[m], label=LABELS[m].replace("\n", " "))
    ax.set_xlabel("Collection coverage  (collect-time tool noise -> recovery states in offline data)")
    ax.set_ylabel("Deployment success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Axis A: offline OPD is coverage-limited, not capacity-limited")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_coverage.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_deploy(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in ["sft", "offline_opd", "rang_kd", "offline_opd_support_aware",
              "online_opd"]:
        sub = df[df.method == m]
        if sub.empty:
            continue
        means, stds = _agg(sub, ["deploy_noise"])
        x = means.index.values
        ax.errorbar(x, means.values, yerr=stds.values, marker="s",
                    capsize=3, color=COLORS[m], label=LABELS[m].replace("\n", " "))
    ax.set_xlabel("Deployment distribution shift  (deploy-time tool noise)")
    ax.set_ylabel("Deployment success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Axis B: the offline/online gap widens with distribution shift")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_deploy.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_representation(df):
    """Grouped bars: linear (generalizing) vs tabular (per-state) student at a
    fixed low-coverage point. Shows the distillation gain over SFT collapsing
    when generalization is removed."""
    cn = sorted(df.collect_noise.unique())[1] if len(df.collect_noise.unique()) > 1 \
        else df.collect_noise.iloc[0]
    sub = df[df.collect_noise == cn]
    methods = [m for m in ["sft", "offline_opd", "rang_kd", "online_opd"]
               if m in sub.method.unique()]
    reprs = ["linear", "tabular"]
    width = 0.38
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    hatch = {"linear": None, "tabular": "//"}
    for i, rep in enumerate(reprs):
        means, stds = _agg(sub[sub.representation == rep], ["method"])
        y = [means.get(m, np.nan) for m in methods]
        err = [stds.get(m, 0.0) if not np.isnan(stds.get(m, np.nan)) else 0.0
               for m in methods]
        bars = ax.bar(x + (i - 0.5) * width, y, width, yerr=err, capsize=3,
                      color=[COLORS[m] for m in methods], hatch=hatch[rep],
                      edgecolor="white",
                      label="linear (generalizes)" if rep == "linear"
                      else "tabular (no generalization)")
        for b, v in zip(bars, y):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                        ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m].replace("\n", " ") for m in methods], fontsize=9)
    ax.set_ylabel("Deployment success rate")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Representation ablation (collect noise {cn}, deploy 0.30)\n"
                 "Removing generalization collapses every offline rescue to the SFT floor")
    # solid vs hatched legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#777", label="linear (generalizes)"),
                       Patch(facecolor="#777", hatch="//", edgecolor="white",
                             label="tabular (no generalization)")],
              fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_representation.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    outs = []
    main_csv = os.path.join(RESULTS_DIR, "results_main.csv")
    cov_csv = os.path.join(RESULTS_DIR, "results_coverage.csv")
    dep_csv = os.path.join(RESULTS_DIR, "results_deploy.csv")
    rep_csv = os.path.join(RESULTS_DIR, "results_representation.csv")
    if os.path.exists(main_csv):
        outs.append(plot_main(pd.read_csv(main_csv)))
    if os.path.exists(cov_csv):
        outs.append(plot_coverage(pd.read_csv(cov_csv)))
    if os.path.exists(dep_csv):
        outs.append(plot_deploy(pd.read_csv(dep_csv)))
    if os.path.exists(rep_csv):
        outs.append(plot_representation(pd.read_csv(rep_csv)))
    for o in outs:
        print("wrote", o)


if __name__ == "__main__":
    main()
