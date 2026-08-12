#!/usr/bin/env python3
"""Render the README charts from the exported CSVs in docs/data/.

The chain is auditable end to end: training logs -> export_charts.py ->
docs/data/*.csv -> this script -> docs/media/*.svg. Every value plotted here is
read from those CSVs at run time. The one exception is the frame-budget figure,
whose three constants are transcribed from the README's own "Performance
(A100-40GB, bf16)" table and are marked as such below.

Each figure is written twice, light and dark, and embedded in the README with a
<picture> element so GitHub serves the right one for the reader's theme.

    python3 scripts/eval/make_readme_charts.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "docs", "data")
OUT = os.path.join(ROOT, "docs", "media")
os.makedirs(OUT, exist_ok=True)

SANS = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# surfaces match GitHub's own README backgrounds; series are slots 1-2 of the
# validated categorical palette, stepped per mode
THEME = {
    "light": dict(surface="#ffffff", primary="#0b0b0b", secondary="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  s1="#2a78d6", s2="#eb6834"),
    "dark": dict(surface="#0d1117", primary="#e6edf3", secondary="#c3c2b7",
                 muted="#898781", grid="#21262d", axis="#383835",
                 s1="#3987e5", s2="#d95926"),
}

# --- the batch-64 run, in the two log segments it was recorded in -------------
MAIN, RESUME = "b64_main_pre_crash", "b64_resume"
RESUME_STEP = 770


def read(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def series(rows, run, col):
    """(step, value) for one run and column, dropping blanks and NaNs."""
    out = []
    for r in rows:
        if r["run"] != run or r[col] in ("", "nan"):
            continue
        out.append((int(r["step"]), float(r[col])))
    return sorted(set(out))


def whole_run(rows, col):
    """The full 0->3000 trajectory, stitched across the crash at step 770."""
    a = [p for p in series(rows, MAIN, col) if p[0] < RESUME_STEP]
    b = series(rows, RESUME, col)
    pts = sorted(set(a + b))
    return [p[0] for p in pts], [p[1] for p in pts]


def style(fig, axes, t, grid_axis="y"):
    fig.patch.set_facecolor(t["surface"])
    for ax in axes:
        ax.set_facecolor(t["surface"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(t["axis"])
            ax.spines[s].set_linewidth(1)
        ax.tick_params(colors=t["muted"], labelsize=9, length=0, pad=6)
        ax.grid(True, color=t["grid"], linewidth=1, axis=grid_axis)
        ax.set_axisbelow(True)


def save(fig, name, mode):
    p = os.path.join(OUT, f"{name}-{mode}.svg")
    fig.savefig(p, format="svg", bbox_inches="tight", pad_inches=0.3,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    s = open(p).read()
    s = s.replace("font-family: sans-serif", f"font-family: {SANS}")
    s = s.replace('font-family="sans-serif"', f'font-family="{SANS}"')
    open(p, "w").write(s)
    print("wrote", os.path.relpath(p, ROOT))


# --- 1. one block of inference ----------------------------------------------
# transcribed from README.md, "Performance (A100-40GB, bf16)"
GEN_MS, DEC_MS, BUDGET_MS = 480, 343, 750


def frame_budget(mode):
    t = THEME[mode]
    fig, ax = plt.subplots(figsize=(9.2, 2.4))
    style(fig, [ax], t)
    ax.grid(False)

    ax.barh(0, GEN_MS, height=0.42, color=t["s1"], zorder=3)
    ax.barh(0, DEC_MS, height=0.42, left=GEN_MS + 8, color=t["s2"], zorder=3)

    # threshold sits above the bar so it never crosses a value label
    ax.plot([BUDGET_MS] * 2, [0.24, 0.78], color=t["primary"], linewidth=2,
            linestyle=(0, (4, 3)), zorder=4, clip_on=False)
    ax.text(BUDGET_MS - 14, 0.58, "real-time budget  750 ms", ha="right",
            va="center", color=t["primary"], fontsize=9.5, fontweight="bold")
    ax.annotate("", xy=(GEN_MS + 8 + DEC_MS, 0.33), xytext=(BUDGET_MS, 0.33),
                arrowprops=dict(arrowstyle="<->", color=t["secondary"], lw=1.2))
    ax.text((BUDGET_MS + GEN_MS + DEC_MS + 8) / 2, 0.47, "+73 ms", ha="center",
            color=t["secondary"], fontsize=9)

    ax.text(GEN_MS / 2, 0, f"generate  {GEN_MS} ms", ha="center", va="center",
            color="#ffffff", fontsize=10, fontweight="bold", zorder=5)
    ax.text(GEN_MS + 8 + DEC_MS / 2, 0, f"VAE decode  {DEC_MS} ms", ha="center",
            va="center", color="#ffffff", fontsize=10, fontweight="bold", zorder=5)

    ax.set_xlim(0, 980)
    ax.set_ylim(-0.75, 0.9)
    ax.set_yticks([])
    ax.set_xticks([0, 250, 500, 750])
    ax.set_xticklabels(["0", "250", "500", "750 ms"])
    ax.spines["left"].set_visible(False)
    ax.text(0, -0.46, "823 ms of compute per 750 ms of video  →  0.91× real time",
            ha="left", va="center", color=t["secondary"], fontsize=10)
    ax.set_title("One block of streaming, on an A100-40GB", loc="left",
                 color=t["primary"], fontsize=12.5, fontweight="bold", pad=16)
    save(fig, "frame-budget", mode)


# --- 2. the 3000-step batch-64 run ------------------------------------------
def training_run(mode):
    t = THEME[mode]
    st = read("step_times.csv")
    tc = read("training_curves.csv")
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.8), sharex=True)
    style(fig, axes, t)

    def heading(ax, title, why):
        ax.set_title(title, loc="left", color=t["primary"], fontsize=11,
                     fontweight="bold", pad=30)
        ax.text(0, 1.045, why, transform=ax.transAxes, color=t["secondary"],
                fontsize=8.5, va="bottom")

    # -- step time
    ax = axes[0]
    xs, ys = whole_run(st, "instantaneous_s_per_it")
    ax.axvline(RESUME_STEP, color=t["muted"], linewidth=1.5,
               linestyle=(0, (4, 3)), zorder=2)
    ax.plot(xs, ys, color=t["s1"], linewidth=1.6, zorder=3)
    ax.text(RESUME_STEP + 40, 97, "crashed at 770,\nresumed from step 750",
            color=t["muted"], fontsize=8.5, va="top")
    # warm-up measures 24.2 s/it against a 66.3 s/it median once it ends at step 160
    ax.annotate("critic warm-up:\ngenerator update skipped,\nso 24 s/it not 66",
                xy=(100, 26), xytext=(150, 57), color=t["muted"], fontsize=8.5,
                va="top", arrowprops=dict(arrowstyle="-", color=t["muted"], lw=1))
    # 760 / 1510 / 2260 are the log intervals containing the 750 / 1500 / 2250 saves
    ax.annotate("milestone checkpoint saves", xy=(1510, 74.5), xytext=(1620, 92),
                color=t["muted"], fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=t["muted"], lw=1))
    ax.set_ylim(0, 105)
    ax.set_ylabel("seconds / iteration", color=t["muted"], fontsize=9)
    heading(ax, "Step time never drifts, because every iteration does identical work",
            "Same batch, same sequence length, same passes each step, so the cost per "
            "step is constant. Median 66.3 s/it.")

    # -- losses
    ax = axes[1]
    gx, gy = whole_run(tc, "loss_gen")
    cx, cy = whole_run(tc, "loss_critic")
    ax.axvline(RESUME_STEP, color=t["muted"], linewidth=1.5,
               linestyle=(0, (4, 3)), zorder=2)
    ax.plot(gx, gy, color=t["s1"], linewidth=1.6, zorder=3, label="generator")
    ax.plot(cx, cy, color=t["s2"], linewidth=1.6, zorder=3, label="critic")
    ax.annotate("generator", (gx[-1], gy[-1]), textcoords="offset points",
                xytext=(8, 0), color=t["s1"], fontsize=9, fontweight="bold",
                va="center", annotation_clip=False)
    ax.annotate("critic", (cx[-1], cy[-1]), textcoords="offset points",
                xytext=(8, 0), color=t["s2"], fontsize=9, fontweight="bold",
                va="center", annotation_clip=False)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel("loss", color=t["muted"], fontsize=9)
    ax.set_xlabel("training step", color=t["secondary"], fontsize=9.5, labelpad=8)
    heading(ax, "Neither loss trends down, because the critic is retrained every step",
            "The generator is holding position against an opponent that keeps improving. "
            "A falling generator loss would mean a stale critic, not progress.")

    for ax in axes:
        ax.set_xlim(0, 3180)

    # upper left, clear of both the direct labels and the resume line at step 770
    leg = axes[1].legend(loc="upper left", frameon=False, fontsize=9, ncol=1,
                         handlelength=1.6, borderpad=0.1, labelspacing=0.3)
    for txt in leg.get_texts():
        txt.set_color(t["secondary"])

    fig.suptitle("The batch-64 distillation run: 3000 steps on 8×H200, 41.6 hours",
                 x=0.005, y=1.005, ha="left", color=t["primary"], fontsize=12.5,
                 fontweight="bold")
    fig.tight_layout(h_pad=2.0)
    save(fig, "training-run", mode)


# --- 3. checkpoint trajectory ------------------------------------------------
def checkpoints(mode):
    t = THEME[mode]
    rows = read("checkpoint_traj.csv")
    ref = {r["arm"]: r for r in rows if not r["step"]}
    run = sorted((int(r["step"]), r) for r in rows if r["step"])
    steps = [s for s, _ in run]

    panels = [("mean_abs_decay_dev", "mean |decay−1|", "lower is better", (0, 0.16)),
              ("mean_abs_drift", "mean |drift|", "lower is better", (0, 0.16)),
              ("mean_motion", "mean motion", "higher is better", (0, 3.1))]

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    style(fig, axes, t)

    for ax, (col, title, hint, ylim) in zip(axes, panels):
        ys = [float(r[col]) for _, r in run]
        ax.axhline(float(ref["v5"][col]), color=t["muted"], linewidth=1.5,
                   linestyle=(0, (1.5, 2.5)), zorder=2)
        ax.axhline(float(ref["base"][col]), color=t["s2"], linewidth=1.5,
                   linestyle=(0, (4, 3)), zorder=2)
        ax.plot(steps, ys, color=t["s1"], linewidth=2, marker="o", markersize=5,
                markerfacecolor=t["surface"], markeredgewidth=1.8, zorder=3)
        ax.plot([steps[-1]], [ys[-1]], marker="o", markersize=8, color=t["s1"],
                zorder=4)
        ax.annotate(f"{ys[-1]:g}", (steps[-1], ys[-1]), textcoords="offset points",
                    xytext=(-4, 12), ha="right", color=t["primary"], fontsize=9.5,
                    fontweight="bold")
        ax.set_ylim(*ylim)
        ax.set_xlim(400, 3350)
        ax.set_xticks(steps)
        ax.set_xticklabels([str(s) for s in steps], fontsize=8.5)
        ax.set_title(title, loc="left", color=t["primary"], fontsize=10.5,
                     fontweight="bold", pad=26)
        ax.text(0, 1.035, hint, transform=ax.transAxes, color=t["muted"],
                fontsize=8.5, va="bottom")

    axes[0].set_ylabel("proxy score", color=t["muted"], fontsize=9)
    axes[1].set_xlabel("training step (this run)", color=t["secondary"],
                       fontsize=9.5, labelpad=6)

    handles = [
        plt.Line2D([], [], color=t["s1"], lw=2, marker="o", markersize=5,
                   markerfacecolor=t["surface"], markeredgewidth=1.8,
                   label="this run"),
        plt.Line2D([], [], color=t["s2"], lw=1.5, linestyle=(0, (4, 3)),
                   label="base (the init this run started from)"),
        plt.Line2D([], [], color=t["muted"], lw=1.5, linestyle=(0, (1.5, 2.5)),
                   label="prior run (the v5 finetune it supersedes)"),
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     bbox_to_anchor=(0.5, -0.11), fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["secondary"])

    fig.suptitle("Proxy metrics across the run. Step 3000 is the shipped checkpoint.",
                 x=0.005, y=1.06, ha="left", color=t["primary"], fontsize=12.5,
                 fontweight="bold")
    fig.tight_layout(w_pad=3.0)
    save(fig, "checkpoint-trajectory", mode)


if __name__ == "__main__":
    for m in ("light", "dark"):
        frame_budget(m)
        training_run(m)
        checkpoints(m)
