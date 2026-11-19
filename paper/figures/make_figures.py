"""Generate the manuscript figures from the committed result files.

Primary result: three training seeds (0, 1, 2) per arm, all trained and evaluated on an
NVIDIA RTX 3060. Inputs:
  uv/results/aggregate_rtx3060_seeds012.json  - output of uv/aggregate_seeds.py
  uv/results/seeds_rtx3060/eval_seed{0,1,2}.json - per-seed evaluation artifacts
  uv/results/eval_results.json                - seed 0 on an NVIDIA A100 (hardware check)
Outputs (vector):
  fig_metrics.pdf - per-arm metrics, mean +/- sd across the three seeds
  fig_ci.pdf      - per-seed steering effect with its within-run bootstrap CI, the
                    seed-level mean with its 95% t-interval, and the A100 seed-0 run

Every plotted value comes from these files. Nothing is hard-coded.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(REPO, "uv", "results")

# Okabe-Ito colourblind-safe palette
PLAIN = "#0072B2"
STEERED = "#D55E00"

AGG = json.load(open(os.path.join(RES, "aggregate_rtx3060_seeds012.json")))
SEEDS = [json.load(open(os.path.join(RES, "seeds_rtx3060", f"eval_seed{s}.json")))
         for s in (0, 1, 2)]
A100 = json.load(open(os.path.join(RES, "eval_results.json")))
band_lo, band_hi = SEEDS[0]["band_lo"], SEEDS[0]["band_hi"]
pm = AGG["per_metric"]

# ---- Figure 1: per-arm metrics, mean +/- sd over 3 seeds ---------------------
metrics = [
    ("validity", "Validity"),
    ("uniqueness", "Uniqueness"),
    ("novelty", "Novelty"),
    ("rule_pass_rate", "Structural-screen\npass rate"),
    ("in_band_fraction", "In-band\nfraction"),
]
labels = [m[1] for m in metrics]
x = np.arange(len(labels))
w = 0.38
fig, ax = plt.subplots(figsize=(6.6, 3.5))
for off, arm, col, lab in ((-w / 2, "plain", PLAIN, "Plain"),
                           (w / 2, "steered", STEERED, "Steered")):
    m = [pm[k][arm]["mean"] for k, _ in metrics]
    s = [pm[k][arm]["sd"] for k, _ in metrics]
    bars = ax.bar(x + off, m, w, yerr=s, capsize=2, label=lab, color=col,
                  error_kw={"elinewidth": 0.8})
    for rect, v in zip(bars, m):
        ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=6.5,
                    xytext=(0, 2), textcoords="offset points")
ax.set_ylabel("Value (unitless fraction)")
ax.set_ylim(0, 1.1)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.legend(frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_metrics.pdf"))
plt.close(fig)

# ---- Figure 2: steering effect per run ---------------------------------------
eff = AGG["steering_effect"]
rows = []
for s, R in zip((0, 1, 2), SEEDS):
    b = R["bootstrap_in_band_diff"]
    rows.append((f"RTX 3060, seed {s}", b["point_estimate"], b["ci_low"], b["ci_high"], "o"))
lo_m, hi_m = eff["ci95_mean_diff"]
rows.append(("RTX 3060, mean of 3 seeds\n(95% t-interval, df=2)", eff["mean_diff"],
             lo_m, hi_m, "D"))
b = A100["bootstrap_in_band_diff"]
rows.append(("A100, seed 0\n(hardware check)", b["point_estimate"], b["ci_low"],
             b["ci_high"], "s"))

fig, ax = plt.subplots(figsize=(6.6, 3.3))
ys = np.arange(len(rows))[::-1]
for y, (lab, p, lo, hi, mk) in zip(ys, rows):
    col = "#000000" if mk != "s" else "#777777"
    ax.errorbar([p], [y], xerr=[[p - lo], [hi - p]], fmt=mk, color=col, capsize=4,
                markersize=6)
    ax.annotate(f"{p:.4f} [{lo:.4f}, {hi:.4f}]", (hi, y), xytext=(6, -3),
                textcoords="offset points", fontsize=7.5)
ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_xlabel(f"Steered minus plain in-band fraction ({band_lo:.0f}-{band_hi:.0f} nm), unitless")
ax.set_xlim(-0.01, max(r[3] for r in rows) + 0.06)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_ci.pdf"))
plt.close(fig)

print("wrote fig_metrics.pdf and fig_ci.pdf")
print(f"3-seed mean diff={eff['mean_diff']:.4f} sd={eff['sd_diff']:.4f} "
      f"t-CI=[{lo_m:.4f},{hi_m:.4f}]")
