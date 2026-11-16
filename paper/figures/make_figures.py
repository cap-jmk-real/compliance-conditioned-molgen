"""Generate the manuscript figures from the committed result file.

Reads uv/results/eval_results.json (the authoritative evaluation artifact) and
writes two vector figures:
  fig_metrics.pdf  - per-arm bar chart of the evaluation metrics
  fig_ci.pdf       - in-band fraction per arm and the bootstrap CI on the difference

Every plotted value comes from the result file. Nothing is hard-coded.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RESULTS = os.path.join(REPO, "uv", "results", "eval_results.json")

# Okabe-Ito colourblind-safe palette
PLAIN = "#0072B2"
STEERED = "#D55E00"

with open(RESULTS) as fh:
    R = json.load(fh)

plain = R["arms"]["plain"]
steered = R["arms"]["steered"]
boot = R["bootstrap_in_band_diff"]
band_lo, band_hi = R["band_lo"], R["band_hi"]

# ---- Figure 1: per-arm metrics bar chart -----------------------------------
metrics = [
    ("validity", "Validity"),
    ("uniqueness", "Uniqueness"),
    ("novelty", "Novelty"),
    ("rule_pass_rate", "Rule-screen\npass rate"),
    ("in_band_fraction", "In-band\nfraction"),
]
labels = [m[1] for m in metrics]
plain_vals = [plain[m[0]] for m in metrics]
steered_vals = [steered[m[0]] for m in metrics]

x = np.arange(len(labels))
w = 0.38
fig, ax = plt.subplots(figsize=(6.6, 3.5))
b1 = ax.bar(x - w / 2, plain_vals, w, label="Plain", color=PLAIN)
b2 = ax.bar(x + w / 2, steered_vals, w, label="Steered", color=STEERED)
ax.set_ylabel("Value (fraction)")
ax.set_ylim(0, 1.08)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.legend(frameon=False, loc="upper right")
for bars in (b1, b2):
    for rect in bars:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=6.5,
                    xytext=(0, 1), textcoords="offset points")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_metrics.pdf"))
plt.close(fig)

# ---- Figure 2: in-band fraction + bootstrap CI on the difference -----------
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 3.2),
                               gridspec_kw={"width_ratios": [1, 1.1]})

# left: in-band fraction per arm
arms = ["Plain", "Steered"]
vals = [plain["in_band_fraction"], steered["in_band_fraction"]]
cols = [PLAIN, STEERED]
axa.bar(arms, vals, color=cols, width=0.6)
axa.set_ylabel(f"Fraction in {band_lo:.0f}-{band_hi:.0f} nm band")
axa.set_ylim(0, max(vals) * 1.3)
for i, v in enumerate(vals):
    axa.annotate(f"{v:.4f}", (i, v), ha="center", va="bottom", fontsize=8,
                 xytext=(0, 1), textcoords="offset points")
axa.spines[["top", "right"]].set_visible(False)
axa.set_title("In-band fraction per arm", fontsize=9)

# right: difference with bootstrap CI
point = boot["point_estimate"]
lo, hi = boot["ci_low"], boot["ci_high"]
axb.errorbar([point], [0], xerr=[[point - lo], [hi - point]], fmt="o",
             color="#000000", capsize=5, markersize=6)
axb.axvline(0, color="#999999", linestyle="--", linewidth=1)
axb.set_yticks([])
axb.set_xlabel("Steered minus plain in-band fraction")
axb.set_title(f"Difference, {int(boot['ci_level']*100)}% bootstrap CI "
              f"({boot['n_bootstrap']} resamples)", fontsize=9)
axb.annotate(f"{point:.4f}\n[{lo:.4f}, {hi:.4f}]", (point, 0),
             ha="center", va="bottom", fontsize=8,
             xytext=(0, 10), textcoords="offset points")
pad = (hi - lo) * 0.8 + 0.01
axb.set_xlim(min(0, lo) - pad, hi + pad)
axb.spines[["top", "right", "left"]].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_ci.pdf"))
plt.close(fig)

print("wrote fig_metrics.pdf and fig_ci.pdf")
print(f"in-band plain={plain['in_band_fraction']} steered={steered['in_band_fraction']} "
      f"diff={point} CI=[{lo},{hi}]")
