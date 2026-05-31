#!/usr/bin/env python3
# ============================================================
# Plot AUTO-07p bifurcation diagrams
# Robust version using str(loadbd(...)) printed AUTO tables
# ============================================================

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from auto import loadbd


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------
outdir = Path("figures")
outdir.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Parse printed AUTO table
# ------------------------------------------------------------
def parse_auto_printed_table(name, kind):
    """
    Parse the human-readable table produced by str(loadbd(name)).

    kind:
      "eq"   equilibrium branch:
             BR PT TY LAB PAR(1) L2-NORM U(1) U(2) U(3)

      "po"   periodic branch:
             BR PT TY LAB PAR(1) L2-NORM MAX U(1) MAX U(2) MAX U(3) PERIOD

      "hc"   Hopf curve:
             BR PT TY LAB PAR(1) L2-NORM U(1) U(2) U(3) PAR(2)
    """

    bd = loadbd(name)
    txt = str(bd)

    rows = []

    for line in txt.splitlines():
        raw = line.rstrip()

        # Data rows begin with two integers: BR PT
        if not re.match(r"^\s*\d+\s+\d+", raw):
            continue

        parts = raw.split()

        if len(parts) < 7:
            continue

        br = int(parts[0])
        pt = int(parts[1])

        # AUTO table can be either:
        # BR PT TY LAB ...
        # BR PT LAB ...
        idx = 2
        ty = ""

        if re.match(r"^[A-Za-z]+$", parts[idx]):
            ty = parts[idx]
            idx += 1

        lab = int(parts[idx])
        idx += 1

        nums = []
        for x in parts[idx:]:
            try:
                nums.append(float(x.replace("D", "E")))
            except Exception:
                pass

        if kind == "eq":
            if len(nums) >= 5:
                rows.append({
                    "BR": br,
                    "PT": pt,
                    "TY": ty,
                    "LAB": lab,
                    "alpha2": nums[0],
                    "L2": nums[1],
                    "v1": nums[2],
                    "v2": nums[3],
                    "H": nums[4],
                })

        elif kind == "po":
            if len(nums) >= 6:
                rows.append({
                    "BR": br,
                    "PT": pt,
                    "TY": ty,
                    "LAB": lab,
                    "alpha2": nums[0],
                    "L2": nums[1],
                    "max_v1": nums[2],
                    "max_v2": nums[3],
                    "max_H": nums[4],
                    "period": nums[5],
                })

        elif kind == "hc":
            if len(nums) >= 6:
                rows.append({
                    "BR": br,
                    "PT": pt,
                    "TY": ty,
                    "LAB": lab,
                    "alpha2": nums[0],
                    "L2": nums[1],
                    "v1": nums[2],
                    "v2": nums[3],
                    "H": nums[4],
                    "beta": nums[5],
                })

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"\nWARNING: No rows parsed from {name}")
        print("Printed AUTO table was:")
        print(txt)

    return df


# ------------------------------------------------------------
# Load AUTO output
# ------------------------------------------------------------
eq_df = parse_auto_printed_table("eq_alpha2_beta_2.0", kind="eq")
po_df = parse_auto_printed_table("po_from_hopf_beta_2.0", kind="po")
hc_df = parse_auto_printed_table("hopf_curve_alpha2_beta", kind="hc")


# ------------------------------------------------------------
# Print diagnostics
# ------------------------------------------------------------
print("\nEquilibrium branch:")
print(eq_df)

print("\nPeriodic branch:")
print(po_df.head())

print("\nHopf curve:")
print(hc_df.head())


# ------------------------------------------------------------
# Save parsed data
# ------------------------------------------------------------
eq_df.to_csv(outdir / "equilibrium_branch_beta_2.csv", index=False)
po_df.to_csv(outdir / "periodic_branch_beta_2.csv", index=False)
hc_df.to_csv(outdir / "hopf_curve_alpha2_beta.csv", index=False)


# ------------------------------------------------------------
# Extract special points
# ------------------------------------------------------------
hb_points = eq_df[eq_df["TY"].str.contains("HB", na=False)]
lp_points = eq_df[eq_df["TY"].str.contains("LP", na=False)]

print("\nDetected Hopf points:")
print(hb_points)

print("\nDetected limit points:")
print(lp_points)


# ------------------------------------------------------------
# Fallback manual HB/LP values if parsing misses them
# ------------------------------------------------------------
if hb_points.empty:
    hb_points = pd.DataFrame([{
        "BR": 1,
        "PT": 46,
        "TY": "HB",
        "LAB": 7,
        "alpha2": 3.5002010469e-2,
        "L2": 2.22879,
        "v1": 0.11087396,
        "v2": 0.10253171,
        "H": 2.22366528,
    }])

if lp_points.empty:
    lp_points = pd.DataFrame([{
        "BR": 1,
        "PT": 50,
        "TY": "LP",
        "LAB": 8,
        "alpha2": 3.44216e-2,
        "L2": 2.29758,
        "v1": 0.118572,
        "v2": 0.102463,
        "H": 2.29223,
    }])


# ------------------------------------------------------------
# Figure A: equilibrium branch H* versus alpha2
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 5.4))

ax.plot(
    eq_df["alpha2"],
    eq_df["H"],
    marker="o",
    markersize=4,
    linewidth=1.8,
    label=r"Equilibrium $H^*$"
)

ax.scatter(
    hb_points["alpha2"],
    hb_points["H"],
    s=150,
    marker="*",
    edgecolor="black",
    zorder=5,
    label="Hopf point"
)

ax.scatter(
    lp_points["alpha2"],
    lp_points["H"],
    s=95,
    marker="s",
    edgecolor="black",
    zorder=5,
    label="Limit point"
)

ax.set_xlabel(r"Migration parameter $\alpha_2$")
ax.set_ylabel(r"Equilibrium herbivore biomass $H^*$")
ax.set_title(r"Equilibrium branch for $\beta=2.0$")
ax.legend(frameon=False)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(outdir / "equilibrium_branch_H_vs_alpha2.png", dpi=300)
plt.savefig(outdir / "equilibrium_branch_H_vs_alpha2.pdf")
plt.close()


# ------------------------------------------------------------
# Figure B: periodic branch max H versus alpha2
# ------------------------------------------------------------
if not po_df.empty:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    ax.plot(
        po_df["alpha2"],
        po_df["max_H"],
        marker="o",
        markersize=4,
        linewidth=1.8,
        label=r"Periodic branch, $\max H(t)$"
    )

    ax.scatter(
        hb_points["alpha2"],
        hb_points["H"],
        s=150,
        marker="*",
        edgecolor="black",
        zorder=5,
        label="Hopf point"
    )

    ax.set_xlabel(r"Migration parameter $\alpha_2$")
    ax.set_ylabel(r"Maximum herbivore biomass, $\max H(t)$")
    ax.set_title(r"Periodic-orbit branch from Hopf point")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(outdir / "periodic_branch_maxH_vs_alpha2.png", dpi=300)
    plt.savefig(outdir / "periodic_branch_maxH_vs_alpha2.pdf")
    plt.close()


# ------------------------------------------------------------
# Figure C: period versus alpha2
# ------------------------------------------------------------
if not po_df.empty:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    valid_period = np.isfinite(po_df["period"]) & (po_df["period"] > 0)

    ax.plot(
        po_df.loc[valid_period, "alpha2"],
        po_df.loc[valid_period, "period"],
        marker="o",
        markersize=4,
        linewidth=1.8
    )

    ax.set_xlabel(r"Migration parameter $\alpha_2$")
    ax.set_ylabel(r"Period $T$")
    ax.set_yscale("log")
    ax.set_title(r"Growth of period along periodic branch")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(outdir / "period_vs_alpha2.png", dpi=300)
    plt.savefig(outdir / "period_vs_alpha2.pdf")
    plt.close()


# ------------------------------------------------------------
# Figure D: Hopf curve in the (alpha2, beta) plane
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 5.4))

ax.plot(
    hc_df["alpha2"],
    hc_df["beta"],
    marker="o",
    markersize=4,
    linewidth=1.8,
    label="Hopf curve"
)

ax.scatter(
    hb_points["alpha2"].iloc[0],
    2.0,
    s=150,
    marker="*",
    edgecolor="black",
    zorder=5,
    label=r"Initial HB, $\beta=2.0$"
)

ax.set_xlabel(r"Migration parameter $\alpha_2$")
ax.set_ylabel(r"Toxicity parameter $\beta$")
ax.set_title(r"Two-parameter Hopf curve")
ax.legend(frameon=False)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(outdir / "hopf_curve_alpha2_beta.png", dpi=300)
plt.savefig(outdir / "hopf_curve_alpha2_beta.pdf")
plt.close()


# ------------------------------------------------------------
# Combined 2x2 figure
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# A
axes[0, 0].plot(eq_df["alpha2"], eq_df["H"], marker="o", markersize=3, linewidth=1.6)
axes[0, 0].scatter(hb_points["alpha2"], hb_points["H"], s=120, marker="*", edgecolor="black", zorder=5)
axes[0, 0].scatter(lp_points["alpha2"], lp_points["H"], s=80, marker="s", edgecolor="black", zorder=5)
axes[0, 0].set_xlabel(r"$\alpha_2$")
axes[0, 0].set_ylabel(r"$H^*$")
axes[0, 0].set_title("(A) Equilibrium branch")
axes[0, 0].grid(alpha=0.25)

# B
if not po_df.empty:
    axes[0, 1].plot(po_df["alpha2"], po_df["max_H"], marker="o", markersize=3, linewidth=1.6)
axes[0, 1].scatter(hb_points["alpha2"], hb_points["H"], s=120, marker="*", edgecolor="black", zorder=5)
axes[0, 1].set_xlabel(r"$\alpha_2$")
axes[0, 1].set_ylabel(r"$\max H(t)$")
axes[0, 1].set_title("(B) Periodic branch")
axes[0, 1].grid(alpha=0.25)

# C
if not po_df.empty:
    valid_period = np.isfinite(po_df["period"]) & (po_df["period"] > 0)
    axes[1, 0].plot(
        po_df.loc[valid_period, "alpha2"],
        po_df.loc[valid_period, "period"],
        marker="o",
        markersize=3,
        linewidth=1.6
    )
axes[1, 0].set_yscale("log")
axes[1, 0].set_xlabel(r"$\alpha_2$")
axes[1, 0].set_ylabel(r"Period $T$")
axes[1, 0].set_title("(C) Period growth")
axes[1, 0].grid(alpha=0.25)

# D
axes[1, 1].plot(hc_df["alpha2"], hc_df["beta"], marker="o", markersize=3, linewidth=1.6)
axes[1, 1].scatter(hb_points["alpha2"].iloc[0], 2.0, s=120, marker="*", edgecolor="black", zorder=5)
axes[1, 1].set_xlabel(r"$\alpha_2$")
axes[1, 1].set_ylabel(r"$\beta$")
axes[1, 1].set_title(r"(D) Hopf curve")
axes[1, 1].grid(alpha=0.25)

plt.tight_layout()
plt.savefig(outdir / "auto_hopf_bifurcation_combined.png", dpi=300)
plt.savefig(outdir / "auto_hopf_bifurcation_combined.pdf")
plt.close()


print("\nSaved figures in:")
print(outdir.resolve())

print("\nCreated:")
print("  equilibrium_branch_H_vs_alpha2.pdf/png")
print("  periodic_branch_maxH_vs_alpha2.pdf/png")
print("  period_vs_alpha2.pdf/png")
print("  hopf_curve_alpha2_beta.pdf/png")
print("  auto_hopf_bifurcation_combined.pdf/png")