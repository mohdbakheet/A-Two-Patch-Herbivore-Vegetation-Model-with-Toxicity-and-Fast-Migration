# two_patch_toolkit.py
"""
Executive Summary (embedded report)
==================================
This module is a ready-to-run Python (SciPy/NumPy/Matplotlib) analysis toolkit for a two-patch
herbivore–vegetation model with fast migration and plant toxicity. It supports:
  • Full model with explicit fast time-scale parameter epsilon (ε) and an explicit fast allocation variable h.
  • Reduced M0 model obtained by substituting the fast equilibrium h*(v1,v2,H) into the slow dynamics.
  • Fast-subsystem diagnostics: evaluate h*(v1,v2,H) and the leading fast Jacobian eigenvalue across a user-specified state region.
  • Full vs. reduced comparisons across ε values: L_inf and L2 errors for v1,v2,H + tail stats (mean, amplitude, period).
  • Equilibrium continuation over alpha2 (pseudo-arclength predictor-corrector fallback in pure Python).
  • Hopf detection/refinement, best-effort first Lyapunov coefficient at Hopf.
  • Periodic orbit continuation (shooting + variational equations) to extract period, amplitude, Floquet multipliers.
  • Two-parameter stability map (beta × alpha2) heatmap (lambda_max) + approximate Hopf curve overlay.
  • Patch occupancy statistics from simulations and periodic orbits (time averages, min/max, cycle averages).
  • Reproducibility outputs: CSV tables and publication-ready PNG figures; environment JSON.

Model summary (as implemented)
-----------------------------
State variables:
  Reduced model: y = [v1, v2, H]
  Full model:    y = [v1, v2, H, h], where h is herbivores in patch 1 (H-h in patch 2)

Vegetation: logistic growth each patch.
Intake: Holling type II-like terms of the form (v-rho)/(b+v-rho), multiplied by toxicity modifier C(v).
Toxicity modifier (as in provided reduced code):
  C(v) = 1 - beta*(v-rho)/(b + v - rho)

Attractiveness/movement rates (consistent with your provided reduced model):
  m1 = alpha1/(k1+v1)
  m2 = alpha2/(k2+v2)

Fast equilibrium allocation used for reduction:
  h*(v1,v2,H) = (m2/(m1+m2))*H

Fast dynamics (full model) consistent with the above equilibrium:
  dh/dt = (m2*H - (m1+m2)*h)/epsilon
Thus (fast time τ=t/ε): dh/dτ = m2*H - (m1+m2)*h and d(dh/dτ)/dh = -(m1+m2) < 0.

Unspecified details
-------------------
• If your manuscript defines “migration propensity” differently (e.g., an explicit migration rate parameter not embedded
  in m1,m2 or not equal to alpha parameters), adjust the fast equation and h* accordingly (both must match).

Primary/official method sources (URLs inside code comments)
----------------------------------------------------------
SciPy IVP solvers and LSODA:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.LSODA.html
SciPy root finding:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
SciPy peak detection:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
AUTO-07p, MatCont, BifurcationKit references are written by external_tools_bridge.py.

Outputs created (by run_analysis.py)
------------------------------------
CSV tables requested:
  • error_vs_epsilon.csv
  • hopf_thresholds_vs_beta.csv
  • periodic_orbit_amp_period_vs_alpha2.csv  (combined across beta)
Additional CSVs:
  • equilibrium_branch_beta_*.csv
  • stability_map_lambda_max.csv
  • fast_subsystem_scan.csv
PNGs:
  • full_vs_reduced_timeseries_*.png
  • bifurcation_branch_beta_*.png
  • periodic_orbit_amp_period_beta_*.png
  • stability_map_lambda_max.png
  • occupancy_vs_alpha2_beta_*.png
A Mermaid workflow file:
  • workflow_timeline.md

Implementation notes
--------------------
• This toolkit is designed to be robust without external continuation software.
• Optional external tool detection and switching notes are implemented in external_tools_bridge.py.

"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.signal import find_peaks

Array = np.ndarray
np.seterr(all="ignore")

# =============================================================================
# Configuration and I/O utilities
# =============================================================================

@dataclass
class SolverOptions:
    """
    Safer default solver settings.
    """
    method: str = "RK45"   # more stable than LSODA for shooting methods
    rtol: float = 1e-7
    atol: float = 1e-9
    max_step: float = 0.25
    dense_output: bool = False


@dataclass
class DefaultParams:
    """
    Default dimensional parameters (from your provided code).
    These are converted to nondimensional parameters in make_params().
    """
    K1: float = 150.0
    K2: float = 100.0
    r: float = 0.2
    c0: float = 0.175
    im: float = 0.75
    vu: float = 10.0
    b: float = 20.0

    mu1: float = 0.003
    G: float = 1.10
    alpha1_dim: float = 0.35


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(header))
        for r in rows:
            w.writerow(list(r))


def write_json(path: str, obj: object) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    def _default(o):
        if hasattr(o, "__dict__"):
            return o.__dict__
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_default)


def savefig(path: str, dpi: int = 250) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


# =============================================================================
# Model definitions
# =============================================================================

def make_params(
    dim: DefaultParams,
    alpha2_dim: float,
    *,
    beta_override: Optional[float] = None,
    eps: Optional[float] = None,
) -> Dict[str, float]:
    """
    Construct nondimensional parameters
    """
    K1, K2 = dim.K1, dim.K2
    r, c0, im, vu, b = dim.r, dim.c0, dim.im, dim.vu, dim.b

    b1 = b / K1
    b2 = b / K2
    eta = im / r
    c = c0 * im / r
    alpha1 = dim.alpha1_dim / r
    alpha2 = alpha2_dim / r
    k1 = 1.0 / K1
    k2 = 1.0 / K2
    mu = dim.mu1 / r

    beta = (im / (4.0 * dim.G)) if beta_override is None else float(beta_override)
    rho1 = vu / K1
    rho2 = vu / K2

    p = dict(b1=b1, b2=b2, eta=eta, c=c, alpha1=alpha1, alpha2=alpha2, k1=k1, k2=k2, mu=mu, beta=beta, rho1=rho1, rho2=rho2)
    if eps is not None:
        p["eps"] = float(eps)
    return p


def Cfun(v: float, b_: float, rho: float, beta: float) -> float:
    """
    Toxicity modifier with numerical safeguards.
    """
    denom = b_ + v - rho

    if denom <= 1e-12:
        denom = 1e-12

    val = 1.0 - beta * (v - rho) / denom

    # prevent negative grazing modifier
    return max(val, 0.0)

def m_rates(v1: float, v2: float, p: Dict[str, float]) -> Tuple[float, float]:
    """
    Attractiveness/migration rate terms.
    """
    m1 = p["alpha1"] / (p["k1"] + v1)
    m2 = p["alpha2"] / (p["k2"] + v2)
    return float(m1), float(m2)


def hstar(v1: float, v2: float, H: float, p: Dict[str, float]) -> float:
    """
    Fast equilibrium allocation used by your reduced M0 model.
    """
    m1, m2 = m_rates(v1, v2, p)
    d = m1 + m2
    if d <= 0:
        return 0.5 * H
    return (m2 / d) * H


def reduced_M0_rhs(t: float, y: Array, p: Dict[str, float]) -> Array:
    """
    Reduced model on M0, state y=[v1,v2,H].
    """
    v1 = max(float(y[0]), 1e-10)
    v2 = max(float(y[1]), 1e-10)
    H  = max(float(y[2]), 1e-10)
    hs = hstar(v1, v2, H, p)
    C1 = Cfun(v1, p["b1"], p["rho1"], p["beta"])
    C2 = Cfun(v2, p["b2"], p["rho2"], p["beta"])

    den1 = p["b1"] + v1 - p["rho1"]
    den2 = p["b2"] + v2 - p["rho2"]

    den1 = max(den1, 1e-12)
    den2 = max(den2, 1e-12)

    dv1 = v1*(1-v1) - p["eta"]*(v1-p["rho1"])*hs/den1*C1
    dv2 = v2*(1-v2) - p["eta"]*(v2-p["rho2"])*(H-hs)/den2*C2

    dH = (
    C1*p["c"]*(v1-p["rho1"])*hs/den1
    + C2*p["c"]*(v2-p["rho2"])*(H-hs)/den2
    - p["mu"]*H)
    
    return np.array([dv1, dv2, dH], dtype=float)


def full_rhs(t: float, y: Array, p: Dict[str, float]) -> Array:
    """
    Full model with explicit epsilon, state y=[v1,v2,H,h].
    """
    v1 = max(float(y[0]),1e-10)
    v2 = max(float(y[1]),1e-10)
    H  = max(float(y[2]),1e-10)
    h  = max(float(y[3]),1e-10)
    eps = float(p["eps"])
    C1 = Cfun(v1, p["b1"], p["rho1"], p["beta"])
    C2 = Cfun(v2, p["b2"], p["rho2"], p["beta"])

    dv1 = v1 * (1.0 - v1) - p["eta"] * (v1 - p["rho1"]) * h / (p["b1"] + v1 - p["rho1"]) * C1
    dv2 = v2 * (1.0 - v2) - p["eta"] * (v2 - p["rho2"]) * (H - h) / (p["b2"] + v2 - p["rho2"]) * C2
    dH = (
        C1 * p["c"] * (v1 - p["rho1"]) * h / (p["b1"] + v1 - p["rho1"])
        + C2 * p["c"] * (v2 - p["rho2"]) * (H - h) / (p["b2"] + v2 - p["rho2"])
        - p["mu"] * H
    )

    m1, m2 = m_rates(v1, v2, p)
    dh = (m2 * H - (m1 + m2) * h) / eps
    return np.array([dv1, dv2, dH, dh], dtype=float)


def equilibrium_fun(y: Array, p: Dict[str, float]) -> Array:
    """
    Equilibrium condition for reduced model: f(y)=0.
    """
    return reduced_M0_rhs(0.0, y, p)


# =============================================================================
# Jacobians and numerical derivatives
# =============================================================================

def numerical_jacobian(fun: Callable[[Array, Dict[str, float]], Array], x: Array, p: Dict[str, float], h: float = 1e-7) -> Array:
    """
    Centered finite-difference Jacobian.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    f0 = np.asarray(fun(x, p), dtype=float)
    m = f0.size
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        xp = x.copy(); xm = x.copy()
        xp[j] += h; xm[j] -= h
        fp = np.asarray(fun(xp, p), dtype=float)
        fm = np.asarray(fun(xm, p), dtype=float)
        J[:, j] = (fp - fm) / (2.0 * h)
    return J


def complex_step_jacobian(fun: Callable[[Array, Dict[str, float]], Array], x: Array, p: Dict[str, float], h: float = 1e-20) -> Array:
    """
    Complex-step Jacobian (when fun supports complex inputs).
    """
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(fun(x, p))
    m, n = f0.size, x.size
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        xc = x.astype(complex)
        xc[j] += 1j * h
        fc = np.asarray(fun(xc, p))
        J[:, j] = np.imag(fc) / h
    return J


# =============================================================================
# Fast subsystem diagnostics
# =============================================================================

def scan_fast_subsystem(
    dim: DefaultParams,
    alpha2_dim: float,
    beta_val: float,
    region: Tuple[Tuple[float, float, int], Tuple[float, float, int], Tuple[float, float, int]],
    outdir: str,
    dpi: int = 250,
) -> Dict[str, float]:
    """
    Evaluate h*(v1,v2,H) and leading fast eigenvalue over region.

    Leading fast eigenvalue (fast time τ=t/ε): λ_fast = -(m1+m2) (scalar here).
    """
    ensure_dir(outdir)
    p = make_params(dim, alpha2_dim, beta_override=beta_val)

    (v1a, v1b, n1), (v2a, v2b, n2), (Ha, Hb, nH) = region
    v1g = np.linspace(v1a, v1b, int(n1))
    v2g = np.linspace(v2a, v2b, int(n2))
    Hg = np.linspace(Ha, Hb, int(nH))

    rows = []
    lam_list = []
    for H in Hg:
        for v1 in v1g:
            for v2 in v2g:
                hs = hstar(float(v1), float(v2), float(H), p)
                m1, m2 = m_rates(float(v1), float(v2), p)
                #lam_fast = -(m1 + m2)  # fast-time Jacobian eigenvalue
                lam_fast = -max(m1 + m2, 1e-12)
                rows.append([float(v1), float(v2), float(H), float(hs), float(lam_fast)])
                lam_list.append(lam_fast)

    write_csv(os.path.join(outdir, "fast_subsystem_scan.csv"),
              ["v1", "v2", "H", "h_star", "lambda_fast_tau_scale"], rows)

    lam_arr = np.asarray(lam_list, dtype=float)
    spectral_gap = float(np.min(np.abs(lam_arr[np.isfinite(lam_arr)]))) if np.any(np.isfinite(lam_arr)) else float("nan")

    # Heatmaps at mid H: lambda_fast and occupancy fraction h*/H
    Hmid = float(Hg[len(Hg) // 2])
    lam_map = np.zeros((len(v1g), len(v2g)), dtype=float)
    frac_map = np.zeros((len(v1g), len(v2g)), dtype=float)
    for i, v1 in enumerate(v1g):
        for j, v2 in enumerate(v2g):
            hs = hstar(float(v1), float(v2), Hmid, p)
            m1, m2 = m_rates(float(v1), float(v2), p)
            lam_map[i, j] = -(m1 + m2)
            frac_map[i, j] = hs / Hmid if Hmid > 0 else np.nan

    plt.figure(figsize=(8, 6))
    plt.imshow(lam_map, origin="lower", aspect="auto",
               extent=(v2a, v2b, v1a, v1b))
    plt.colorbar(label=r"$\lambda_{fast}$ (tau scale)")
    plt.xlabel("v2"); plt.ylabel("v1")
    plt.title(f"Fast eigenvalue (tau scale), H={Hmid:.3g}, beta={beta_val}, alpha2={alpha2_dim}")
    savefig(os.path.join(outdir, "fast_lambda_heatmap.png"), dpi=dpi)

    plt.figure(figsize=(8, 6))
    plt.imshow(frac_map, origin="lower", aspect="auto", vmin=0.0, vmax=1.0,
               extent=(v2a, v2b, v1a, v1b))
    plt.colorbar(label=r"$h^*/H$")
    plt.xlabel("v2"); plt.ylabel("v1")
    plt.title(f"Fast equilibrium occupancy fraction, H={Hmid:.3g}, beta={beta_val}, alpha2={alpha2_dim}")
    savefig(os.path.join(outdir, "fast_fraction_heatmap.png"), dpi=dpi)

    summary = dict(beta=float(beta_val), alpha2_dim=float(alpha2_dim), H_mid=Hmid, spectral_gap_min_abs_lambda_fast=spectral_gap)
    write_json(os.path.join(outdir, "fast_subsystem_summary.json"), summary)
    return summary


# =============================================================================
# Simulation utilities, tail statistics, error norms
# =============================================================================

def integrate(rhs: Callable[[float, Array, Dict[str, float]], Array], y0: Array, p: Dict[str, float], t_eval: Array, sol: SolverOptions) -> solve_ivp:
    """
    Integrate with solve_ivp. SciPy docs:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    """
    return solve_ivp(
        fun=lambda t, y: rhs(t, y, p),
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=np.asarray(y0, dtype=float),
        t_eval=np.asarray(t_eval, dtype=float),
        method=sol.method,
        rtol=sol.rtol,
        atol=sol.atol,
        max_step=sol.max_step,
        dense_output=sol.dense_output,
    )


def tail_stats(t: Array, x: Array, tcut: float) -> Dict[str, float]:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    m = t > float(tcut)
    if np.sum(m) < 10:
        return dict(mean=np.nan, min=np.nan, max=np.nan, std=np.nan, n=int(np.sum(m)))
    xt = x[m]
    return dict(mean=float(np.mean(xt)), min=float(np.min(xt)), max=float(np.max(xt)), std=float(np.std(xt)), n=int(xt.size))


def estimate_period_from_peaks(t: Array, x: Array, tcut: float) -> float:
    """
    Approximate period from peak-to-peak spacing using SciPy find_peaks:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    m = t > float(tcut)
    if np.sum(m) < 50:
        return float("nan")
    tt = t[m]; xx = x[m]
    prom = 0.25 * float(np.std(xx)) if float(np.std(xx)) > 0 else 0.0
    peaks, _ = find_peaks(xx, prominence=prom)
    if peaks.size < 2:
        return float("nan")
    return float(np.median(np.diff(tt[peaks])))


def linf_l2(t: Array, a: Array, b: Array) -> Tuple[float, float]:
    t = np.asarray(t, dtype=float)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    Linf = float(np.max(np.abs(d)))
    T = float(t[-1] - t[0]) if t.size > 1 else 1.0
    L2 = float(np.sqrt(np.trapz(d * d, t) / T)) if T > 0 else float(np.sqrt(np.trapz(d * d, t)))
    return Linf, L2


def long_run_envelope(
    rhs: Callable[[float, Array, Dict[str, float]], Array],
    y0: Array,
    p: Dict[str, float],
    *,
    tmax: float,
    tcut: float,
    n_eval: int,
    sol: SolverOptions,
) -> Dict[str, object]:
    """
    Compute long-run envelope (tail min/max/mean) by time simulation.
    """
    t_eval = np.linspace(0.0, float(tmax), int(n_eval))
    soln = integrate(rhs, y0, p, t_eval, sol)
    if not soln.success:
        return dict(success=False, message=str(soln.message))
    m = soln.t > float(tcut)
    if np.sum(m) < 10:
        return dict(success=True, t=soln.t, y=soln.y, tail_min=np.full(soln.y.shape[0], np.nan), tail_max=np.full(soln.y.shape[0], np.nan), tail_mean=np.full(soln.y.shape[0], np.nan))
    tail = soln.y[:, m]
    return dict(success=True, t=soln.t, y=soln.y, tail_min=np.min(tail, axis=1), tail_max=np.max(tail, axis=1), tail_mean=np.mean(tail, axis=1))


def full_vs_reduced_comparison(
    dim: DefaultParams,
    alpha2_dim: float,
    beta_val: float,
    eps: float,
    *,
    y0_reduced: Array,
    y0_full: Array,
    tmax: float,
    tcut: float,
    n_eval: int,
    sol_full: SolverOptions,
    sol_reduced: SolverOptions,
    outdir: Optional[str] = None,
    make_plot: bool = True,
    dpi: int = 250,
) -> Dict[str, float]:
    """
    Integrate full and reduced models on a common time grid and compute:
      • L_inf and L2 errors for v1,v2,H
      • Tail stats for H (mean, amplitude, period)
      • Tail occupancy stats (patch-1 fraction)
    """
    t_eval = np.linspace(0.0, float(tmax), int(n_eval))
    pF = make_params(dim, alpha2_dim, beta_override=beta_val, eps=eps)
    pR = make_params(dim, alpha2_dim, beta_override=beta_val)

    solF = integrate(full_rhs, y0_full, pF, t_eval, sol_full)
    solR = integrate(reduced_M0_rhs, y0_reduced, pR, t_eval, sol_reduced)
    if not (solF.success and solR.success):
        return dict(success=0.0, eps=float(eps), alpha2_dim=float(alpha2_dim), beta=float(beta_val))

    v1F, v2F, HF, hF = solF.y
    v1R, v2R, HR = solR.y

    Linf_v1, L2_v1 = linf_l2(solF.t, v1F, v1R)
    Linf_v2, L2_v2 = linf_l2(solF.t, v2F, v2R)
    Linf_H, L2_H = linf_l2(solF.t, HF, HR)

    Hfull = tail_stats(solF.t, HF, tcut)
    Hred = tail_stats(solR.t, HR, tcut)
    per_full = estimate_period_from_peaks(solF.t, HF, tcut)
    per_red = estimate_period_from_peaks(solR.t, HR, tcut)

    frac_full = np.where(HF > 0, hF / HF, np.nan)
    hs_red = np.array([hstar(float(v1R[i]), float(v2R[i]), float(HR[i]), pR) for i in range(HR.size)], dtype=float)
    frac_red = np.where(HR > 0, hs_red / HR, np.nan)
    occ_full = tail_stats(solF.t, frac_full, tcut)
    occ_red = tail_stats(solR.t, frac_red, tcut)

    if outdir is not None and make_plot:
        ensure_dir(outdir)
        plt.figure(figsize=(9, 9))
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(solF.t, v1F, lw=2, label="Full v1")
        ax1.plot(solR.t, v1R, "--", lw=2, label="Reduced v1")
        ax1.set_ylabel("v1"); ax1.legend()

        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(solF.t, v2F, lw=2, label="Full v2")
        ax2.plot(solR.t, v2R, "--", lw=2, label="Reduced v2")
        ax2.set_ylabel("v2"); ax2.legend()

        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(solF.t, HF, lw=2, label="Full H")
        ax3.plot(solR.t, HR, "--", lw=2, label="Reduced H")
        ax3.set_ylabel("H"); ax3.set_xlabel("t"); ax3.legend()

        plt.suptitle(f"Full vs Reduced (beta={beta_val}, alpha2={alpha2_dim}, eps={eps})")
        savefig(os.path.join(outdir, f"full_vs_reduced_timeseries_beta_{beta_val:.3g}_a2_{alpha2_dim:.3g}_eps_{eps:.3g}.png"), dpi=dpi)

    return dict(
        success=1.0,
        beta=float(beta_val),
        alpha2_dim=float(alpha2_dim),
        eps=float(eps),
        Linf_v1=Linf_v1, L2_v1=L2_v1,
        Linf_v2=Linf_v2, L2_v2=L2_v2,
        Linf_H=Linf_H, L2_H=L2_H,
        H_tail_mean_full=Hfull["mean"], H_tail_mean_reduced=Hred["mean"],
        H_tail_amp_full=float(Hfull["max"] - Hfull["min"]), H_tail_amp_reduced=float(Hred["max"] - Hred["min"]),
        H_tail_period_full=float(per_full), H_tail_period_reduced=float(per_red),
        occ_tail_mean_full=occ_full["mean"], occ_tail_mean_reduced=occ_red["mean"],
        occ_tail_min_full=occ_full["min"], occ_tail_max_full=occ_full["max"],
        occ_tail_min_reduced=occ_red["min"], occ_tail_max_reduced=occ_red["max"],
    )


# =============================================================================
# Equilibrium continuation, Hopf detection, Lyapunov coefficient (best-effort)
# =============================================================================

def eigvals_and_lambda_max(y_eq: Array, p: Dict[str, float]) -> Tuple[Array, float]:
    J = numerical_jacobian(equilibrium_fun, y_eq, p)
    eigvals = np.linalg.eigvals(J)
    return eigvals, float(np.max(np.real(eigvals)))


def complex_pair_realpart_and_omega(eigvals: Array, imag_tol: float = 1e-7) -> Tuple[float, float]:
    """
    Return (real_part_of_dominant_complex_eig, |imag_part|) if complex eigenvalues exist; else (nan,nan).
    """
    eigvals = np.asarray(eigvals)
    mask = np.abs(np.imag(eigvals)) > float(imag_tol)
    if not np.any(mask):
        return float("nan"), float("nan")
    cands = eigvals[mask]
    lam = cands[np.argmax(np.real(cands))]
    return float(np.real(lam)), float(abs(np.imag(lam)))


def palc_equilibrium_continuation(
    dim: DefaultParams,
    beta_val: float,
    alpha2_start: float,
    y_guess: Array,
    *,
    ds: float,
    n_steps: int,
    newton_tol: float,
    newton_maxiter: int,
    outdir: Optional[str] = None,
    imag_tol: float = 1e-7,
) -> Dict[str, Array]:
    """
    Pseudo-arclength continuation (PALC) predictor-corrector for equilibria of reduced model over alpha2.

    If external continuation tools are available, use external_tools_bridge.py to document switching.
    This is a pure-Python fallback (robust for many smooth branches, though not as mature as AUTO/MatCont).
    """
    def F(y: Array, a2: float) -> Array:
        p = make_params(dim, a2, beta_override=beta_val)
        return equilibrium_fun(y, p)

    p0 = make_params(dim, alpha2_start, beta_override=beta_val)
    sol0 = root(lambda z: equilibrium_fun(z, p0), np.asarray(y_guess, float), method="hybr")  # SciPy root docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
    if not sol0.success:
        raise RuntimeError(f"Initial equilibrium solve failed: {sol0.message}")
    y0 = sol0.x.astype(float)
    a0 = float(alpha2_start)

    # initial tangent via dy/da approximation:
    Jy = numerical_jacobian(equilibrium_fun, y0, p0)
    da = 1e-6
    p_plus = make_params(dim, a0 + da, beta_override=beta_val)
    p_minus = make_params(dim, a0 - da, beta_override=beta_val)
    dFda = (equilibrium_fun(y0, p_plus) - equilibrium_fun(y0, p_minus)) / (2.0 * da)
    try:
        dyda = -np.linalg.solve(Jy, dFda)
    except np.linalg.LinAlgError:
        dyda = -np.linalg.lstsq(Jy, dFda, rcond=None)[0]

    t_ext = np.concatenate([dyda, np.array([1.0])])
    t_ext = t_ext / np.linalg.norm(t_ext)

    alphas = [a0]
    Ys = [y0.copy()]
    lam_maxs = []
    cpr_list = []
    omega_list = []

    eigvals, lammax = eigvals_and_lambda_max(y0, p0)
    cpr, om = complex_pair_realpart_and_omega(eigvals, imag_tol=imag_tol)
    lam_maxs.append(lammax); cpr_list.append(cpr); omega_list.append(om)

    n = y0.size
    z_prev = np.concatenate([y0, np.array([a0])])

    for _k in range(int(n_steps)):
        z_pred = z_prev + float(ds) * t_ext

        def G(z: Array) -> Array:
            y = z[:n]
            a = float(z[n])
            Fy = F(y, a)
            arc = float(np.dot(t_ext, z - z_pred))
            return np.concatenate([Fy, np.array([arc])])

        z = z_pred.copy()
        converged = False
        for _it in range(int(newton_maxiter)):
            g = G(z)
            if float(np.linalg.norm(g)) < float(newton_tol):
                converged = True
                break
            JG = numerical_jacobian(lambda zz, _p: G(zz), z, p={}, h=1e-7)
            try:
                dz = np.linalg.solve(JG, -g)
            except np.linalg.LinAlgError:
                dz = np.linalg.lstsq(JG, -g, rcond=None)[0]
            z = z + dz

        if not converged:
            break

        y_new = z[:n].astype(float)
        a_new = float(z[n])

        # Update tangent using secant
        t_new = z - z_prev
        nt = float(np.linalg.norm(t_new))
        if nt == 0.0:
            break
        t_new = t_new / nt
        if float(np.dot(t_new, t_ext)) < 0:
            t_new = -t_new
        t_ext = t_new
        z_prev = z

        p_new = make_params(dim, a_new, beta_override=beta_val)
        eigvals, lammax = eigvals_and_lambda_max(y_new, p_new)
        cpr, om = complex_pair_realpart_and_omega(eigvals, imag_tol=imag_tol)

        alphas.append(a_new)
        Ys.append(y_new.copy())
        lam_maxs.append(lammax)
        cpr_list.append(cpr)
        omega_list.append(om)

    out = dict(alpha2=np.array(alphas, float),
               Y=np.array(Ys, float),
               lambda_max=np.array(lam_maxs, float),
               complex_pair_real=np.array(cpr_list, float),
               omega=np.array(omega_list, float))

    if outdir is not None:
        ensure_dir(outdir)
        rows = []
        for i in range(out["alpha2"].size):
            rows.append([float(out["alpha2"][i]), float(out["Y"][i, 0]), float(out["Y"][i, 1]), float(out["Y"][i, 2]),
                         float(out["lambda_max"][i]), float(out["complex_pair_real"][i]), float(out["omega"][i])])
        write_csv(os.path.join(outdir, f"equilibrium_branch_beta_{beta_val:.3g}.csv"),
                  ["alpha2_dim", "v1_eq", "v2_eq", "H_eq", "lambda_max", "complex_pair_real", "omega"], rows)
    return out


def refine_hopf_bisection(
    dim: DefaultParams,
    beta_val: float,
    a_left: float, y_left: Array,
    a_right: float, y_right: Array,
    *,
    imag_tol: float,
    real_tol: float,
    maxiter: int,
) -> Dict[str, float]:
    """
    Refine Hopf by bisection on the real part of the dominant complex eigenvalue.
    """
    def eval_cpr(a2: float, y_guess: Array) -> Tuple[float, float, Array]:
        p = make_params(dim, a2, beta_override=beta_val)
        sol = root(lambda z: equilibrium_fun(z, p), np.asarray(y_guess, float), method="hybr")
        if not sol.success:
            return np.nan, np.nan, np.full_like(y_guess, np.nan)
        y = sol.x.astype(float)
        eigvals, _ = eigvals_and_lambda_max(y, p)
        cpr, om = complex_pair_realpart_and_omega(eigvals, imag_tol=imag_tol)
        return cpr, om, y

    cL, wL, yL = eval_cpr(float(a_left), y_left)
    cR, wR, yR = eval_cpr(float(a_right), y_right)
    if not (np.isfinite(cL) and np.isfinite(cR) and cL * cR < 0):
        return dict(alpha2_dim=np.nan, omega=np.nan, v1_eq=np.nan, v2_eq=np.nan, H_eq=np.nan, complex_pair_real=np.nan)

    lo_a, hi_a = float(a_left), float(a_right)
    lo_c, hi_c = float(cL), float(cR)
    lo_y, hi_y = yL.copy(), yR.copy()
    best = dict(alpha2_dim=np.nan, omega=np.nan, v1_eq=np.nan, v2_eq=np.nan, H_eq=np.nan, complex_pair_real=np.nan)

    for _k in range(int(maxiter)):
        mid_a = 0.5 * (lo_a + hi_a)
        mid_guess = 0.5 * (lo_y + hi_y)
        mid_c, mid_w, mid_y = eval_cpr(mid_a, mid_guess)
        if not np.isfinite(mid_c):
            hi_a, hi_y, hi_c = mid_a, mid_guess, mid_c
            continue
        best = dict(alpha2_dim=float(mid_a), omega=float(mid_w), v1_eq=float(mid_y[0]), v2_eq=float(mid_y[1]), H_eq=float(mid_y[2]), complex_pair_real=float(mid_c))
        if abs(float(mid_c)) < float(real_tol):
            break
        if lo_c * mid_c < 0:
            hi_a, hi_y, hi_c = mid_a, mid_y, mid_c
        else:
            lo_a, lo_y, lo_c = mid_a, mid_y, mid_c

    return best


# ---- Finite-difference multilinear forms for Hopf Lyapunov coefficient (best-effort)
def _B(fun: Callable[[Array, Dict[str, float]], Array], x0: Array, p: Dict[str, float], u: Array, v: Array, h: float = 1e-5) -> Array:
    return (fun(x0 + h*(u+v), p) - fun(x0 + h*(u-v), p) - fun(x0 + h*(-u+v), p) + fun(x0 - h*(u+v), p)) / (4*h*h)

def _C(fun: Callable[[Array, Dict[str, float]], Array], x0: Array, p: Dict[str, float], u: Array, v: Array, w: Array, h: float = 1e-4) -> Array:
    acc = np.zeros_like(fun(x0, p), dtype=float)
    for s1 in (+1.0, -1.0):
        for s2 in (+1.0, -1.0):
            for s3 in (+1.0, -1.0):
                acc += (s1*s2*s3) * fun(x0 + h*(s1*u + s2*v + s3*w), p)
    return acc / (8*h**3)

def first_lyapunov_coefficient_hopf(
    dim: DefaultParams,
    beta_val: float,
    alpha2_dim: float,
    y_eq: Array,
    *,
    imag_tol: float = 1e-7,
) -> float:
    """
    Best-effort numerical approximation of the first Lyapunov coefficient l1 at a Hopf bifurcation.

    Notes:
    • This is a numerically delicate quantity. Treat as “attempt”; verify with external tools (MatCont/AUTO)
      if l1 is a central claim.
    """
    p0 = make_params(dim, alpha2_dim, beta_override=beta_val)
    try:
        A = complex_step_jacobian(equilibrium_fun, y_eq, p0).astype(complex)
    except Exception:
        A = numerical_jacobian(equilibrium_fun, y_eq, p0).astype(complex)

    eigvals, eigvecs = np.linalg.eig(A)
    idx_c = [i for i, lam in enumerate(eigvals) if abs(np.imag(lam)) > imag_tol]
    if not idx_c:
        return float("nan")
    i = min(idx_c, key=lambda j: abs(np.real(eigvals[j])))
    lam = eigvals[i]
    omega = abs(np.imag(lam))
    if omega <= imag_tol:
        return float("nan")
    q = eigvecs[:, i]

    eigvalsT, eigvecsT = np.linalg.eig(A.T)
    j = int(np.argmin(np.abs(eigvalsT - np.conjugate(lam))))
    pvec = eigvecsT[:, j]
    inner = np.vdot(pvec, q)
    if inner == 0:
        return float("nan")
    pvec = pvec / inner

    def f(x: Array, pp: Dict[str, float]) -> Array:
        return equilibrium_fun(x, pp).astype(float)

    def Bc(u: Array, v: Array) -> Array:
        uR, uI = np.real(u), np.imag(u)
        vR, vI = np.real(v), np.imag(v)
        return (_B(f, y_eq, p0, uR, vR) - _B(f, y_eq, p0, uI, vI)) + 1j*(_B(f, y_eq, p0, uR, vI) + _B(f, y_eq, p0, uI, vR))

    def Cc(u: Array, v: Array, w: Array) -> Array:
        uR, uI = np.real(u), np.imag(u)
        vR, vI = np.real(v), np.imag(v)
        wR, wI = np.real(w), np.imag(w)
        # Expand multilinear complex mapping via real/imag components (compact but sufficient)
        total = np.zeros_like(f(y_eq, p0), dtype=complex)
        for uu, au in ((uR, 0), (uI, 1)):
            for vv, av in ((vR, 0), (vI, 1)):
                for ww, aw in ((wR, 0), (wI, 1)):
                    total += (1j)**(au+av+aw) * _C(f, y_eq, p0, uu, vv, ww)
        return total

    I = np.eye(A.shape[0], dtype=complex)
    invA = np.linalg.pinv(A)
    inv2 = np.linalg.pinv(2j*omega*I - A)

    term1 = Cc(q, q, np.conjugate(q))
    u = invA @ Bc(q, np.conjugate(q))
    term2 = 2.0 * Bc(q, u)
    v = inv2 @ Bc(q, q)
    term3 = Bc(np.conjugate(q), v)

    c1 = np.vdot(pvec, term1 - term2 + term3)
    l1 = float(np.real(c1) / (2.0 * omega))
    return l1


# =============================================================================
# Periodic orbit analysis: shooting + variational (Floquet multipliers)
# =============================================================================

def _integrate_state_and_variational(
    dim: DefaultParams,
    beta_val: float,
    alpha2_dim: float,
    x0: Array,
    T: float,
    sol: SolverOptions,
    n_eval: int = 800,
) -> Tuple[Array, Array, Array]:
    """
    Integrate reduced state and fundamental matrix over one period.
    """
    p = make_params(dim, alpha2_dim, beta_override=beta_val)
    x0 = np.asarray(x0, float)
    n = x0.size
    Phi0 = np.eye(n, dtype=float).reshape(-1)
    y0 = np.concatenate([x0, Phi0])
    t_eval = np.linspace(0.0, float(T), int(n_eval))

    def rhs_aug(t: float, y: Array) -> Array:
        x = y[:n]
        Phi = y[n:].reshape((n, n))
        fx = reduced_M0_rhs(t, x, p)
        J = numerical_jacobian(equilibrium_fun, x, p, h=1e-7)
        dPhi = J @ Phi
        return np.concatenate([fx, dPhi.reshape(-1)])

    soln = solve_ivp(rhs_aug, (0.0, float(T)), y0, t_eval=t_eval, method=sol.method, rtol=sol.rtol, atol=sol.atol, max_step=sol.max_step)
    #if not soln.success:
    #    raise RuntimeError(f"Augmented integration failed: {soln.message}")
    if not soln.success:
        return None, None, None
    x_traj = soln.y[:n, :]
    PhiT = soln.y[n:, -1].reshape((n, n))
    return soln.t, x_traj, PhiT


def shooting_newton_periodic_orbit(
    dim: DefaultParams,
    beta_val: float,
    alpha2_dim: float,
    x0_guess: Array,
    T_guess: float,
    phase_ref: Array,
    *,
    sol: SolverOptions,
    tol: float,
    maxiter: int,
    res = _integrate_state_and_variational(...)
    if res[0] is None:
        return dict(success=False)
) -> Dict[str, object]:
    """
    Single-shooting Newton method for periodic orbit:
      residual: x(T; x0) - x0 = 0 plus a phase condition.

    Uses monodromy matrix (variational equation) to form Newton Jacobian.
    Floquet multipliers are eigenvalues of the monodromy matrix.
    """
    p = make_params(dim, alpha2_dim, beta_override=beta_val)
    x0 = np.asarray(x0_guess, float).copy()
    T = float(T_guess)
    if T <= 0 or not np.isfinite(T):
        return dict(success=False)
    phase_ref = np.asarray(phase_ref, float)
    vref = reduced_M0_rhs(0.0, phase_ref, p)
    if float(np.linalg.norm(vref)) == 0.0:
        vref = np.array([1.0, 0.0, 0.0], float)

    n = x0.size
    success = False
    resid_norm = float("nan")
    monodromy = None

    for it in range(int(maxiter)):
        t, x_traj, M = _integrate_state_and_variational(dim, beta_val, alpha2_dim, x0, T, sol, n_eval=800)
        xT = x_traj[:, -1]
        fT = reduced_M0_rhs(T, xT, p)

        R1 = xT - x0
        R2 = float(np.dot(x0 - phase_ref, vref))
        R = np.concatenate([R1, np.array([R2])])
        resid_norm = float(np.linalg.norm(R))
        monodromy = M

        if resid_norm < float(tol):
            success = True
            break

        # Jacobian blocks:
        J11 = M - np.eye(n)
        J12 = fT.reshape((n, 1))         # d(xT-x0)/dT = f(xT)
        J21 = vref.reshape((1, n))       # phase condition derivative wrt x0
        J22 = np.array([[0.0]])
        J = np.block([[J11, J12], [J21, J22]])

        try:
            delta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(J, -R, rcond=None)[0]

        x0 = x0 + delta[:n]
        T = T + float(delta[n])
        if T <= 0:
            T = abs(T) + 1e-6

    floquet = np.linalg.eigvals(monodromy) if monodromy is not None else np.array([np.nan]*n)
    return dict(success=success, x0=x0, T=float(T), monodromy=monodromy, floquet=floquet, residual_norm=float(resid_norm), iterations=int(it+1))


def _periodic_orbit_metrics(
    dim: DefaultParams,
    beta_val: float,
    alpha2_dim: float,
    orbit: Dict[str, object],
    sol: SolverOptions,
) -> Dict[str, float]:
    """
    Compute amplitudes/means and occupancy statistics over one period.
    """
    x0 = np.asarray(orbit["x0"], float)
    T = float(orbit["T"])
    p = make_params(dim, alpha2_dim, beta_override=beta_val)
    t_eval = np.linspace(0.0, T, 800)
    soln = integrate(reduced_M0_rhs, x0, p, t_eval, sol)
    if not soln.success:
        return dict(beta=float(beta_val), alpha2_dim=float(alpha2_dim), T=float(T))

    v1, v2, H = soln.y
    hs = np.array([hstar(float(v1[i]), float(v2[i]), float(H[i]), p) for i in range(H.size)], float)
    occ = np.where(H > 0, hs / H, np.nan)

    return dict(
        beta=float(beta_val), alpha2_dim=float(alpha2_dim), T=float(T),
        v1_min=float(np.min(v1)), v1_max=float(np.max(v1)), v1_amp=float(np.max(v1)-np.min(v1)), v1_mean=float(np.mean(v1)),
        v2_min=float(np.min(v2)), v2_max=float(np.max(v2)), v2_amp=float(np.max(v2)-np.min(v2)), v2_mean=float(np.mean(v2)),
        H_min=float(np.min(H)), H_max=float(np.max(H)), H_amp=float(np.max(H)-np.min(H)), H_mean=float(np.mean(H)),
        occ_mean=float(np.nanmean(occ)), occ_min=float(np.nanmin(occ)), occ_max=float(np.nanmax(occ)),
    )


def continue_periodic_orbits_by_parameter(
    dim: DefaultParams,
    beta_val: float,
    alpha2_hopf: float,
    y_eq_hopf: Array,
    omega: float,
    alpha2_values: Sequence[float],
    *,
    amp0: float,
    sol: SolverOptions,
    tol: float,
    maxiter: int,
    outdir: Optional[str] = None,
    dpi: int = 250,
    if omega <= 1e-6 or not np.isfinite(omega):
        return []
) -> List[Dict[str, object]]:
    """
    Continue periodic orbits from Hopf by simple parameter stepping:
      use previous orbit (x0,T) as initial guess for next alpha2.

    This is a practical fallback when external collocation continuation is not available.
    """
    if omega <= 0 or not np.isfinite(omega):
        return []

    # initial guess around Hopf using eigenvector direction
    pH = make_params(dim, alpha2_hopf, beta_override=beta_val)
    try:
        A = complex_step_jacobian(equilibrium_fun, y_eq_hopf, pH).astype(complex)
    except Exception:
        A = numerical_jacobian(equilibrium_fun, y_eq_hopf, pH).astype(complex)
    eigvals, eigvecs = np.linalg.eig(A)
    idx = int(np.argmin(np.abs(eigvals - 1j*omega)))
    q = eigvecs[:, idx]
    x0_guess = np.asarray(y_eq_hopf, float) + float(amp0) * np.real(q)
    T_guess = float(2.0*np.pi/omega)
    phase_ref = np.asarray(y_eq_hopf, float)

    orbit0 = shooting_newton_periodic_orbit(dim, beta_val, alpha2_hopf, x0_guess, T_guess, phase_ref, sol=sol, tol=tol, maxiter=maxiter)
    if not orbit0["success"]:
        orbit0 = shooting_newton_periodic_orbit(dim, beta_val, alpha2_hopf, np.asarray(y_eq_hopf,float) + 10*float(amp0)*np.real(q), T_guess, phase_ref, sol=sol, tol=tol, maxiter=maxiter)
    if not orbit0["success"]:
        return []

    orbits = [dict(alpha2_dim=float(alpha2_hopf), **orbit0)]
    x_prev = np.asarray(orbit0["x0"], float).copy()
    T_prev = float(orbit0["T"])

    for a2 in alpha2_values:
        a2 = float(a2)
        if abs(a2 - alpha2_hopf) < 1e-12:
            continue
        orb = shooting_newton_periodic_orbit(dim, beta_val, a2, x_prev, T_prev, phase_ref, sol=sol, tol=tol, maxiter=maxiter)
        if not orb["success"]:
            break
        orbits.append(dict(alpha2_dim=a2, **orb))
        x_prev = np.asarray(orb["x0"], float).copy()
        T_prev = float(orb["T"])

    # Write CSV + plot
    metrics = []
    for o in orbits:
        met = _periodic_orbit_metrics(dim, beta_val, float(o["alpha2_dim"]), o, sol)
        floq = np.asarray(o["floquet"])
        if floq.size:
            # remove the trivial multiplier closest to 1
            k1 = int(np.argmin(np.abs(floq - 1.0)))
            others = np.delete(floq, k1) if floq.size > 1 else np.array([], dtype=complex)
            met["max_mod_floquet_excl_1"] = float(np.max(np.abs(others))) if others.size else np.nan
        else:
            met["max_mod_floquet_excl_1"] = np.nan
        metrics.append(met)

    if outdir is not None:
        ensure_dir(outdir)
        write_csv(os.path.join(outdir, f"periodic_orbit_metrics_beta_{beta_val:.3g}.csv"),
                  list(metrics[0].keys()) if metrics else ["beta"], [list(m.values()) for m in metrics])

        a2 = np.array([m["alpha2_dim"] for m in metrics], float)
        H_amp = np.array([m.get("H_amp", np.nan) for m in metrics], float)
        T = np.array([m.get("T", np.nan) for m in metrics], float)

        plt.figure(figsize=(8, 6))
        plt.plot(a2, H_amp, lw=2)
        plt.xlabel("alpha2_dim"); plt.ylabel("H amplitude")
        plt.title(f"Periodic orbit amplitude vs alpha2 (beta={beta_val})")
        savefig(os.path.join(outdir, f"periodic_orbit_amp_beta_{beta_val:.3g}.png"), dpi=dpi)

        plt.figure(figsize=(8, 6))
        plt.plot(a2, T, lw=2)
        plt.xlabel("alpha2_dim"); plt.ylabel("Period T")
        plt.title(f"Periodic orbit period vs alpha2 (beta={beta_val})")
        savefig(os.path.join(outdir, f"periodic_orbit_period_beta_{beta_val:.3g}.png"), dpi=dpi)

    return orbits


# =============================================================================
# Two-parameter stability map beta × alpha2
# =============================================================================

def stability_map_beta_alpha2(
    dim: DefaultParams,
    beta_grid: Sequence[float],
    alpha2_grid: Sequence[float],
    y_guess0: Array,
    *,
    outdir: str,
    dpi: int = 250,
) -> Dict[str, object]:
    """
    Compute equilibrium and lambda_max across (beta, alpha2) grid with warm-start root solves.
    Also compute an approximate Hopf curve: first sign-change index in complex-pair real part along alpha2 per beta.
    """
    ensure_dir(outdir)
    beta_grid = np.asarray(beta_grid, float)
    alpha2_grid = np.asarray(alpha2_grid, float)

    lam_grid = np.full((beta_grid.size, alpha2_grid.size), np.nan, float)
    hopf_curve = []

    y_seed = np.asarray(y_guess0, float).copy()
    for i, beta_val in enumerate(beta_grid):
        y_guess = y_seed.copy()
        cpr_row = np.full(alpha2_grid.size, np.nan, float)
        succ_row = np.zeros(alpha2_grid.size, bool)

        for j, a2 in enumerate(alpha2_grid):
            p = make_params(dim, float(a2), beta_override=float(beta_val))
            sol = root(lambda z: equilibrium_fun(z, p), y_guess, method="hybr")
            if sol.success:
                y_eq = sol.x.astype(float)
                eigvals, lammax = eigvals_and_lambda_max(y_eq, p)
                cpr, _om = complex_pair_realpart_and_omega(eigvals)
                lam_grid[i, j] = lammax
                cpr_row[j] = cpr
                succ_row[j] = True
                y_guess = y_eq.copy()

        # Hopf approx by sign change in cpr
        for j in range(1, alpha2_grid.size):
            if np.isfinite(cpr_row[j-1]) and np.isfinite(cpr_row[j]) and (cpr_row[j-1] * cpr_row[j] < 0):
                hopf_curve.append([float(beta_val), float(alpha2_grid[j])])
                break

        # carry seed forward to speed up next beta (if any success at small alpha2)
        ok = np.where(succ_row)[0]
        if ok.size:
            y_seed = y_guess.copy()

    # Save grid as long CSV
    rows = []
    for i, beta_val in enumerate(beta_grid):
        for j, a2 in enumerate(alpha2_grid):
            rows.append([float(beta_val), float(a2), float(lam_grid[i, j])])
    write_csv(os.path.join(outdir, "stability_map_lambda_max.csv"), ["beta", "alpha2_dim", "lambda_max"], rows)

    # Heatmap PNG
    plt.figure(figsize=(9, 6))
    plt.imshow(lam_grid, origin="lower", aspect="auto",
               extent=(float(alpha2_grid.min()), float(alpha2_grid.max()), float(beta_grid.min()), float(beta_grid.max())))
    plt.colorbar(label=r"$\lambda_{max}$")
    plt.xlabel("alpha2_dim"); plt.ylabel("beta")
    plt.title("Two-parameter stability map (equilibrium)")

    if hopf_curve:
        hc = np.asarray(hopf_curve, float)
        plt.plot(hc[:, 1], hc[:, 0], "w--", lw=2, label="Approx. Hopf curve")
        plt.legend()

    savefig(os.path.join(outdir, "stability_map_lambda_max.png"), dpi=dpi)

    write_csv(os.path.join(outdir, "hopf_curve_approx.csv"), ["beta", "alpha2_dim"], hopf_curve if hopf_curve else [])
    return dict(beta_grid=beta_grid, alpha2_grid=alpha2_grid, lambda_max_grid=lam_grid, hopf_curve=np.asarray(hopf_curve, float))


# =============================================================================
# Occupancy statistics utilities
# =============================================================================

def occupancy_stats_from_time_series(t: Array, v1: Array, v2: Array, H: Array, h_or_hs: Array, tcut: float) -> Dict[str, float]:
    frac = np.where(H > 0, h_or_hs / H, np.nan)
    st = tail_stats(t, frac, tcut)
    return dict(mean=st["mean"], min=st["min"], max=st["max"], std=st["std"], n=st["n"])


def environment_info() -> Dict[str, object]:
    """
    Capture reproducibility metadata (versions + platform). Written by run_analysis.py.
    """
    env = dict(
        python=sys.version,
        platform=platform.platform(),
        executable=sys.executable,
        numpy=np.__version__,
    )
    try:
        import scipy
        env["scipy"] = scipy.__version__
    except Exception:
        env["scipy"] = "unknown"
    try:
        import matplotlib
        env["matplotlib"] = matplotlib.__version__
    except Exception:
        env["matplotlib"] = "unknown"
    return env
# ================================================================
# Simple simulation wrappers (used by external scripts)
# ================================================================

def simulate_full(params, eps, init_state, t_eval):
    """
    Simulate the full model.

    State order expected by this wrapper:
    [H, h1, v1, v2]

    Internally the toolkit uses:
    [v1, v2, H, h]
    """
    # convert state order
    H0, h10, v10, v20 = init_state
    y0 = np.array([v10, v20, H0, h10])

    # convert parameter dict to toolkit format
    p = params.copy()
    p["eps"] = eps

    sol = solve_ivp(
        lambda t, y: full_rhs(t, y, p),
        (t_eval[0], t_eval[-1]),
        y0,
        t_eval=t_eval,
        method="LSODA"
    )

    v1, v2, H, h = sol.y

    # return in user-friendly order
    return np.column_stack([H, h, v1, v2])


def simulate_reduced(params, eps, init_state, t_eval):
    """
    Simulate reduced model.

    Input state order:
    [H, v1, v2]

    Toolkit order:
    [v1, v2, H]
    """
    H0, v10, v20 = init_state
    y0 = np.array([v10, v20, H0])

    p = params.copy()

    sol = solve_ivp(
        lambda t, y: reduced_M0_rhs(t, y, p),
        (t_eval[0], t_eval[-1]),
        y0,
        t_eval=t_eval,
        method="LSODA"
    )

    v1, v2, H = sol.y

    return np.column_stack([H, v1, v2])