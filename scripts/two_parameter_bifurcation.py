#!/usr/bin/env python3
"""
Corrected two-parameter bifurcation / stability map for the reduced two-patch
herbivore-vegetation model.

What this fixes relative to the previous plotting routine:
1. It computes the eigenvalue only at a positive biologically feasible coexistence equilibrium.
2. It rejects failed roots and roots with large residuals instead of plotting them.
3. It masks parameter combinations where the coexistence equilibrium does not exist.
4. It overlays the zero contour only where max Re(lambda) genuinely changes sign.
5. It saves both a publication-ready figure and the underlying grid data.

State variables of the reduced system are y = [v1, v2, H].
The model implementation matches the nondimensional code currently used in the
uploaded toolkit/Rmd: h* = (m2/(m1+m2))H with m1=alpha1/(k1+v1),
m2=alpha2/(k2+v2), and toxicity modifier C(v).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import root

np.seterr(all="ignore")


@dataclass
class ParamsDimensional:
    K1: float = 150.0
    K2: float = 100.0
    r: float = 0.2
    c0: float = 0.175
    im: float = 0.75
    vu: float = 10.0
    b: float = 20.0
    mu1: float = 0.06
    G: float = 1.50
    alpha1_dim: float = 0.15


def make_params(dim: ParamsDimensional, alpha2_dim: float, beta: float) -> dict:
    r = dim.r
    return {
        "b1": dim.b / dim.K1,
        "b2": dim.b / dim.K2,
        "eta": dim.im / r,
        "c": dim.c0 * dim.im / r,
        "alpha1": dim.alpha1_dim / r,
        "alpha2": alpha2_dim / r,
        "k1": 1.0 / dim.K1,
        "k2": 1.0 / dim.K2,
        "mu": dim.mu1 / r,
        "beta": float(beta),
        "rho1": dim.vu / dim.K1,
        "rho2": dim.vu / dim.K2,
    }


def toxicity_modifier(v: float, b: float, rho: float, beta: float) -> float:
    """Old nondimensional toxicity modifier, clipped to prevent negative intake."""
    den = max(b + v - rho, 1e-12)
    return max(0.0, 1.0 - beta * (v - rho) / den)


def migration_rates(v1: float, v2: float, p: dict) -> Tuple[float, float]:
    m1 = p["alpha1"] / (p["k1"] + v1)
    m2 = p["alpha2"] / (p["k2"] + v2)
    return m1, m2


def allocation_fraction(v1: float, v2: float, p: dict) -> float:
    m1, m2 = migration_rates(v1, v2, p)
    return m2 / (m1 + m2)


def intake_terms(v1: float, v2: float, p: dict) -> Tuple[float, float]:
    C1 = toxicity_modifier(v1, p["b1"], p["rho1"], p["beta"])
    C2 = toxicity_modifier(v2, p["b2"], p["rho2"], p["beta"])
    den1 = max(p["b1"] + v1 - p["rho1"], 1e-12)
    den2 = max(p["b2"] + v2 - p["rho2"], 1e-12)
    T1 = C1 * (v1 - p["rho1"]) / den1
    T2 = C2 * (v2 - p["rho2"]) / den2
    return T1, T2


def reduced_rhs(y: np.ndarray, p: dict) -> np.ndarray:
    v1, v2, H = np.asarray(y, dtype=float)
    q = allocation_fraction(v1, v2, p)
    T1, T2 = intake_terms(v1, v2, p)
    return np.array(
        [
            v1 * (1.0 - v1) - p["eta"] * T1 * q * H,
            v2 * (1.0 - v2) - p["eta"] * T2 * (1.0 - q) * H,
            H * (p["c"] * (T1 * q + T2 * (1.0 - q)) - p["mu"]),
        ],
        dtype=float,
    )


def numerical_jacobian(y: np.ndarray, p: dict, h: float = 1e-5) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    J = np.zeros((3, 3), dtype=float)
    for k in range(3):
        yp = y.copy()
        ym = y.copy()
        yp[k] += h
        ym[k] -= h
        J[:, k] = (reduced_rhs(yp, p) - reduced_rhs(ym, p)) / (2.0 * h)
    return J


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _to_v(z: float, lower: float, upper: float = 0.999) -> float:
    return lower + (upper - lower) * _sigmoid(np.asarray(z)).item()


def _to_z(v: float, lower: float, upper: float = 0.999) -> float:
    u = np.clip((v - lower) / (upper - lower), 1e-8, 1.0 - 1e-8)
    return float(np.log(u / (1.0 - u)))


def coexistence_residual_z(z: np.ndarray, p: dict) -> np.ndarray:
    """
    Two-equation coexistence condition for v1 and v2.
    H is eliminated analytically using dv1=0 and dv2=0.
    """
    v1 = _to_v(z[0], p["rho1"] + 1e-6)
    v2 = _to_v(z[1], p["rho2"] + 1e-6)
    q = allocation_fraction(v1, v2, p)
    T1, T2 = intake_terms(v1, v2, p)

    A1 = p["eta"] * T1 * q
    A2 = p["eta"] * T2 * (1.0 - q)
    growth_balance = p["c"] * (T1 * q + T2 * (1.0 - q)) - p["mu"]

    if A1 <= 1e-12 or A2 <= 1e-12:
        return np.array([1e2 + growth_balance, 1e2], dtype=float)

    H1 = v1 * (1.0 - v1) / A1
    H2 = v2 * (1.0 - v2) / A2
    return np.array(
        [
            growth_balance / 0.05,
            np.log(max(H1, 1e-12)) - np.log(max(H2, 1e-12)),
        ],
        dtype=float,
    )


def equilibrium_from_z(z: np.ndarray, p: dict) -> np.ndarray:
    v1 = _to_v(z[0], p["rho1"] + 1e-6)
    v2 = _to_v(z[1], p["rho2"] + 1e-6)
    q = allocation_fraction(v1, v2, p)
    T1, T2 = intake_terms(v1, v2, p)
    A1 = p["eta"] * T1 * q
    A2 = p["eta"] * T2 * (1.0 - q)
    if A1 <= 0.0 or A2 <= 0.0:
        return np.array([np.nan, np.nan, np.nan])
    H = 0.5 * (v1 * (1.0 - v1) / A1 + v2 * (1.0 - v2) / A2)
    return np.array([v1, v2, H], dtype=float)


def solve_coexistence_equilibrium(p: dict, z_start: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solve for a feasible coexistence equilibrium and return (y, z, residual_norm)."""
    starts = [z_start]
    for v1, v2 in [(0.20, 0.20), (0.40, 0.40), (0.70, 0.70), (0.20, 0.70), (0.70, 0.20)]:
        starts.append(
            np.array(
                [
                    _to_z(v1, p["rho1"] + 1e-6),
                    _to_z(v2, p["rho2"] + 1e-6),
                ],
                dtype=float,
            )
        )

    best_norm = np.inf
    best_z = starts[0]
    for guess in starts:
        sol = root(
            lambda zz: coexistence_residual_z(zz, p),
            guess,
            method="hybr",
            options={"xtol": 1e-10, "maxfev": 80},
        )
        nrm = float(np.linalg.norm(coexistence_residual_z(sol.x, p)))
        if nrm < best_norm:
            best_norm = nrm
            best_z = sol.x.astype(float)
        if nrm < 1e-6:
            break

    y = equilibrium_from_z(best_z, p)
    rhs_residual = float(np.linalg.norm(reduced_rhs(y, p))) if np.all(np.isfinite(y)) else np.inf
    return y, best_z, rhs_residual


def compute_two_parameter_map(
    alpha2_grid: np.ndarray,
    beta_grid: np.ndarray,
    dim: ParamsDimensional,
    residual_tol: float = 1e-6,
) -> dict:
    lam = np.full((len(beta_grid), len(alpha2_grid)), np.nan, dtype=float)
    valid = np.zeros_like(lam, dtype=bool)
    residual = np.full_like(lam, np.nan, dtype=float)
    eq = np.full((len(beta_grid), len(alpha2_grid), 3), np.nan, dtype=float)

    z_row = None
    for i, beta in enumerate(beta_grid):
        if z_row is None:
            z = np.array([_to_z(0.40, 0.07), _to_z(0.40, 0.10)], dtype=float)
        else:
            z = z_row.copy()

        first_valid_z = None
        for j, alpha2 in enumerate(alpha2_grid):
            p = make_params(dim, float(alpha2), float(beta))
            y, z, res = solve_coexistence_equilibrium(p, z)
            residual[i, j] = res

            feasible = (
                res < residual_tol
                and np.all(np.isfinite(y))
                and y[0] > p["rho1"]
                and y[1] > p["rho2"]
                and y[0] < 1.0
                and y[1] < 1.0
                and y[2] > 0.0
            )

            if feasible:
                eig = np.linalg.eigvals(numerical_jacobian(y, p))
                lam[i, j] = float(np.max(np.real(eig)))
                valid[i, j] = True
                eq[i, j, :] = y
                if first_valid_z is None:
                    first_valid_z = z.copy()

        if first_valid_z is not None:
            z_row = first_valid_z.copy()

    return {
        "alpha2": alpha2_grid,
        "beta": beta_grid,
        "lambda_max": lam,
        "valid": valid,
        "residual": residual,
        "equilibria": eq,
    }


def save_grid_csv(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    alpha2 = result["alpha2"]
    beta = result["beta"]
    lam = result["lambda_max"]
    valid = result["valid"]
    residual = result["residual"]
    eq = result["equilibria"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha2", "beta", "lambda_max", "valid", "v1_eq", "v2_eq", "H_eq", "residual"])
        for i, b in enumerate(beta):
            for j, a in enumerate(alpha2):
                w.writerow([a, b, lam[i, j], int(valid[i, j]), eq[i, j, 0], eq[i, j, 1], eq[i, j, 2], residual[i, j]])


def plot_two_parameter_map(result: dict, out_png: str, out_pdf: str | None = None) -> None:
    alpha2 = result["alpha2"]
    beta = result["beta"]
    lam = result["lambda_max"]

    finite = lam[np.isfinite(lam)]
    if finite.size == 0:
        raise RuntimeError("No valid coexistence equilibria found on the supplied grid.")

    # Use robust symmetric limits so a few extreme values do not dominate the color scale.
    lim = max(abs(np.nanpercentile(finite, 2)), abs(np.nanpercentile(finite, 98)), 1e-4)
    levels = np.linspace(-lim, lim, 41)

    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    ax.set_facecolor("0.92")
    masked = np.ma.masked_invalid(lam)

    cf = ax.contourf(
        alpha2,
        beta,
        masked,
        levels=levels,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
        extend="both",
    )

    if np.nanmin(lam) < 0.0 < np.nanmax(lam):
        ax.contour(alpha2, beta, masked, levels=[0.0], colors="black", linewidths=2.0)
    else:
        ax.text(
            0.98,
            0.04,
            "No zero crossing in valid grid",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
        )

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(r"Dominant eigenvalue, $\max\operatorname{Re}(\lambda)$")

    ax.set_xlabel(r"$\alpha_2$")
    ax.set_ylabel(r"$\beta$")
    ax.grid(alpha=0.20)

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    outdir = "/mnt/data/two_parameter_bifurcation_outputs"
    os.makedirs(outdir, exist_ok=True)

    dim = ParamsDimensional()
    alpha2_grid = np.linspace(0.02, 4.0, 60)
    beta_grid = np.linspace(0.20, 1.50, 60)

    result = compute_two_parameter_map(alpha2_grid, beta_grid, dim)
    np.savez(os.path.join(outdir, "two_par_bifurcation_grid.npz"), **result)
    save_grid_csv(result, os.path.join(outdir, "two_par_bifurcation_grid.csv"))
    plot_two_parameter_map(
        result,
        os.path.join(outdir, "two_par_bifurcation_corrected.png"),
        os.path.join(outdir, "two_par_bifurcation_corrected.pdf"),
    )

    valid_fraction = float(np.mean(result["valid"]))
    finite = result["lambda_max"][np.isfinite(result["lambda_max"])]
    print(f"Valid grid fraction: {valid_fraction:.3f}")
    print(f"lambda_max range on valid grid: {np.nanmin(finite):.6g} to {np.nanmax(finite):.6g}")
    print(f"Positive lambda_max cells: {int(np.sum(finite > 0))}")
    print(f"Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
