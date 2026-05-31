#!/usr/bin/env python3
"""
Find a biologically feasible positive coexistence equilibrium for the reduced
three-dimensional two-patch model.

This replaces the earlier simple root solve, which can converge to the
vegetation-only equilibrium (v1,v2,H)=(1,1,0). For Hopf continuation in AUTO-07p
or MATCONT, use the positive coexistence equilibrium printed by this script.

Run, for example:
    python find_equilibrium_coexistence.py --alpha2 0.25 --beta 0.125

Optional custom guesses:
    python find_equilibrium_coexistence.py --alpha2 0.25 --beta 0.125 --guess 0.071 0.104 1.85
"""

from __future__ import annotations

import argparse
import numpy as np
from scipy.optimize import root, least_squares


def params(alpha2_dim: float = 0.25, beta: float = 0.125) -> dict[str, float]:
    # Dimensional defaults matching the previous package/toolkit.
    K1, K2 = 150.0, 100.0
    r, c0, im, vu, b = 0.2, 0.175, 0.75, 10.0, 20.0
    return dict(
        b1=b / K1,
        b2=b / K2,
        eta=im / r,
        c=c0 * im / r,
        alpha1=0.35 / r,
        alpha2=alpha2_dim / r,
        k1=1.0 / K1,
        k2=1.0 / K2,
        mu=0.003 / r,
        beta=float(beta),
        rho1=vu / K1,
        rho2=vu / K2,
    )


def rhs(x: np.ndarray, p: dict[str, float]) -> np.ndarray:
    """Reduced model RHS, state x=[v1,v2,H]. Smooth version for continuation."""
    v1, v2, H = np.asarray(x, dtype=float)

    den1 = p["b1"] + v1 - p["rho1"]
    den2 = p["b2"] + v2 - p["rho2"]

    q1 = (v1 - p["rho1"]) / den1
    q2 = (v2 - p["rho2"]) / den2

    C1 = 1.0 - p["beta"] * q1
    C2 = 1.0 - p["beta"] * q2

    m1 = p["alpha1"] / (p["k1"] + v1)
    m2 = p["alpha2"] / (p["k2"] + v2)
    h = (m2 / (m1 + m2)) * H

    dv1 = v1 * (1.0 - v1) - p["eta"] * q1 * C1 * h
    dv2 = v2 * (1.0 - v2) - p["eta"] * q2 * C2 * (H - h)
    dH = p["c"] * C1 * q1 * h + p["c"] * C2 * q2 * (H - h) - p["mu"] * H

    return np.array([dv1, dv2, dH], dtype=float)


def jacobian_fd(x: np.ndarray, p: dict[str, float], step: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    J = np.zeros((n, n), dtype=float)
    for j in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[j] += step
        xm[j] -= step
        J[:, j] = (rhs(xp, p) - rhs(xm, p)) / (2.0 * step)
    return J


def is_feasible(x: np.ndarray, p: dict[str, float], residual_tol: float = 1e-8) -> bool:
    v1, v2, H = x
    residual = np.linalg.norm(rhs(x, p))
    C1 = 1.0 - p["beta"] * (v1 - p["rho1"]) / (p["b1"] + v1 - p["rho1"])
    C2 = 1.0 - p["beta"] * (v2 - p["rho2"]) / (p["b2"] + v2 - p["rho2"])
    return bool(
        residual < residual_tol
        and p["rho1"] < v1 < 1.0
        and p["rho2"] < v2 < 1.0
        and H > 1e-8
        and C1 > 0.0
        and C2 > 0.0
    )


def hstar(x: np.ndarray, p: dict[str, float]) -> tuple[float, float, float]:
    v1, v2, H = x
    m1 = p["alpha1"] / (p["k1"] + v1)
    m2 = p["alpha2"] / (p["k2"] + v2)
    h = (m2 / (m1 + m2)) * H
    return float(h), float(H - h), float(h / H)


def find_equilibrium(alpha2: float, beta: float, user_guess: list[float] | None = None) -> tuple[np.ndarray, float, bool]:
    p = params(alpha2, beta)

    # Bounds enforce the biologically relevant interior region.
    lower = np.array([p["rho1"] + 1e-8, p["rho2"] + 1e-8, 1e-8], dtype=float)
    upper = np.array([1.0 - 1e-8, 1.0 - 1e-8, 100.0], dtype=float)

    guesses: list[np.ndarray] = []
    if user_guess is not None:
        guesses.append(np.asarray(user_guess, dtype=float))

    # Important: include guesses near the reserve thresholds because the
    # coexistence equilibrium for the baseline parameter set is close to them.
    for v1 in [p["rho1"] + 0.002, 0.071, 0.08, 0.10, 0.15, 0.25, 0.50, 0.80]:
        for v2 in [p["rho2"] + 0.002, 0.104, 0.12, 0.15, 0.25, 0.50, 0.80]:
            for H in [0.05, 0.10, 0.25, 0.50, 1.0, 1.85, 3.0, 6.0]:
                g = np.array([v1, v2, H], dtype=float)
                g = np.minimum(np.maximum(g, lower + 1e-9), upper - 1e-9)
                guesses.append(g)

    candidates: list[tuple[float, np.ndarray, bool]] = []

    for guess in guesses:
        # First use bounded least squares; this prevents convergence to H=0 or v=1.
        ls = least_squares(
            lambda z: rhs(z, p),
            guess,
            bounds=(lower, upper),
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=20000,
        )
        x_ls = ls.x.astype(float)
        res_ls = float(np.linalg.norm(rhs(x_ls, p)))
        candidates.append((res_ls, x_ls, is_feasible(x_ls, p)))

        # Then polish with unconstrained root only if the LS solution is safely interior.
        if np.all(x_ls > lower + 1e-5) and np.all(x_ls < upper - 1e-5):
            rt = root(lambda z: rhs(z, p), x_ls, method="hybr")
            if rt.success:
                x_rt = rt.x.astype(float)
                res_rt = float(np.linalg.norm(rhs(x_rt, p)))
                candidates.append((res_rt, x_rt, is_feasible(x_rt, p)))

    feasible = [(res, x, ok) for (res, x, ok) in candidates if ok]
    if feasible:
        feasible.sort(key=lambda item: item[0])
        return feasible[0][1], feasible[0][0], True

    # Fall back to the smallest residual, but flag infeasibility.
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][0], False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha2", type=float, default=0.25, help="Dimensional alpha2 value")
    ap.add_argument("--beta", type=float, default=0.125, help="Common toxicity parameter beta")
    ap.add_argument("--guess", type=float, nargs=3, default=None, metavar=("v1", "v2", "H"))
    args = ap.parse_args()

    p = params(args.alpha2, args.beta)
    x, residual, feasible = find_equilibrium(args.alpha2, args.beta, args.guess)
    J = jacobian_fd(x, p)
    eigvals = np.linalg.eigvals(J)
    h1, h2, frac = hstar(x, p)

    print("positive coexistence equilibrium found:", feasible)
    print("equilibrium v1 v2 H:")
    print("{:.16g} {:.16g} {:.16g}".format(*x))
    print("residual norm:", "{:.6e}".format(residual))
    print("rho1 rho2:", "{:.16g} {:.16g}".format(p["rho1"], p["rho2"]))
    print("h1 h2 h1/H:")
    print("{:.16g} {:.16g} {:.16g}".format(h1, h2, frac))
    print("Jacobian eigenvalues:")
    for ev in eigvals:
        print("  {:.16g} {:+.16g}i".format(ev.real, ev.imag))
    print("dominant real part:", "{:.6e}".format(np.max(eigvals.real)))

    print("\nAUTO-07p STPNT replacement:")
    print(f"  PAR(1) = {args.alpha2:.16g}D0")
    print(f"  PAR(2) = {args.beta:.16g}D0")
    print(f"  U(1) = {x[0]:.16g}D0")
    print(f"  U(2) = {x[1]:.16g}D0")
    print(f"  U(3) = {x[2]:.16g}D0")

    print("\nMATCONT xeq replacement:")
    print("  xeq = [{:.16g}; {:.16g}; {:.16g}];".format(*x))
    print("  p0  = [{:.16g}; {:.16g}];  % [alpha2_dim; beta]".format(args.alpha2, args.beta))

    if not feasible:
        print("\nWARNING: No feasible positive coexistence equilibrium passed the checks.")
        print("Try a different alpha2/beta pair or inspect the model equations/parameterization.")


if __name__ == "__main__":
    main()
