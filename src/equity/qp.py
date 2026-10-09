"""A small quadratic-programming solver for the problems of portfolio construction.

    minimise    1/2 x'Px + q'x        subject to    l <= Ax <= u,    lb <= x <= ub

``P`` is positive semi-definite, so a covariance penalty, a transaction-cost term (written on split buy and sell variables) and a linear alpha term all fit. The method is the operator-splitting
scheme of Stellato, Banjac, Goulart, Bemporad and Boyd (OSQP, 2020): an alternating-direction method of multipliers that factors one positive-definite matrix, with Ruiz scaling, an adaptive
step, over-relaxation and a final polish on the active set that returns the high-accuracy answer an interior-point or active-set solver would. It exists because the problems of this package
(hundreds of stocks, dozens of constraints, several periods at once) are far too large for the SLSQP calls used for 15 ETFs, and because nothing else is installed.

Equality constraints are rows with ``l == u``; an infinite bound means no bound. ``QPResult.ok`` is false when the iteration did not converge (the problem is usually infeasible or unbounded) and
the answer must not be used.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgWarning, cho_factor, cho_solve, lu_factor, lu_solve


@dataclass
class QPResult:
    x: np.ndarray
    y: np.ndarray                      # the multipliers of the constraint rows (signs: positive at an upper bound)
    objective: float
    iterations: int
    status: str                        # "solved", "solved (polished)", "max iterations"
    primal_residual: float
    dual_residual: float

    @property
    def ok(self) -> bool:
        return self.status.startswith("solved")


def _stack(A, l, u, lb, ub, n):
    rows, lo, hi = [], [], []
    if A is not None:
        A = np.atleast_2d(np.asarray(A, float))
        rows.append(A)
        lo.append(np.asarray(l, float) if l is not None else np.full(A.shape[0], -np.inf))
        hi.append(np.asarray(u, float) if u is not None else np.full(A.shape[0], np.inf))
    if lb is not None or ub is not None:
        lbv = np.full(n, -np.inf) if lb is None else np.broadcast_to(np.asarray(lb, float), (n,)).copy()
        ubv = np.full(n, np.inf) if ub is None else np.broadcast_to(np.asarray(ub, float), (n,)).copy()
        keep = np.isfinite(lbv) | np.isfinite(ubv)
        if keep.any():
            rows.append(np.eye(n)[keep])
            lo.append(lbv[keep])
            hi.append(ubv[keep])
    if not rows:
        return np.zeros((0, n)), np.zeros(0), np.zeros(0)
    return np.vstack(rows), np.concatenate(lo), np.concatenate(hi)


def _ruiz(P, q, A, iterations: int = 15):
    """Diagonal scalings D (variables), E (rows) and a cost scale c that make the columns of the KKT matrix about the same size."""
    n, m = P.shape[0], A.shape[0]
    D, E, c = np.ones(n), np.ones(m), 1.0
    Ps, qs, As = P.copy(), q.copy(), A.copy()
    for _ in range(iterations):
        col = np.maximum(np.abs(Ps).max(axis=0) if n else np.zeros(0), np.abs(As).max(axis=0) if m else np.zeros(n))
        row = np.abs(As).max(axis=1) if m else np.zeros(0)
        d = 1.0 / np.sqrt(np.where(col > 1e-8, col, 1.0))
        e = 1.0 / np.sqrt(np.where(row > 1e-8, row, 1.0))
        Ps = Ps * d[:, None] * d[None, :]
        qs = qs * d
        As = As * e[:, None] * d[None, :]
        D, E = D * d, E * e
        mean_col = np.abs(Ps).max(axis=0).mean() if n else 1.0
        gamma = 1.0 / max(mean_col, np.abs(qs).max() if n else 0.0, 1e-8)
        Ps, qs, c = Ps * gamma, qs * gamma, c * gamma
    return Ps, qs, As, D, E, c


def solve_qp(P, q, A=None, l=None, u=None, lb=None, ub=None, x0=None, eps: float = 1e-9, max_iter: int = 40000, rho: float = 0.1, sigma: float = 1e-6, alpha: float = 1.6,
             polish: bool = True, check_every: int = 25) -> QPResult:
    """Solve the quadratic program above. ``P`` (n x n, PSD), ``q`` (n); ``A`` (m x n) with ``l``/``u`` (m) and/or box bounds ``lb``/``ub`` (scalars or length n)."""
    P = np.atleast_2d(np.asarray(P, float))
    q = np.asarray(q, float).ravel()
    n = len(q)
    P = 0.5 * (P + P.T)
    A0, l0, u0 = _stack(A, l, u, lb, ub, n)
    m = A0.shape[0]
    if m and (l0 > u0 + 1e-12).any():
        return QPResult(np.full(n, np.nan), np.zeros(m), np.nan, 0, "infeasible bounds", np.inf, np.inf)
    Ps, qs, As, D, E, c = _ruiz(P, q, A0)
    ls = np.where(np.isfinite(l0), l0 * E, -np.inf) if m else l0
    us = np.where(np.isfinite(u0), u0 * E, np.inf) if m else u0
    equality = (np.isfinite(ls) & np.isfinite(us) & (np.abs(us - ls) < 1e-9)) if m else np.zeros(0, bool)
    x = np.zeros(n) if x0 is None else np.asarray(x0, float) / D
    z = np.clip(As @ x, ls, us) if m else np.zeros(0)
    y = np.zeros(m)

    def factor(rho_vec):
        H = Ps + sigma * np.eye(n) + (As.T * rho_vec) @ As if m else Ps + sigma * np.eye(n)
        try:
            return ("chol", cho_factor(H, lower=True, check_finite=False))
        except np.linalg.LinAlgError:
            return ("lu", lu_factor(H, check_finite=False))

    def solve(fac, rhs):
        return cho_solve(fac[1], rhs, check_finite=False) if fac[0] == "chol" else lu_solve(fac[1], rhs, check_finite=False)

    rho_k = rho
    rho_vec = np.where(equality, 1e3 * rho_k, rho_k) if m else np.zeros(0)
    fac = factor(rho_vec)
    status, iteration, r_prim, r_dual = "max iterations", 0, np.inf, np.inf
    for iteration in range(1, max_iter + 1):
        rhs = sigma * x - qs + (As.T @ (rho_vec * z - y) if m else 0.0)
        xt = solve(fac, rhs)
        zt = As @ xt if m else z
        x_new = alpha * xt + (1.0 - alpha) * x
        z_relaxed = alpha * zt + (1.0 - alpha) * z
        z_new = np.clip(z_relaxed + (y / rho_vec if m else 0.0), ls, us) if m else z
        if m:
            y = y + rho_vec * (z_relaxed - z_new)
        x, z = x_new, z_new
        if iteration % check_every == 0 or iteration == 1:
            Ax = As @ x if m else np.zeros(0)
            Px = Ps @ x
            Aty = As.T @ y if m else np.zeros(n)
            r_prim = np.abs(Ax - z).max() if m else 0.0
            r_dual = np.abs(Px + qs + Aty).max()
            tol_p = eps * (1.0 + max(np.abs(Ax).max() if m else 0.0, np.abs(z).max() if m else 0.0))
            tol_d = eps * (1.0 + max(np.abs(Px).max(), np.abs(Aty).max(), np.abs(qs).max()))
            if r_prim <= tol_p and r_dual <= tol_d:
                status = "solved"
                break
            if m and iteration % (4 * check_every) == 0:                                     # rebalance the primal and dual residuals
                num = r_prim / max(np.abs(Ax).max(), np.abs(z).max(), 1e-12)
                den = r_dual / max(np.abs(Px).max(), np.abs(Aty).max(), np.abs(qs).max(), 1e-12)
                ratio = np.sqrt(num / max(den, 1e-30))
                if ratio > 5.0 or ratio < 0.2:
                    rho_k = float(np.clip(rho_k * ratio, 1e-6, 1e6))
                    rho_vec = np.where(equality, 1e3 * rho_k, rho_k)
                    fac = factor(rho_vec)
    xs, ys = x, y
    if polish and m and status == "solved":
        polished = _polish(Ps, qs, As, ls, us, xs, ys, zs=z)
        if polished is not None:
            xs = polished
            status = "solved (polished)"
    x_out = xs * D
    y_out = (ys * E) / c if m else ys
    obj = float(0.5 * x_out @ P @ x_out + q @ x_out)
    return QPResult(x_out, y_out, obj, iteration, status, float(r_prim), float(r_dual))


def _polish(P, q, A, l, u, x, y, zs, delta: float = 1e-7, refine: int = 4):
    """Guess the active constraints from the multipliers, solve the equality-constrained QP they define, and keep the answer only if it is feasible and no worse (OSQP's polishing step)."""
    equality = np.isfinite(l) & np.isfinite(u) & (np.abs(u - l) < 1e-9)
    lower = np.isfinite(l) & ((zs - l) < -y)
    upper = np.isfinite(u) & ((u - zs) < y)
    act = lower | upper | equality
    if not act.any():
        return None
    Aa = A[act]
    ba = np.where(upper[act] & ~equality[act], u[act], l[act])
    n, k = len(q), Aa.shape[0]
    K0 = np.block([[P, Aa.T], [Aa, np.zeros((k, k))]])
    Kd = K0 + np.diag(np.concatenate([np.full(n, delta), np.full(k, -delta)]))
    rhs = np.concatenate([-q, ba])
    try:
        lu = lu_factor(Kd, check_finite=False)
    except (np.linalg.LinAlgError, ValueError):
        return None
    sol = lu_solve(lu, rhs, check_finite=False)
    for _ in range(refine):
        sol = sol + lu_solve(lu, rhs - K0 @ sol, check_finite=False)
    xp = sol[:n]
    Ax = A @ xp
    slack = 1e-7 * (1.0 + np.abs(Ax).max())
    if (Ax < l - slack).any() or (Ax > u + slack).any() or not np.isfinite(xp).all():
        return None
    old = 0.5 * x @ P @ x + q @ x
    new = 0.5 * xp @ P @ xp + q @ xp
    if new > old + 1e-9 * (1.0 + abs(old)):
        return None
    return xp


# ------------------------------------------------------------------------------------------------------------------ an interior-point method for small, degenerate problems
def solve_qp_ipm(P, q, A=None, l=None, u=None, lb=None, ub=None, tol: float = 1e-12, max_iter: int = 100, delta: float = 1e-11) -> QPResult:
    """The same problem as :func:`solve_qp`, solved by a primal-dual interior-point method (Mehrotra's predictor-corrector) with a dense Newton system.

    ADMM is the right tool for hundreds of variables and loose tolerances; it is slow on small problems with many nearly parallel constraints (a cutting-plane method produces exactly those), where it
    needs thousands of iterations. An interior-point method needs 10 to 30 whatever the degeneracy, and returns the answer to the tolerance asked for without a polishing step. The Newton system has
    ``n`` rows plus the equality rows, so this is for problems of up to a few hundred variables. ``delta`` regularises the system so dependent equality rows do not make it singular."""
    P = np.atleast_2d(np.asarray(P, float))
    q = np.asarray(q, float).ravel()
    n = len(q)
    P = 0.5 * (P + P.T)
    A0, l0, u0 = _stack(A, l, u, lb, ub, n)
    m0 = A0.shape[0]
    if m0 and (l0 > u0 + 1e-12).any():
        return QPResult(np.full(n, np.nan), np.zeros(m0), np.nan, 0, "infeasible bounds", np.inf, np.inf)
    eq = (np.isfinite(l0) & np.isfinite(u0) & (np.abs(u0 - l0) < 1e-12)) if m0 else np.zeros(0, bool)
    low = np.isfinite(l0) & ~eq if m0 else np.zeros(0, bool)
    up = np.isfinite(u0) & ~eq if m0 else np.zeros(0, bool)
    Ae, be = (A0[eq], l0[eq]) if m0 else (np.zeros((0, n)), np.zeros(0))
    G = np.vstack([-A0[low], A0[up]]) if m0 else np.zeros((0, n))
    h = np.concatenate([-l0[low], u0[up]]) if m0 else np.zeros(0)
    p, m = Ae.shape[0], G.shape[0]
    # scale: the objective to order one, every constraint row to unit size
    cs = max(np.abs(P).max() if n else 0.0, np.abs(q).max() if n else 0.0, 1e-12)
    Ps, qs = P / cs, q / cs
    ge = 1.0 / np.maximum(np.abs(G).max(axis=1), 1e-12) if m else np.zeros(0)
    ae = 1.0 / np.maximum(np.abs(Ae).max(axis=1), 1e-12) if p else np.zeros(0)
    Gs, hs, Aes, bes = G * ge[:, None], h * ge, Ae * ae[:, None], be * ae
    # starting point: the least-squares solution of the KKT conditions with the slack identity, shifted into the interior (the choice CVXOPT makes)
    K0 = np.block([[Ps + 1e-8 * np.eye(n), Aes.T, Gs.T], [Aes, -1e-8 * np.eye(p), np.zeros((p, m))], [Gs, np.zeros((m, p)), -np.eye(m)]]) if (p or m) else Ps + 1e-8 * np.eye(n)
    rhs0 = np.concatenate([-qs, bes, hs])
    try:
        sol0 = np.linalg.solve(K0, rhs0)
    except np.linalg.LinAlgError:
        sol0 = np.zeros(n + p + m)
    x, y, z = sol0[:n], sol0[n:n + p], sol0[n + p:]
    s = hs - Gs @ x if m else np.zeros(0)
    if m:
        shift_s = max(0.0, -float(s.min()))
        s = s + (1.0 + shift_s)
        z = -(Gs @ x - hs)
        shift_z = max(0.0, -float(z.min()))
        z = z + (1.0 + shift_z)
    status, iteration = "max iterations", 0
    best = (np.inf, x.copy(), y.copy(), s.copy(), z.copy())                                    # the best iterate so far: near the optimum rounding error can start to undo the progress
    np_state = np.seterr(all="ignore")                                                      # a singular Newton system gives NaN steps, which end the iteration below
    for iteration in range(1, max_iter + 1):
        rd = Ps @ x + qs + (Gs.T @ z if m else 0.0) + (Aes.T @ y if p else 0.0)
        rp = Aes @ x - bes if p else np.zeros(0)
        ri = Gs @ x + s - hs if m else np.zeros(0)
        mu = float(s @ z) / m if m else 0.0
        scale = 1.0 + np.abs(qs).max()
        merit = max(np.abs(rd).max(), np.abs(rp).max() if p else 0.0, np.abs(ri).max() if m else 0.0, mu) / scale
        if merit < best[0]:
            best = (merit, x.copy(), y.copy(), s.copy(), z.copy())
        if merit <= tol:
            status = "solved"
            break
        if best[0] <= 1e-6 and merit > 10.0 * best[0]:                                       # diverging after having been nearly there: stop at the best point
            break
        H = Ps + (Gs.T * (z / s)) @ Gs if m else Ps.copy()
        K = np.block([[H + delta * np.eye(n), Aes.T], [Aes, -delta * np.eye(p)]]) if p else H + delta * np.eye(n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LinAlgWarning)                                   # an infeasible problem makes the system singular: that is reported as no solution
                lu = lu_factor(K, check_finite=False)
        except (np.linalg.LinAlgError, ValueError):
            break

        def newton(rc_t):
            rhs_x = -rd - (Gs.T @ ((z / s) * ri - rc_t / s) if m else 0.0)
            sol = lu_solve(lu, np.concatenate([rhs_x, -rp]) if p else rhs_x, check_finite=False)
            dx, dy = sol[:n], sol[n:]
            ds = -ri - Gs @ dx if m else np.zeros(0)
            dz = (z / s) * (Gs @ dx + ri) - rc_t / s if m else np.zeros(0)
            return dx, dy, ds, dz

        def longest(v, dv):
            neg = dv < 0
            return min(1.0, float(np.min(-v[neg] / dv[neg]))) if neg.any() else 1.0

        if m:
            dx_a, dy_a, ds_a, dz_a = newton(s * z)
            a_aff = min(longest(s, ds_a), longest(z, dz_a))
            mu_aff = float((s + a_aff * ds_a) @ (z + a_aff * dz_a)) / m
            sigma = (mu_aff / mu) ** 3 if mu > 0 else 0.0
            dx, dy, ds, dz = newton(s * z - sigma * mu + ds_a * dz_a)
            alpha = max(0.995 * min(longest(s, ds), longest(z, dz)), 1e-8)
            if not (np.isfinite(dx).all() and np.isfinite(ds).all() and np.isfinite(dz).all()):
                break                                                                          # a singular Newton system: no further progress is possible
        else:
            dx, dy, ds, dz = newton(np.zeros(0))
            alpha = 1.0
        x, y = x + alpha * dx, y + alpha * dy
        if m:
            s, z = s + alpha * ds, z + alpha * dz
    np.seterr(**np_state)
    if status != "solved" and best[0] <= 1e-7:
        status = "solved (reduced accuracy)"
    if status.startswith("solved"):
        _, x, y, s, z = best
    x_out = x
    y_out = np.zeros(m0)
    if m0:
        z_orig = cs * ge * z if m else z                                                    # undo the row and cost scaling: P x + q + A'y = 0 with y positive at an upper bound
        n_low = int(low.sum())
        y_out[np.flatnonzero(low)] = -z_orig[:n_low]
        y_out[np.flatnonzero(up)] += z_orig[n_low:n_low + int(up.sum())]
        if p:
            y_out[np.flatnonzero(eq)] = cs * ae * y
    obj = float(0.5 * x_out @ P @ x_out + q @ x_out)
    rd_final = np.abs(P @ x_out + q + (A0.T @ y_out if m0 else 0.0)).max() if n else 0.0
    rp_final = float(max(np.maximum(l0 - A0 @ x_out, 0).max(initial=0.0), np.maximum(A0 @ x_out - u0, 0).max(initial=0.0))) if m0 else 0.0
    return QPResult(x_out, y_out, obj, iteration, status, rp_final, float(rd_final))
