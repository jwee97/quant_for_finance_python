"""Solving the linear systems that finite-difference schemes produce.

An implicit finite-difference step for an option price turns a PDE into ``M V_new = rhs`` where ``M`` is tridiagonal (one unknown per grid point, each coupled to its two
neighbours) and, for a model with jumps, an explicit convolution on the right. Which solver is right depends on the problem:

``thomas``               Gaussian elimination written for a tridiagonal matrix: one forward and one backward sweep, ``O(n)``. No pivoting, so it needs a diagonally dominant matrix,
                         which implicit schemes for a parabolic equation give. It is here to be read and to check the others.
``tridiagonal_factor``   LAPACK's tridiagonal LU (``dgttrf``/``dgttrs``) with partial pivoting: factor once, solve at every time step with a new right-hand side. This is what the
                         PDE code uses, because the matrix of a constant-coefficient scheme never changes.
``jacobi``, ``gauss_seidel``, ``sor``
                         the classical iterations for a general (dense or sparse) matrix. Jacobi updates every unknown from the last sweep; Gauss-Seidel uses new values as soon as they
                         exist; successive over-relaxation (SOR) moves each update a factor ``omega`` further than Gauss-Seidel would, and with the best ``omega`` (``optimal_omega``)
                         converges in far fewer sweeps. They converge for strictly diagonally dominant matrices, and SOR for symmetric positive definite ones with ``0 < omega < 2``.
``sor_tridiagonal``      SOR specialised to a tridiagonal matrix, ``O(n)`` a sweep.
``psor_tridiagonal``     projected SOR (Cryer 1971) for the *linear complementarity problem* of an American option: find ``x >= g`` with ``M x >= d`` and ``(M x - d)'(x - g) = 0``,
                         i.e. the price satisfies the PDE where holding is better and equals the payoff where exercising is. Each SOR update is clipped at the payoff.
``brennan_schwartz``     the direct ``O(n)`` solution of the same problem (Brennan and Schwartz 1977; Jaillet, Lamberton and Lapeyre 1990): Thomas's elimination, with the clipping applied
                         in the back substitution. Exact when the exercise region is one interval at one end of the grid, as it is for a put (low prices) and a call on a dividend payer.

All the iterative routines return an ``IterativeResult`` with the number of sweeps and the residual after each, so convergence can be inspected and plotted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.linalg import eigvalsh_tridiagonal, lapack, lu_factor, lu_solve


@dataclass
class IterativeResult:
    """The outcome of an iterative solve. ``residual`` is ``max |b - A x|`` (``max |x_new - x_old|`` for the projected solver); ``history`` has it after every sweep."""

    x: np.ndarray
    iterations: int
    residual: float
    converged: bool
    history: list[float] = field(default_factory=list)


def _tridiagonal_arrays(lower, diag, upper) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower, diag, upper = (np.asarray(v, dtype=float) for v in (lower, diag, upper))
    n = len(diag)
    if n < 1 or lower.shape != (n - 1,) or upper.shape != (n - 1,):
        raise ValueError("a tridiagonal system of size n has a diagonal of length n and off-diagonals of length n - 1")
    return lower, diag, upper


# ------------------------------------------------------------------------------------------------------------------------------------------ direct
def thomas(lower, diag, upper, rhs) -> np.ndarray:
    """Solve ``T x = rhs`` for the tridiagonal ``T`` with sub-diagonal ``lower``, diagonal ``diag`` and super-diagonal ``upper`` (the Thomas algorithm).

    ``rhs`` may be a vector or an ``(n, k)`` array of ``k`` right-hand sides. Raises ``ZeroDivisionError`` on a zero pivot (the matrix needs pivoting: use ``tridiagonal_solve``).
    """
    lower, diag, upper = _tridiagonal_arrays(lower, diag, upper)
    x = np.array(rhs, dtype=float)
    n = len(diag)
    if x.shape[0] != n:
        raise ValueError("the right-hand side must have one row per unknown")
    c = np.empty(max(n - 1, 0))
    pivot = diag[0]
    if pivot == 0.0:
        raise ZeroDivisionError("zero pivot in the Thomas algorithm")
    if n > 1:
        c[0] = upper[0] / pivot
    x[0] = x[0] / pivot
    for i in range(1, n):
        pivot = diag[i] - lower[i - 1] * c[i - 1]
        if pivot == 0.0:
            raise ZeroDivisionError("zero pivot in the Thomas algorithm")
        if i < n - 1:
            c[i] = upper[i] / pivot
        x[i] = (x[i] - lower[i - 1] * x[i - 1]) / pivot
    for i in range(n - 2, -1, -1):
        x[i] = x[i] - c[i] * x[i + 1]
    return x


class TridiagonalLU:
    """A tridiagonal matrix factored once (LAPACK ``dgttrf``, partial pivoting) and solved against any number of right-hand sides (``dgttrs``)."""

    def __init__(self, lower, diag, upper):
        lower, diag, upper = _tridiagonal_arrays(lower, diag, upper)
        self.n = len(diag)
        self._dense = None
        self._factors = None
        if self.n <= 2:                                              # LAPACK's tridiagonal routines have no 1 x 1 or 2 x 2 case: use a dense LU
            dense = np.diag(diag) + np.diag(lower, -1) + np.diag(upper, 1)
            if abs(np.linalg.det(dense)) == 0.0:
                raise ZeroDivisionError("the tridiagonal matrix is singular")
            self._dense = lu_factor(dense)
            return
        dl, d, du, du2, ipiv, info = lapack.dgttrf(lower, diag, upper)
        if info > 0:
            raise ZeroDivisionError("the tridiagonal matrix is singular")
        if info < 0:
            raise ValueError("illegal argument to LAPACK dgttrf")
        self._factors = (dl, d, du, du2, ipiv)

    def solve(self, rhs) -> np.ndarray:
        b = np.asarray(rhs, dtype=float)
        if b.shape[0] != self.n:
            raise ValueError("the right-hand side must have one row per unknown")
        if self._dense is not None:
            return lu_solve(self._dense, b)
        x, info = lapack.dgttrs(*self._factors, b)
        if info != 0:
            raise ValueError("LAPACK dgttrs failed")
        return x


def tridiagonal_factor(lower, diag, upper) -> TridiagonalLU:
    """Factor a tridiagonal matrix for repeated solves (``.solve(rhs)``): ``O(n)`` to factor, ``O(n)`` to solve."""
    return TridiagonalLU(lower, diag, upper)


def tridiagonal_solve(lower, diag, upper, rhs) -> np.ndarray:
    """Solve a tridiagonal system with pivoting (one factorisation, one solve)."""
    return TridiagonalLU(lower, diag, upper).solve(rhs)


# ----------------------------------------------------------------------------------------------------------------------------------- iterations
def _as_square(A):
    if sparse.issparse(A):
        A = A.tocsr()
    else:
        A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("a square matrix is needed")
    return A


def _residual(A, x, b) -> float:
    return float(np.max(np.abs(b - A @ x)))


def jacobi(A, b, x0=None, tol: float = 1e-10, max_iter: int = 10_000) -> IterativeResult:
    """Jacobi iteration ``x <- D^-1 (b - (A - D) x)``; converges when ``A`` is strictly diagonally dominant."""
    A = _as_square(A)
    b = np.asarray(b, dtype=float)
    d = A.diagonal() if sparse.issparse(A) else np.diag(A)
    if np.any(d == 0.0):
        raise ZeroDivisionError("a zero on the diagonal: Jacobi iteration needs a non-zero diagonal")
    off = A - (sparse.diags(d) if sparse.issparse(A) else np.diag(d))
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    history: list[float] = []
    for k in range(1, max_iter + 1):
        x = (b - off @ x) / d
        res = _residual(A, x, b)
        history.append(res)
        if res < tol:
            return IterativeResult(x, k, res, True, history)
    return IterativeResult(x, max_iter, history[-1] if history else float("nan"), False, history)


def sor(A, b, omega: float = 1.5, x0=None, tol: float = 1e-10, max_iter: int = 10_000) -> IterativeResult:
    """Successive over-relaxation: ``x_i <- (1 - omega) x_i + omega (b_i - sum_{j != i} A_ij x_j) / A_ii``, using the new ``x_j`` for ``j < i`` in the same sweep.

    ``omega = 1`` is Gauss-Seidel; ``1 < omega < 2`` over-relaxes and, for the right value, converges much faster. A sweep costs one pass over the non-zeros of ``A``.
    """
    if not 0.0 < omega < 2.0:
        raise ValueError("SOR converges only for 0 < omega < 2")
    A = _as_square(A)
    b = np.asarray(b, dtype=float)
    n = len(b)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    csr = sparse.issparse(A)
    if csr:
        indptr, indices, data = A.indptr, A.indices, A.data
        diag = A.diagonal()
    else:
        diag = np.diag(A)
    if np.any(diag == 0.0):
        raise ZeroDivisionError("a zero on the diagonal: SOR needs a non-zero diagonal")
    history: list[float] = []
    for k in range(1, max_iter + 1):
        for i in range(n):
            if csr:
                s, e = indptr[i], indptr[i + 1]
                sigma = float(data[s:e] @ x[indices[s:e]]) - diag[i] * x[i]
            else:
                sigma = float(A[i] @ x) - diag[i] * x[i]
            x[i] = (1.0 - omega) * x[i] + omega * (b[i] - sigma) / diag[i]
        res = _residual(A, x, b)
        history.append(res)
        if res < tol:
            return IterativeResult(x, k, res, True, history)
    return IterativeResult(x, max_iter, history[-1], False, history)


def gauss_seidel(A, b, x0=None, tol: float = 1e-10, max_iter: int = 10_000) -> IterativeResult:
    """Gauss-Seidel iteration: SOR with ``omega = 1``."""
    return sor(A, b, 1.0, x0, tol, max_iter)


def optimal_omega(lower, diag, upper) -> float:
    """The over-relaxation factor that minimises the SOR sweeps for a tridiagonal matrix: ``2 / (1 + sqrt(1 - rho^2))`` with ``rho`` the spectral radius of the Jacobi iteration
    matrix (Young 1950; valid for consistently ordered matrices, which tridiagonal ones are). For a matrix whose off-diagonal products are not positive the Jacobi eigenvalues are
    not real and 1 (Gauss-Seidel) is returned."""
    lower, diag, upper = _tridiagonal_arrays(lower, diag, upper)
    n = len(diag)
    if n < 2:
        return 1.0
    prod = lower * upper / (diag[:-1] * diag[1:])
    if np.any(prod <= 0.0):
        return 1.0
    rho = float(np.max(np.abs(eigvalsh_tridiagonal(np.zeros(n), np.sqrt(prod)))))
    return 1.0 if rho >= 1.0 else float(2.0 / (1.0 + np.sqrt(1.0 - rho ** 2)))


def sor_tridiagonal(lower, diag, upper, rhs, omega: float = 1.5, x0=None, tol: float = 1e-10, max_iter: int = 10_000) -> IterativeResult:
    """SOR for a tridiagonal system, ``O(n)`` a sweep (the stopping rule is the largest change of any unknown in a sweep)."""
    if not 0.0 < omega < 2.0:
        raise ValueError("SOR converges only for 0 < omega < 2")
    lower, diag, upper = _tridiagonal_arrays(lower, diag, upper)
    n = len(diag)
    lo, di, up, b = lower.tolist(), diag.tolist(), upper.tolist(), np.asarray(rhs, dtype=float).tolist()
    x = [0.0] * n if x0 is None else np.asarray(x0, dtype=float).tolist()
    history: list[float] = []
    for k in range(1, max_iter + 1):
        change = 0.0
        for i in range(n):
            s = b[i]
            if i > 0:
                s -= lo[i - 1] * x[i - 1]
            if i < n - 1:
                s -= up[i] * x[i + 1]
            new = (1.0 - omega) * x[i] + omega * s / di[i]
            change = max(change, abs(new - x[i]))
            x[i] = new
        history.append(change)
        if change < tol:
            xa = np.array(x)
            return IterativeResult(xa, k, _tridiagonal_residual(lower, diag, upper, xa, np.asarray(rhs, dtype=float)), True, history)
    xa = np.array(x)
    return IterativeResult(xa, max_iter, _tridiagonal_residual(lower, diag, upper, xa, np.asarray(rhs, dtype=float)), False, history)


def _tridiagonal_residual(lower, diag, upper, x, rhs) -> float:
    ax = diag * x
    ax[1:] += lower * x[:-1]
    ax[:-1] += upper * x[1:]
    return float(np.max(np.abs(rhs - ax)))


# ----------------------------------------------------------------------------------------------------------------- the complementarity problem
def psor_tridiagonal(lower, diag, upper, rhs, obstacle, omega: float = 1.2, x0=None, tol: float = 1e-10, max_iter: int = 10_000) -> IterativeResult:
    """Projected SOR for the linear complementarity problem ``x >= g``, ``M x >= d``, ``(M x - d)'(x - g) = 0`` with a tridiagonal ``M`` (``g`` is ``obstacle``, ``d`` is ``rhs``).

    Each unknown is updated as in SOR and then clipped from below at the obstacle. Converges for a positive definite or an M-matrix ``M`` and ``0 < omega < 2``; the iteration
    stops when no unknown changes by more than ``tol``. ``residual`` is that last change.
    """
    if not 0.0 < omega < 2.0:
        raise ValueError("projected SOR converges only for 0 < omega < 2")
    lower, diag, upper = _tridiagonal_arrays(lower, diag, upper)
    n = len(diag)
    g = np.asarray(obstacle, dtype=float)
    if g.shape != (n,):
        raise ValueError("one obstacle value per unknown")
    lo, di, up, b, gl = lower.tolist(), diag.tolist(), upper.tolist(), np.asarray(rhs, dtype=float).tolist(), g.tolist()
    x = [max(v, w) for v, w in zip(([0.0] * n if x0 is None else np.asarray(x0, dtype=float).tolist()), gl)]
    history: list[float] = []
    for k in range(1, max_iter + 1):
        change = 0.0
        for i in range(n):
            s = b[i]
            if i > 0:
                s -= lo[i - 1] * x[i - 1]
            if i < n - 1:
                s -= up[i] * x[i + 1]
            new = (1.0 - omega) * x[i] + omega * s / di[i]
            if new < gl[i]:
                new = gl[i]
            change = max(change, abs(new - x[i]))
            x[i] = new
        history.append(change)
        if change < tol:
            return IterativeResult(np.array(x), k, change, True, history)
    return IterativeResult(np.array(x), max_iter, history[-1], False, history)


def brennan_schwartz(lower, diag, upper, rhs, obstacle, exercise: str = "low") -> np.ndarray:
    """Solve the tridiagonal linear complementarity problem directly in ``O(n)`` (Brennan and Schwartz 1977).

    Eliminate as in the Thomas algorithm, starting from the end of the grid that is *not* exercised, then substitute back starting from the exercised end, clipping each
    unknown at the obstacle: ``x_i = max(g_i, (d_i' - c_i x_{i+1}) / b_i')``. Jaillet, Lamberton and Lapeyre (1990) proved that this is the exact solution when ``M`` is an
    M-matrix and the exercise region is one interval at the end named by ``exercise``: ``"low"`` (the lowest prices, an American put) or ``"high"`` (a call on an asset that pays
    a dividend). With the exercise region anywhere else it can be wrong; ``psor_tridiagonal`` makes no such assumption.
    """
    if exercise not in ("low", "high"):
        raise ValueError("exercise must be 'low' or 'high'")
    lower, diag, upper = _tridiagonal_arrays(lower, diag, upper)
    d = np.asarray(rhs, dtype=float)
    g = np.asarray(obstacle, dtype=float)
    n = len(diag)
    if d.shape != (n,) or g.shape != (n,):
        raise ValueError("one right-hand side and one obstacle value per unknown")
    if exercise == "low":                                    # the sweep below ends at the top of the arrays: mirror the grid so that the exercised end is the top, then mirror back
        return _brennan_schwartz_top(upper[::-1], diag[::-1], lower[::-1], d[::-1], g[::-1])[::-1]
    return _brennan_schwartz_top(lower, diag, upper, d, g)


def _brennan_schwartz_top(lower, diag, upper, d, g) -> np.ndarray:
    """Thomas elimination downwards and back substitution from the last unknown with the obstacle: exact when the exercise region is at the end of the arrays."""
    n = len(diag)
    b = diag.tolist()
    y = d.tolist()
    lo, up, gl = lower.tolist(), upper.tolist(), g.tolist()
    for i in range(1, n):                                    # forward elimination
        w = lo[i - 1] / b[i - 1]
        b[i] -= w * up[i - 1]
        y[i] -= w * y[i - 1]
    x = [0.0] * n
    x[n - 1] = max(gl[n - 1], y[n - 1] / b[n - 1])
    for i in range(n - 2, -1, -1):                           # back substitution with the obstacle
        x[i] = max(gl[i], (y[i] - up[i] * x[i + 1]) / b[i])
    return np.array(x)
