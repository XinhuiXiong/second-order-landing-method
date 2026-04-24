"""Second-Order Landing (SOL) method on the Stiefel manifold.

This module implements the **Second-Order Landing (SOL)** algorithm described in the
paper *"A second-order landing method on the Stiefel manifold via Newton–Schulz
iteration"*.

The algorithm targets problems of the form

    min f(X)  s.t.  X^T X = I_p.

SOL uses the decomposition

    Λ(X) = T(X) + N(X),

where:
    - N(X) is the normal feasibility restoration term, chosen as the order-1
      Newton–Schulz displacement

            N(X) = -1/2 ∇N(X),   ∇N(X) = X(X^T X - I).

    - T(X) is the tangent second-order term obtained by solving

            A_T(X)[T(X)] = -grad f(X) - A_N(X)[N(X)],

The implementation below intentionally avoids explicit retractions (QR/SVD/polar),
using only matrix multiplications and a Krylov linear solver for the subproblem.

Dependencies:
    numpy, scipy (for lgmres, bicgstab + LinearOperator)

Compatibility:
    Works with or without Pymanopt. If you pass a Pymanopt Problem, it will use
    problem.manifold.random_point() for initialization. Otherwise you can pass
    explicit callables for f, ∇f, and ∇^2 f.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any

from linear_solvers import g_metric_cg, g_metric_minres



import numpy as np
import time

try:
    from scipy.sparse.linalg import LinearOperator, cg, gmres, minres, bicgstab, cgs, lgmres, gcrotmk,tfqmr
    from scipy.linalg import cho_factor, cho_solve
except Exception as e:  # pragma: no cover
    raise ImportError(
        "second_order_landing.py requires scipy (scipy.sparse.linalg.gmres)."
    ) from e



Array = np.ndarray

from typing import Optional, Tuple


def sym(A: Array) -> Array:
    return 0.5 * (A + A.T)


def skew(A: Array) -> Array:
    return 0.5 * (A - A.T)


def grad_N(X: Array) -> Array:
    """Penalty gradient ∇N(X) with N(X)=1/4||X^T X - I||_F^2."""
    p = X.shape[1]
    return X @ (X.T @ X - np.eye(p))

def NS_displacement_r(X: Array, r: int) -> Array:
    """
    Order-r truncated Newton–Schulz displacement:
        N_r(X) = X @ (g_r(E) - I),
    where
        E = X.T @ X - I
    and
        g_r(E) = sum_{j=0}^r (-1)^j * C(2j,j) / 4^j * E^j.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Current iterate.
    r : int
        Truncation order, must satisfy r >= 1.

    Returns
    -------
    np.ndarray, shape (n, p)
        The displacement N_r(X).
    """
    E = X.T @ X - np.eye(X.shape[1], dtype=X.dtype)

    if r == 1:
        P = -0.5 * E
    elif r == 2:
        E2 = E @ E
        P = -0.5 * E + 3.0 / 8.0 * E2
    elif r == 3:
        E2 = E @ E
        E3 = E2 @ E
        P = -0.5 * E + 3.0 / 8.0 * E2 - 5.0 / 16.0 * E3
    else:
        raise ValueError("Only r=1,2,3 are supported in this implementation.")

    return X @ P


def T1(X: Array, grad_f: Callable[[Array], Array]) -> Array:
    """Relative gradient T1(X) = 2*skew(∇f(X) X^T) X.

    Algebraically: ∇f(X)(X^T X) - X(∇f(X)^T X).
    """
    G = grad_f(X)
    XTX = X.T @ X
    XTG = X.T @ G
    return G @ XTX - X @ XTG.T
    # G = grad_f(X)
    # return G @ (X.T @ X) - X @ (G.T @ X)


def proj_tangent(X: Array, V: Array, XTX_inv: Optional[Array] = None) -> Array:
    """Project V onto T_X St_{X^T X}(p,n) under the canonical metric.

    Π_T(V) = V - X (X^T X)^{-1} sym(X^T V).
    """
    if XTX_inv is None:
        XTX_inv = np.linalg.inv(X.T @ X)
    return V - X @ (XTX_inv @ sym(X.T @ V))



def AT_action(
    X: Array,
    V: Array,
    grad_f: Callable[[Array], Array],
    hess_f: Callable[[Array, Array], Array],
    *,
    XTX: Optional[Array] = None,
    G: Optional[Array] = None,
    XtG: Optional[Array] = None,
) -> Array:
    """
    A_T(X)[V] = HV @ (X^T X) - X @ (HV^T X) + G @ (V^T X) - V @ (G^T X)
    """
    Xt = X.T

    if XTX is None:
        XTX = Xt @ X
    if G is None:
        G = grad_f(X)
    if XtG is None:
        XtG = Xt @ G  # p×p

    HV = hess_f(X, V)

    # Small matrices (p×p)
    HVtX = HV.T @ X        # p×p
    VtX  = V.T @ X         # p×p

    # Terms: all n×p
    term2 = HV @ XTX - X @ HVtX
    term3 = G @ VtX  - V @ XtG.T

    return term2 + term3


# Adjoint of the tractable tangential operator with respect to Euclidean inner product, prepared for BiCGSTAB
def AT_adjoint_action(
    X, W, grad_f, hess_f, *, XTX=None, G=None, XtG=None
):
    """
    A_T(X)^*[W] = H(W XTX) - H(X (W^T X)) + X (W^T G) - W (X^T G)
    """
    if XTX is None:
        XTX = X.T @ X
    if G is None:
        G = grad_f(X)
    if XtG is None:
        XtG = X.T @ G  # p×p

    # small p×p
    WtX = W.T @ X
    WtG = W.T @ G

    D = W @ XTX - X @ WtX            # n×p
    HD = hess_f(X, D)

    term_g1 = X @ WtG                # n×p
    term_g2 = W @ XtG                # n×p

    return HD + (term_g1 - term_g2)


def AN_action(
    X: Array,
    V_normal: Array,
    grad_f: Callable[[Array], Array],
    hess_f: Callable[[Array, Array], Array],
    *,
    XTX: Optional[Array] = None,
    G: Optional[Array] = None,
    XtG: Optional[Array] = None,
) -> Array:
    """Normal-to-tangent coupling operator A_N(X)[V] in the paper (4.10).

    A_N(X)[V] = 2*skew(∇^2 f(X)[V] X^T + ∇f(X) V^T) X.

    Same algebra as AT_action, but V is the normal displacement.
    """
    if XTX is None:
        XTX = X.T @ X
    if G is None:
        G = grad_f(X)
    if XtG is None:
        XtG = X.T @ G  # p×p

    # if XTX is None:
    #     XTX = X.T @ X
    #
    # G = grad_f(X)
    # HV = hess_f(X, V_normal)
    # term2 = HV @ XTX - X @ (HV.T @ X)
    # term3 = G @ (V_normal.T @ X) - V_normal @ (G.T @ X)
    # Z = term2 + term3
    return AT_action(X, V_normal, grad_f, hess_f, XTX=XTX, G=G, XtG=XtG)


# def eta_safe(d: float, g: float, epsilon: float, lam: float = 0.5) -> float:
#     """Safeguard stepsize η_safe from Lemma 5.1 (paper).
#
#     Inputs:
#         d = ||X^T X - I||_F
#         g = ||Λ(X)||_F
#         epsilon: safety radius ε (paper's Definition 2.1)
#         lam: fixed parameter lambda
#     """
#     max_step = float('inf')
#     if lam > 1e-12:
#         max_step = 1.0 / (2 * lam)
#
#     if g <= 1e-12:
#         return 1.0
#
#
#     one_minus_d = 1.0 - d
#     if one_minus_d <= 0:
#         return 0.0
#
#     b = 0.5 * d * one_minus_d
#     denom = g * g
#
#     # inside = b^2 + g^2(epsilon - d)
#     delta = denom * (epsilon - d)
#     inside = (b * b) + delta
#
#     if inside <= 0:
#         return 0.0
#
#     sqrt_inside = np.sqrt(inside)
#
#     denominator_stable = b + sqrt_inside
#
#     if denominator_stable <= 1e-12:
#         eta = 0.0
#     else:
#         # Num = (-b + sqrt) * (b + sqrt) = inside - b^2 = delta = g^2(eps - d)
#         # Denom = g^2 * (b + sqrt)
#         # Result = (eps - d) / (b + sqrt)
#         eta = (epsilon - d) / denominator_stable
#
#     return float(min(max(eta, 0.0), max_step))

def eta_safe(d: float, g: float, epsilon: float, lam: float = 0.5) -> float:
    """Safeguard stepsize η_safe from Lemma 5.1 (paper).

    Inputs:
        d = ||X^T X - I||_F
        g = ||Λ(X)||_F
        epsilon: safety radius ε (paper's Definition 2.1)
        lam: fixed parameter lambda
    """
    if g <= 0:
        return 1.0

    # the paper's Lemma 5.1 (Eq. (5.2)):
    #
    #   η_safe(X) = min{ ( -1/2 d(1-d) + sqrt( 1/4 d^2 (1-d)^2 + g^2 (ε-d) ) ) / g^2 , 1 / (2 * lambda) }.
    #
    # This has the correct limiting behavior at d -> 0:
    #     η_safe -> sqrt(ε)/g.
    denom = g * g

    one_minus_d = 1.0 - d
    if one_minus_d <= 0:
        return 0.0

    b = 0.5 * d * one_minus_d
    inside = (b * b) + denom * (epsilon - d)
    if inside <= 0:
        return 0.0

    eta = (-b + np.sqrt(inside)) / denom

    if lam > 0.0:
        eta_safe_size = float(min(max(eta, 0.0), 1.0 / (2 * lam )))
    else:
        eta_safe_size = float(max(eta, 0.0))
    return eta_safe_size


# --------- First-order landing optimizer and result ----------


@dataclass
class Landing1Result:
    """Return object for FirstOrderLanding."""

    X: Array
    iterations: int
    stopping_reason: str
    log: Dict[str, list]


class FirstOrderLanding:
    """First-order landing (Landing-1) optimizer.

    This is the global phase used to find an initial point X0 such that
        ||Λ1(X0)||_F <= tol,
    where (paper notation)
        Λ1(X) = T1(X) + λ ∇N(X).

    Update:
        X_{k+1} = X_k - η_k Λ1(X_k),
    with a safeguard step size to stay inside the safety region
        St(p,n)_ε = { X : ||X^T X - I||_F <= ε }.

    Parameters:
        epsilon: safety region radius ε.
        lam: hyperparameter λ.
        eta: nominal step size η.
        tol: stopping tolerance on ||Λ1(X)||_F.
        max_iter: maximum iterations.
        verbosity: 0 silent; >=2 prints per-iteration info.
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.75,
        lam: float = 0.5,
        eta: float = 0.1,
        tol: float = 1e-3,
        max_iter: int = 2000,
        verbosity: int = 0,
    ):
        if not (0 < epsilon <= 0.75):
            raise ValueError("epsilon is too big.")
        if lam <= 0:
            raise ValueError("lam must be positive.")
        if eta <= 0:
            raise ValueError("eta must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")
        self.epsilon = float(epsilon)
        self.lam = float(lam)
        self.eta = float(eta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.verbosity = int(verbosity)

    def run(
        self,
        *,
        n: int,
        p: int,
        grad_f: Callable[[Array], Array],
        cost: Optional[Callable[[Array], float]] = None,
        X0: Optional[Array] = None,
        manifold: Optional[Any] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable[[Array, int], Any]] = None,
    ) -> Landing1Result:
        """Run first-order landing until ||Λ1||_F <= tol (or max_iter)."""
        rng = np.random.default_rng(seed)
        if X0 is None:
            if manifold is not None and hasattr(manifold, "random_point"):
                X = manifold.random_point()
            else:
                X = rng.standard_normal((n, p))
        else:
            X = np.array(X0, dtype=float, copy=True)

        log: Dict[str, list] = {
            "time": [],  # cumulative seconds since start
            "cost": [],
            "ortho_error": [],
            "tangent_residual": [],  # ||T1(X)||_F
            "landing_residual": [],  # ||Λ1(X)||_F
            "step_size": [],
        }

        t_start = time.perf_counter()
        stopping_reason = "max_iter"

        # Callback contract (per-iterate):
        #   callback(X, k) -> Any
        # Called once per iteration with the current iterate X and iteration index k.
        # If it returns True or the string "stop", the optimizer terminates early.
        #
        # Example:
        #   def cb(X, k):
        #       history.append(subspace_distance(X, X_ref))
        #       return False

        for k in range(1, self.max_iter + 1):
            XTX = X.T @ X
            ortho = float(np.linalg.norm(XTX - np.eye(p), ord="fro"))

            t1 = T1(X, grad_f)
            t1_norm = float(np.linalg.norm(t1, ord="fro"))

            lam1 = t1 + self.lam * grad_N(X)
            lam1_norm = float(np.linalg.norm(lam1, ord="fro"))

            f_val = float(cost(X)) if cost is not None else float("nan")
            elapsed = float(time.perf_counter() - t_start)

            log["time"].append(elapsed)
            log["cost"].append(f_val)
            log["ortho_error"].append(ortho)
            log["tangent_residual"].append(t1_norm)
            log["landing_residual"].append(lam1_norm)

            if self.verbosity >= 2:
                print(
                    f"[L1 {k:4d}] t={elapsed:8.3f}s  f={f_val:.6e}  ||T1||={t1_norm:.3e}  ||X^T X-I||={ortho:.3e}  ||Λ1||={lam1_norm:.3e}  (λ={self.lam:.2g})"
                )

            # Per-iteration callback hook (only expose current iterate X)
            if callable(callback):
                try:
                    cb_ret = callback(X, k)
                except Exception as ex:
                    raise RuntimeError(f"Callback raised an exception at iteration {k}: {ex}") from ex
                if cb_ret is True or cb_ret == "stop":
                    stopping_reason = "callback"
                    break

            if lam1_norm <= self.tol:
                stopping_reason = "tol"
                break

            # Try nominal step; safeguard if it exits the safety region.
            step = self.eta
            X_trial = X - step * lam1
            ortho_trial = float(
                np.linalg.norm(X_trial.T @ X_trial - np.eye(p), ord="fro")
            )
            if ortho_trial > self.epsilon:
                print(f"{ortho_trial}")
                step = eta_safe(ortho, lam1_norm, self.epsilon, self.lam)
            log["step_size"].append(float(step))
            X = X - step * lam1

        if log["time"]:
            t0 = log["time"][0]
            log["time"] = [t - t0 for t in log["time"]]

        return Landing1Result(
            X=X,
            iterations=len(log["landing_residual"]),
            stopping_reason=stopping_reason,
            log=log,
        )

@dataclass
class SOLResult:
    X: Array
    iterations: int
    stopping_reason: str
    log: Dict[str, list]


class SecondOrderLanding:
    """Second-Order Landing (SOL) solver.

    Parameters (paper-aligned):
        epsilon: safety region radius ε.
        eta: nominal stepsize η (we attempt η=1 by default; safeguard may shrink it).
        tol: stopping tolerance on ||Λ_1(X)||_F with lam=0.5.
        max_iter: maximum iterations.
        linear_rtol: LGMRES relative tolerance for the tangent solve.
            If None, uses an inexact-forcing rule similar to the original code.
        theta: forcing-term order θ for the inexact tangential solve.
        zeta_max: maximum forcing parameter ζ_max.
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.75,
        eta: float = 1.0,
        tol: float = 1e-12,
        max_iter: int = 200,
        linear_solver: str = "lgmres",
        linear_solver_options: Optional[dict] = None,
        linear_rtol: Optional[float] = None,
        theta: float = 1.0,
        zeta_max: float = 1e-2,
        linear_maxiter: int = 200,
        verbosity: int = 1,
    ):
        if not (0 < epsilon <= 0.75):
            raise ValueError("epsilon is too big.")
        self.epsilon = float(epsilon)
        self.eta = float(eta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.linear_solver = str(linear_solver).lower()
        self.linear_solver_options = dict(linear_solver_options or {})
        self.linear_rtol = linear_rtol
        if theta <= 0:
            raise ValueError("theta must be positive.")
        if not (0.0 < zeta_max < 1.0):
            raise ValueError("zeta_max must be in (0, 1).")
        self.theta = float(theta)
        self.zeta_max = float(zeta_max)
        self.linear_maxiter = int(linear_maxiter)
        self.verbosity = int(verbosity)

    def _solve_linear_system(self, Aop, b_flat, rtol: float):
        opts = dict(self.linear_solver_options)
        maxiter = opts.pop("maxiter", self.linear_maxiter)
        atol = opts.pop("atol", 1e-12)

        solver = self.linear_solver

        if solver == "lgmres":
            x, info = lgmres(Aop, b_flat, atol=atol, rtol=rtol, maxiter=maxiter, **opts)

        elif solver == "gmres":
            restart = opts.pop("restart", 20)
            x, info = gmres(
                Aop, b_flat, atol=atol, rtol=rtol,
                restart=restart, maxiter=maxiter, **opts
            )

        elif solver == "bicgstab":
            x, info = bicgstab(Aop, b_flat, atol=atol, rtol=rtol, maxiter=maxiter, **opts)

        elif solver == "cgs":
            x, info = cgs(Aop, b_flat, atol=atol, rtol=rtol, maxiter=maxiter, **opts)

        elif solver == "gcrotmk":
            x, info = gcrotmk(Aop, b_flat, atol=atol, rtol=rtol, maxiter=maxiter, **opts)

        else:
            raise ValueError(f"Unknown linear_solver: {solver}")

        return x, info


    def run(
        self,
        *,
        n: int,
        p: int,
        cost: Callable[[Array], float],
        grad_f: Callable[[Array], Array],
        hess_f: Callable[[Array, Array], Array],
        X0: Optional[Array] = None,
        manifold: Optional[Any] = None,
        NS_order: int = 1,
        seed: Optional[int] = None,
        callback: Optional[Callable[[Array, int], Any]] = None,
    ) -> SOLResult:
        """Run SOL.

        If X0 is None and a Pymanopt manifold is provided, uses manifold.random_point().
        Otherwise uses a random Gaussian matrix.
        """
        rng = np.random.default_rng(seed)
        if X0 is None:
            if manifold is not None and hasattr(manifold, "random_point"):
                X = manifold.random_point()
            else:
                X = rng.standard_normal((n, p))
        else:
            X = np.array(X0, dtype=float, copy=True)

        log: Dict[str, list] = {
            "time": [],
            "cost": [],
            "ortho_error": [],
            "tangent_residual": [],  # ||T1(X)||_F
            "landing_residual": [],  # ||Λ1(X)||_F with λ=1/2
            "step_size": [],
            "solver_info": [],
            "solver_flag": [],
            "rtol": [],
        }
        t_start = time.perf_counter()
        def landing_field_first_order(Xk: Array) -> Array:
            # Λ1(X) = T1(X) + λ ∇N(X) with λ = 1/2 (paper's SOL choice)
            return T1(Xk, grad_f) + 0.5 * grad_N(Xk)

        stopping_reason = "max_iter"

        # Callback contract (per-iterate):
        #   callback(X, k) -> Any
        # Called once per iteration with the current iterate X and iteration index k.
        # If it returns True or the string "stop", the optimizer terminates early.

        for k in range(1, self.max_iter + 1):
            XTX = X.T @ X
            G = grad_f(X)
            XtG = X.T @ G

            t1 = T1(X, grad_f)
            t1_norm = float(np.linalg.norm(t1, ord="fro"))
            ortho = float(np.linalg.norm(XTX - np.eye(p), ord="fro"))
            lam1 = landing_field_first_order(X)
            lam1_norm = float(np.linalg.norm(lam1, ord="fro"))


            log["cost"].append(float(cost(X)))
            log["ortho_error"].append(ortho)
            log["tangent_residual"].append(t1_norm)
            log["landing_residual"].append(lam1_norm)

            elapsed = float(time.perf_counter() - t_start)
            log["time"].append(elapsed)

            if self.verbosity >= 2:
                print(
                    f"[SOL {k:4d}] t={elapsed:8.3f}s  cost={log['cost'][-1]:.6e}  "
                    f"||T1||={log['tangent_residual'][-1]:.3e}  "
                    f"||X^T X-I||={ortho:.3e}  ||Λ1||={lam1_norm:.3e}"
                )

            # Per-iteration callback hook (only expose current iterate X)
            if callable(callback):
                try:
                    cb_ret = callback(X, k)
                except Exception as ex:
                    raise RuntimeError(f"Callback raised an exception at iteration {k}: {ex}") from ex
                if cb_ret is True or cb_ret == "stop":
                    stopping_reason = "callback"
                    break

            if lam1_norm <= self.tol:
                stopping_reason = "tol"
                break

            # Normal step (order-r Newton–Schulz displacement)
            N = NS_displacement_r(X, NS_order)

            # Right-hand side b(X) = -T1(X) - A_N(X)[N(X)]
            b = -t1 - AN_action(X, N, grad_f, hess_f, XTX=XTX, G=G, XtG=XtG)


            chol_G = cho_factor(XTX, overwrite_a=False, check_finite=False)
            Ginv = cho_solve(chol_G, np.eye(p))
            XGinv = X @ Ginv
            b = proj_tangent_cached(X, b, Ginv=Ginv, XGinv=XGinv)

            # Tangential solve (symmetrized): 0.5*(A_T(X) + A_T(X)^*)[T] = b
            b_flat = b.ravel()
            dim = n * p
            def matvec(v_flat: Array) -> Array:
                V = v_flat.reshape(n, p)
                AV = AT_action(X, V, grad_f, hess_f, XTX=XTX, G=G, XtG=XtG)
                return AV.ravel()
                # AstarV = AT_adjoint_action(X, V, grad_f, hess_f, XTX=XTX)
                # return (0.5 * (AV + AstarV)).ravel()

            def rmatvec(v_flat: Array) -> Array:
                V = v_flat.reshape(n, p)
                AstarV = AT_adjoint_action(X, V, grad_f, hess_f, XTX=XTX, G=G, XtG=XtG)
                return AstarV.ravel()

            Aop = LinearOperator((dim, dim), matvec=matvec, rmatvec=rmatvec, dtype=float)

            if self.linear_rtol is None:
                b_norm = float(np.linalg.norm(b_flat))
                print(f"b_norm:{b_norm}")
                if b_norm == 0.0:
                    rtol = 0.0
                else:
                    rtol = min(self.zeta_max, b_norm ** self.theta)
            else:
                rtol = float(self.linear_rtol)
            print(f"rtol:{rtol}")
            log["rtol"].append(float(rtol))


            T_flat, info = self._solve_linear_system(Aop, b_flat, rtol)
            
            # T_flat, info = tfqmr(Aop, b_flat, rtol=rtol, maxiter=self.linear_maxiter)

            T = T_flat.reshape(n, p)

            T = proj_tangent_cached(X, T, Ginv=Ginv, XGinv=XGinv)

            # tangent test
            print(f"tangent test:", np.linalg.norm(X.T @ T + T.T @ X, ord="fro"))

            # SOL field Λ(X) = T(X) + N(X)
            Lambda = T + N

            # Try full step first
            X_trial = X + self.eta * Lambda
            ortho_trial = float(np.linalg.norm(X_trial.T @ X_trial - np.eye(p), ord="fro"))
            if ortho_trial <= self.epsilon:
                step = self.eta
                X = X_trial
            else:
                # Safeguard
                g_norm = float(np.linalg.norm(Lambda, ord="fro"))
                step = eta_safe(ortho, g_norm, self.epsilon)
                X = X + step * Lambda

            log["step_size"].append(float(step))

        if self.verbosity >= 1:
            print(
                f"Second-Order Landing finished: {stopping_reason} after {len(log['cost'])} iterations."
            )

        if log["time"]:
            t0 = log["time"][0]
            log["time"] = [t - t0 for t in log["time"]]

        return SOLResult(X=X, iterations=len(log["cost"]), stopping_reason=stopping_reason, log=log)


# --------- Symmetric SOL helpers for Section 4.3 ----------


def proj_tangent_cached(
    X: Array,
    V: Array,
    *,
    Ginv: Array,
    XGinv: Optional[Array] = None,
) -> Array:
    """Project V onto T_X St_{X^T X}(p,n) using cached G^{-1}."""
    if XGinv is None:
        XGinv = X @ Ginv
    return V - XGinv @ sym(X.T @ V)



# def metric_map_action_cached(
#     X: Array,
#     Z: Array,
#     *,
#     Ginv: Array,
#     XGinv: Optional[Array] = None,
# ) -> Array:
#     r"""Canonical-metric Riesz map M_X[Z].
#
#     M_X[Z] = (I - 0.5 X G^{-1} X^T) Z G^{-1},  where G = X^T X.
#
#     This cached implementation performs only matrix multiplications once G^{-1}
#     (and optionally X G^{-1}) is available.
#     """
#     if XGinv is None:
#         XGinv = X @ Ginv
#     ZGinv = Z @ Ginv
#     return ZGinv - 0.5 * XGinv @ (X.T @ ZGinv)



def exact_tangent_hessian_action_cached(
    X: Array,
    V: Array,
    grad_f: Callable[[Array], Array],
    hess_f: Callable[[Array, Array], Array],
    *,
    XTX: Array,
    Ginv: Array,
    XGinv: Array,
    G: Array,
    XtG: Array,
    project_input: bool = False,
) -> Array:
    r"""Exact Riemannian Hessian action from the paper's Proposition 4.1.

    For tangent ``V``,

        Hess f(X)[V]
          = Π_T^X(
                2*skew(∇²f(X)[V] X^T) X
              + 2*skew(∇f(X) V^T) X
              + 2*skew(∇f(X) X^T) V
              - (I + P_X) Xi_X(V)
            ),

    where ``P_X = X (X^T X)^{-1} X^T`` and ``Xi_X(V)`` is the Levi-Civita
    connection correction term from equation (4.8) / ``eq:Riemannian_Hessian``.
    ``G`` denotes the ambient Euclidean gradient ``∇f(X)``.
    """
    if project_input:
        V = proj_tangent_cached(X, V, Ginv=Ginv, XGinv=XGinv)

    HV = hess_f(X, V)
    XtV = X.T @ V
    VtX = XtV.T
    GtV = G.T @ V
    HVtX = HV.T @ X

    # 2*skew(HV X^T)X + 2*skew(G V^T)X + 2*skew(G X^T)V.
    # Group the two gradient terms so that G @ (XtV + VtX) vanishes
    # automatically for tangent V (XtV skew-symmetric).
    hv_term = HV @ XTX - X @ HVtX
    grad_terms =- V @ XtG.T - X @ GtV

    # Riemannian gradient G(X) = 2*skew(∇f(X) X^T)X.
    # Compute all p×p ingredients from cached small matrices, avoiding an
    # explicit construction of rgrad = G @ XTX - X @ XtG.T.
    Xt_rgrad = XtG @ XTX - XTX @ XtG.T
    VQXt_rgrad = V @ (Ginv @ Xt_rgrad)
    rgradQXtV = G @ XtV - X @ (XtG.T @ (Ginv @ XtV))
    Vt_rgrad = GtV.T @ XTX - VtX @ XtG.T
    # rgrad_t_V = XTX @ GtV - XtG @ XtV
    xi = (
        0.5 * (VQXt_rgrad + rgradQXtV)
        + 0.25 * XGinv @ (Vt_rgrad + Vt_rgrad.T)
    )
    connection = xi + XGinv @ (X.T @ xi)

    ambient_hess = hv_term + grad_terms - connection
    return proj_tangent_cached(X, ambient_hess, Ginv=Ginv, XGinv=XGinv)



# def symmetric_tangential_operator_action_cached(
#     X: Array,
#     V: Array,
#     grad_f: Callable[[Array], Array],
#     hess_f: Callable[[Array, Array], Array],
#     *,
#     XTX: Array,
#     Ginv: Array,
#     XGinv: Array,
#     G: Array,
#     XtG: Array,
#     project_input: bool = False,
# ) -> Array:
#     r"""Symmetric operator S_X[V] = M_X(Hess f(X)[V])."""
#     HV = exact_tangent_hessian_action_cached(
#         X,
#         V,
#         grad_f,
#         hess_f,
#         XTX=XTX,
#         Ginv=Ginv,
#         XGinv=XGinv,
#         G=G,
#         XtG=XtG,
#         project_input=project_input,
#     )
#     return metric_map_action_cached(X, HV, Ginv=Ginv, XGinv=XGinv)


class SecondOrderLandingSymmetric:
    r"""Second-order landing based on the symmetric tangential system from Section 4.3.

    At each iteration we form
        N(X) = -0.5 * grad_N(X),
        b(X) = -T1(X) - A_N(X)[N(X)],
    and solve the symmetric system
        S_X[T(X)] = M_X[b(X)],
    where
        S_X = M_X \circ Hess f(X).

    The tangential subproblem is solved by boundary-step truncated CG (tCG).
    If negative curvature is detected, tCG returns a boundary step instead of
    discarding the direction. An optional MINRES fallback is only used when
    tCG breaks down numerically.
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.75,
        eta: float = 1.0,
        tol: float = 1e-12,
        max_iter: int = 200,
        linear_rtol: Optional[float] = None,
        theta: float = 1.0,
        zeta_max: float = 1e-2,
        linear_maxiter: int = 200,
        tcg_delta_factor: float = 10.0,
        tcg_delta_min: float = 1e-16,
        fallback_to_minres: bool = True,
        verbosity: int = 1,
    ):
        if not (0 < epsilon <= 0.75):
            raise ValueError("epsilon is too big.")
        if theta <= 0:
            raise ValueError("theta must be positive.")
        if not (0.0 < zeta_max < 1.0):
            raise ValueError("zeta_max must be in (0, 1).")
        if tcg_delta_factor <= 0.0:
            raise ValueError("tcg_delta_factor must be positive.")
        if tcg_delta_min <= 0.0:
            raise ValueError("tcg_delta_min must be positive.")

        self.epsilon = float(epsilon)
        self.eta = float(eta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.linear_rtol = linear_rtol
        self.theta = float(theta)
        self.zeta_max = float(zeta_max)
        self.linear_maxiter = int(linear_maxiter)
        self.tcg_delta_factor = float(tcg_delta_factor)
        self.tcg_delta_min = float(tcg_delta_min)
        self.fallback_to_minres = bool(fallback_to_minres)
        self.verbosity = int(verbosity)

    def run(
        self,
        *,
        n: int,
        p: int,
        cost: Callable[[Array], float],
        grad_f: Callable[[Array], Array],
        hess_f: Callable[[Array, Array], Array],
        X0: Optional[Array] = None,
        manifold: Optional[Any] = None,
        NS_order: int = 1,
        seed: Optional[int] = None,
        callback: Optional[Callable[[Array, int], Any]] = None,
    ) -> SOLResult:
        rng = np.random.default_rng(seed)
        if X0 is None:
            if manifold is not None and hasattr(manifold, "random_point"):
                X = manifold.random_point()
            else:
                X = rng.standard_normal((n, p))
        else:
            X = np.array(X0, dtype=float, copy=True)

        log: Dict[str, list] = {
            "time": [],
            "cost": [],
            "ortho_error": [],
            "tangent_residual": [],
            "landing_residual": [],
            "step_size": [],
            "solver_info": [],
            "solver_flag": [],
            "rtol": [],
            "rhs_norm": [],
            "delta_t": [],
        }
        t_start = time.perf_counter()
        stopping_reason = "max_iter"

        def landing_field_first_order(Xk: Array) -> Array:
            return T1(Xk, grad_f) + 0.5 * grad_N(Xk)

        for k in range(1, self.max_iter + 1):
            XTX = X.T @ X
            chol_G = cho_factor(XTX, overwrite_a=False, check_finite=False)
            Ginv = cho_solve(chol_G, np.eye(p))
            XGinv = X @ Ginv
            G = grad_f(X)
            XtG = X.T @ G

            t1 = T1(X, grad_f)
            t1_norm = float(np.linalg.norm(t1, ord="fro"))
            gn = grad_N(X)
            ortho = float(np.linalg.norm(XTX - np.eye(p), ord="fro"))
            lam1 = landing_field_first_order(X)
            lam1_norm = float(np.linalg.norm(lam1, ord="fro"))

            elapsed = float(time.perf_counter() - t_start)
            log["time"].append(elapsed)
            log["cost"].append(float(cost(X)))
            log["ortho_error"].append(ortho)
            log["tangent_residual"].append(t1_norm)
            log["landing_residual"].append(lam1_norm)

            if self.verbosity >= 2:
                print(
                    f"[SOL-sym {k:4d}] t={elapsed:8.3f}s  cost={log['cost'][-1]:.6e}  "
                    f"||T1||={t1_norm:.3e}  ||X^T X-I||={ortho:.3e}  ||Λ1||={lam1_norm:.3e}"
                )

            if callable(callback):
                try:
                    cb_ret = callback(X, k)
                except Exception as ex:
                    raise RuntimeError(
                        f"Callback raised an exception at iteration {k}: {ex}"
                    ) from ex
                if cb_ret is True or cb_ret == "stop":
                    stopping_reason = "callback"
                    break

            if lam1_norm <= self.tol:
                stopping_reason = "tol"
                break

            # normal step
            N = NS_displacement_r(X, NS_order)


            b = -t1 - AN_action(X, N, grad_f, hess_f, XTX=XTX, G=G, XtG=XtG)
            b = proj_tangent_cached(X, b, Ginv=Ginv, XGinv=XGinv)
            # check that b is already tangent (should be by construction)
            ##################### CG or MINRES without the metric g #############
            # rhs = b
            # rhs_flat = rhs.ravel()
            # rhs_norm = float(np.linalg.norm(rhs_flat))
            # log["rhs_norm"].append(rhs_norm)
            #
            # dim = n * p
            #
            # def matvec(v_flat: Array) -> Array:
            #     V = v_flat.reshape(n, p)
            #     HessV = exact_tangent_hessian_action_cached(X, V, grad_f, hess_f, XTX=XTX, Ginv=Ginv, XGinv=XGinv, G=G, XtG=XtG)
            #     return HessV.ravel()
            #
            # Aop = LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec, dtype=float)
            #
            # if self.linear_rtol is None:
            #     if rhs_norm == 0.0:
            #         rtol = 0.0
            #     else:
            #         rtol = min(self.zeta_max, rhs_norm ** self.theta)
            # else:
            #     rtol = float(self.linear_rtol)
            # log["rtol"].append(rtol)
            #
            #
            #
            # print("minres info:", info)
            #
            #             T, info = g_metric_cg(
            #                 X,
            #                 Hessop,
            #                 b,
            #                 rtol=rtol,
            #                 maxiter=self.linear_maxiter,
            #             )
            #
            # T = T_flat.reshape(n, p)
            # # T = proj_tangent_cached(X, T, Ginv=Ginv, XGinv=XGinv)

            ##################### CG or MINRES without the metric g #############

            ##################### CG or MINRES with the metric g #############

            def Hessop(V: Array) -> Array:
                HessV = exact_tangent_hessian_action_cached(X, V, grad_f, hess_f, XTX=XTX, Ginv=Ginv, XGinv=XGinv, G=G,
                                                            XtG=XtG)
                return HessV

            b_norm = float(np.linalg.norm(b, ord="fro"))

            if self.linear_rtol is None:
                if b_norm == 0.0:
                    rtol = 0.0
                else:
                    rtol = min(self.zeta_max, b_norm ** self.theta)
            else:
                rtol = float(self.linear_rtol)

            log["rtol"].append(rtol)
            T, info = g_metric_minres(
                X,
                Hessop,
                b,
                rtol=rtol,
                # atol = 1e-14,
                maxiter=self.linear_maxiter,
            )

            # # check T
            # error =0.5 * (X.T @ T + T.T @ X)
            # print(f"T orthogonality check (should be close to 0): {np.linalg.norm(error, ord='fro')}")
            ##################### CG or MINRES with the metric g #############
            # T = proj_tangent_cached(X, T, Ginv=Ginv, XGinv=XGinv)

            Lambda = T + N

            X_trial = X + self.eta * Lambda
            ortho_trial = float(
                np.linalg.norm(X_trial.T @ X_trial - np.eye(p), ord="fro")
            )
            if ortho_trial <= self.epsilon:
                step = self.eta
                X = X_trial
            else:
                g_norm = float(np.linalg.norm(Lambda, ord="fro"))
                step = eta_safe(ortho, g_norm, self.epsilon)
                X = X + step * Lambda

            log["step_size"].append(float(step))

        if self.verbosity >= 1:
            print(
                f"SecondOrderLandingSymmetric finished: {stopping_reason} after {len(log['cost'])} iterations."
            )

        if log["time"]:
            t0 = log["time"][0]
            log["time"] = [t - t0 for t in log["time"]]

        return SOLResult(
            X=X,
            iterations=len(log["cost"]),
            stopping_reason=stopping_reason,
            log=log,
        )


# def _boundary_step_length(x: np.ndarray, p: np.ndarray, delta: float) -> float:
#     """Return tau >= 0 such that ||x + tau p||_2 = delta.

#     If no positive root exists numerically, return 0.
#     """
#     if delta <= 0.0:
#         return 0.0

#     xx = float(np.dot(x, x))
#     pp = float(np.dot(p, p))
#     xp = float(np.dot(x, p))

#     if pp <= 0.0:
#         return 0.0

#     # Solve ||x + tau p||^2 = delta^2:
#     #   pp * tau^2 + 2*xp*tau + (xx - delta^2) = 0
#     disc = xp * xp + pp * (delta * delta - xx)
#     if disc < 0.0:
#         disc = 0.0

#     sqrt_disc = np.sqrt(disc)
#     tau1 = (-xp + sqrt_disc) / pp
#     tau2 = (-xp - sqrt_disc) / pp

#     # choose the largest nonnegative root
#     candidates = [tau for tau in (tau1, tau2) if tau >= 0.0]
#     if not candidates:
#         return 0.0
#     return max(candidates)


# def tcg(
#     A,
#     b,
#     *,
#     rtol: float = 1e-6,
#     maxiter: int = 200,
#     x0: Optional[np.ndarray] = None,
#     delta: Optional[float] = None,
# ):
#     """tCG with boundary step only for negative curvature.

#     Parameters
#     ----------
#     A : LinearOperator-like
#         Must provide A.matvec(v).
#     b : ndarray
#         Right-hand side.
#     rtol : float
#         Relative residual tolerance.
#     maxiter : int
#         Maximum number of CG iterations.
#     x0 : ndarray or None
#         Optional initial guess.
#     delta : float or None
#         Boundary radius used only when negative curvature is detected.
#         If None, a default radius is chosen automatically.

#     Returns
#     -------
#     x : ndarray
#         Approximate solution.
#     info : int
#         0 : converged by residual
#         1 : reached maxiter
#         2 : numerical breakdown / failed negative-curvature step
#         4 : accepted boundary step due to negative curvature
#     """
#     b = np.asarray(b, dtype=float).ravel()

#     if x0 is None:
#         x = np.zeros_like(b)
#     else:
#         x = np.asarray(x0, dtype=float).ravel().copy()

#     b_norm = float(np.linalg.norm(b))
#     if b_norm == 0.0:
#         return x, 0

#     # delta is only needed if negative curvature is encountered
#     if delta is None:
#         delta = max(1.0, b_norm)
#     else:
#         delta = float(delta)
#         if delta <= 0.0:
#             raise ValueError("delta must be positive.")

#     r = b - A.matvec(x)
#     r_norm = float(np.linalg.norm(r))
#     if r_norm <= rtol * b_norm:
#         return x, 0

#     p = r.copy()
#     rr = float(np.dot(r, r))

#     for _ in range(int(maxiter)):
#         Ap = A.matvec(p)
#         pAp = float(np.dot(p, Ap))

#         # Only use boundary-step when negative curvature is detected
#         if (not np.isfinite(pAp)) or pAp <= 0.0:
#             tau = _boundary_step_length(x, p, delta)
#             if tau > 0.0:
#                 return x + tau * p, 4
#             return x, 2

#         alpha = rr / pAp

#         # Take the full CG step without radius truncation
#         x = x + alpha * p
#         r = r - alpha * Ap

#         r_norm = float(np.linalg.norm(r))
#         if r_norm <= rtol * b_norm:
#             return x, 0

#         rr_new = float(np.dot(r, r))
#         if (not np.isfinite(rr_new)) or rr_new <= 0.0:
#             return x, 2

#         beta = rr_new / rr
#         p = r + beta * p
#         rr = rr_new

#     return x, 1




def _boundary_step_length(x: np.ndarray, p: np.ndarray, delta: float) -> Optional[float]:
    """
    Return tau >= 0 such that ||x + tau p|| = delta.

    If no numerically valid nonnegative root exists, return None.
    """
    x = np.asarray(x, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()

    pp = float(np.dot(p, p))
    if (not np.isfinite(pp)) or pp <= 0.0:
        return None

    xp = float(np.dot(x, p))
    xx = float(np.dot(x, x))
    dd = float(delta * delta)

    # Solve: ||x + tau p||^2 = delta^2
    # => (p^T p) tau^2 + 2 (x^T p) tau + (x^T x - delta^2) = 0
    c = xx - dd
    disc = xp * xp - pp * c

    # Small negative values can happen from roundoff
    tol = 1e-14 * max(1.0, abs(xp * xp), abs(pp * c))
    if disc < -tol:
        return None

    disc = max(disc, 0.0)
    sqrt_disc = float(np.sqrt(disc))

    tau1 = (-xp - sqrt_disc) / pp
    tau2 = (-xp + sqrt_disc) / pp

    candidates = [tau for tau in (tau1, tau2) if tau >= -1e-14]
    if not candidates:
        return None

    # For an interior point, tau2 is the forward boundary hit.
    tau = max(candidates)
    return max(0.0, tau)


def tcg(
    A,
    b,
    *,
    rtol: float = 1e-6,
    maxiter: int = 200,
    x0: Optional[np.ndarray] = None,
    delta: Optional[float] = None,
):
    b = np.asarray(b, dtype=float).ravel()

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = np.asarray(x0, dtype=float).ravel().copy()

    b_norm = float(np.linalg.norm(b))
    if b_norm == 0.0:
        return x, 0

    if delta is None:
        delta = max(1.0, 10 * b_norm)
    else:
        delta = float(delta)
        if delta <= 0.0:
            raise ValueError("delta must be positive.")

    x_norm = float(np.linalg.norm(x))
    if x_norm > delta * (1.0 + 1e-12):
        raise ValueError("x0 must satisfy ||x0|| <= delta.")

    r = b - A.matvec(x)
    r = np.asarray(r, dtype=float).ravel()

    if not np.all(np.isfinite(r)):
        return x, 2

    r_norm = float(np.linalg.norm(r))
    if r_norm <= rtol * b_norm:
        return x, 0

    p = r.copy()
    rr = float(np.dot(r, r))
    if (not np.isfinite(rr)) or rr < 0.0:
        return x, 2

    for _ in range(int(maxiter)):
        Ap = A.matvec(p)
        Ap = np.asarray(Ap, dtype=float).ravel()

        if not np.all(np.isfinite(Ap)):
            return x, 2

        pAp = float(np.dot(p, Ap))
        if pAp <= 0.0:
            tau = _boundary_step_length(x, p, delta)
            if tau is None:
                return x, 2
            return x + tau * p, 4

        alpha = rr / pAp
        if (not np.isfinite(alpha)) or alpha < 0.0:
            return x, 2

        x_next = x + alpha * p
        x_next_norm = float(np.linalg.norm(x_next))
        if x_next_norm >= delta * (1.0 - 1e-14):
            tau = _boundary_step_length(x, p, delta)
            if tau is None:
                return x, 2
            return x + tau * p, 3

        x = x_next
        r = r - alpha * Ap

        if not np.all(np.isfinite(r)):
            return x, 2

        r_norm = float(np.linalg.norm(r))
        if r_norm <= rtol * b_norm:
            return x, 0

        rr_new = float(np.dot(r, r))
        if (not np.isfinite(rr_new)) or rr_new < 0.0:
            return x, 2
        if rr_new == 0.0:
            return x, 0

        beta = rr_new / rr
        if (not np.isfinite(beta)) or beta < 0.0:
            return x, 2

        p = r + beta * p
        rr = rr_new

    return x, 1

class AltSecondOrderLanding:
    """Alternating Second-Order Landing (SOL) solver.

    Parameters (paper-aligned):
        epsilon: safety region radius ε.
        eta: nominal stepsize η (we attempt η=1 by default; safeguard may shrink it).
        tol: stopping tolerance on ||Λ_1(X)||_F.
        max_iter: maximum iterations.
        linear_rtol: GMRES relative tolerance for the tangential solve.
            If None, uses an inexact-forcing rule similar to the original code.
        theta: forcing-term order θ for the inexact tangential solve.
        zeta_max: maximum forcing parameter ζ_max.
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.75,
        eta: float = 1.0,
        tol: float = 1e-12,
        max_iter: int = 200,
        linear_solver: str = "lgmres",
        linear_solver_options: Optional[dict] = None,
        linear_rtol: Optional[float] = None,
        theta: float = 1.0,
        zeta_max: float = 1e-2,
        linear_maxiter: int = 200,
        verbosity: int = 1,
    ):
        if not (0 < epsilon <= 0.75):
            raise ValueError("epsilon is too big.")
        self.epsilon = float(epsilon)
        self.eta = float(eta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.linear_solver = str(linear_solver).lower()
        self.linear_solver_options = dict(linear_solver_options or {})
        self.linear_rtol = linear_rtol
        if theta <= 0:
            raise ValueError("theta must be positive.")
        if not (0.0 < zeta_max < 1.0):
            raise ValueError("zeta_max must be in (0, 1).")
        self.theta = float(theta)
        self.zeta_max = float(zeta_max)
        self.linear_maxiter = int(linear_maxiter)
        self.verbosity = int(verbosity)

    def _solve_linear_system(self, Aop, b_flat, rtol: float):
        opts = dict(self.linear_solver_options)
        maxiter = opts.pop("maxiter", self.linear_maxiter)
        atol = opts.pop("atol", 1e-12)

        solver = self.linear_solver

        if solver == "lgmres":
            x, info = lgmres(Aop, b_flat, atol=atol, rtol=rtol, maxiter=maxiter, **opts)

        elif solver == "gmres":
            restart = opts.pop("restart", 20)
            x, info = gmres(
                Aop, b_flat, atol=atol, rtol=rtol,
                restart=restart, maxiter=maxiter, **opts
            )

        elif solver == "bicgstab":
            x, info = bicgstab(Aop, b_flat, atol=atol, rtol=rtol, maxiter=maxiter, **opts)

        elif solver == "cgs":
            x, info = cgs(Aop, b_flat, atol=atol, rtol=rtol, maxiter=maxiter, **opts)

        elif solver == "gcrotmk":
            x, info = gcrotmk(Aop, b_flat, atol=atol, rtol=rtol, maxiter=maxiter, **opts)

        else:
            raise ValueError(f"Unknown linear_solver: {solver}")

        return x, info


    def run(
        self,
        *,
        n: int,
        p: int,
        cost: Callable[[Array], float],
        grad_f: Callable[[Array], Array],
        hess_f: Callable[[Array, Array], Array],
        X0: Optional[Array] = None,
        manifold: Optional[Any] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable[[Array, int], Any]] = None,
    ) -> SOLResult:
        """Run SOL.

        If X0 is None and a Pymanopt manifold is provided, uses manifold.random_point().
        Otherwise uses a random Gaussian matrix.
        """
        rng = np.random.default_rng(seed)
        if X0 is None:
            if manifold is not None and hasattr(manifold, "random_point"):
                X = manifold.random_point()
            else:
                X = rng.standard_normal((n, p))
        else:
            X = np.array(X0, dtype=float, copy=True)

        log: Dict[str, list] = {
            "time": [],
            "cost": [],
            "ortho_error": [],
            "tangent_residual": [],  # ||T1(X)||_F
            "landing_residual": [],  # ||Λ1(X)||_F with λ=1/2
            "step_size": [],
            "solver_info": [],
            "solver_flag": [],
            "rtol": [],
        }
        t_start = time.perf_counter()

        def landing_field_first_order(Xk: Array) -> Array:
            # Λ1(X) = T1(X) + λ ∇N(X) with λ = 1/2 (paper's SOL choice)
            return T1(Xk, grad_f) + 0.5 * grad_N(Xk)

        stopping_reason = "max_iter"

        # Callback contract (per-iterate):
        #   callback(X, k) -> Any
        # Called once per iteration with the current iterate X and iteration index k.
        # If it returns True or the string "stop", the optimizer terminates early.

        for k in range(1, self.max_iter + 1):
            gn = grad_N(X)
            N = -0.5 * gn
            X = X + N

            XTX = X.T @ X
            G = grad_f(X)
            XtG = X.T @ G

            t1 = T1(X, grad_f)
            t1_norm = float(np.linalg.norm(t1, ord="fro"))
            ortho = float(np.linalg.norm(XTX - np.eye(p), ord="fro"))
            lam1 = landing_field_first_order(X)
            lam1_norm = float(np.linalg.norm(lam1, ord="fro"))


            log["cost"].append(float(cost(X)))
            log["ortho_error"].append(ortho)
            log["tangent_residual"].append(t1_norm)
            log["landing_residual"].append(lam1_norm)

            elapsed = float(time.perf_counter() - t_start)
            log["time"].append(elapsed)

            if self.verbosity >= 2:
                print(
                    f"[AltSOL {k:4d}] t={elapsed:8.3f}s  cost={log['cost'][-1]:.6e}  "
                    f"||T1||={log['tangent_residual'][-1]:.3e}  "
                    f"||X^T X-I||={ortho:.3e}  ||Λ1||={lam1_norm:.3e}"
                )

            # Per-iteration callback hook (only expose current iterate X)
            if callable(callback):
                try:
                    cb_ret = callback(X, k)
                except Exception as ex:
                    raise RuntimeError(f"Callback raised an exception at iteration {k}: {ex}") from ex
                if cb_ret is True or cb_ret == "stop":
                    stopping_reason = "callback"
                    break

            if lam1_norm <= self.tol:
                stopping_reason = "tol"
                break

            # # Normal step (order-1 Newton–Schulz displacement)
            # N = -0.5 * gn

            # Right-hand side b(X) = -T1(X) - A_N(X)[N(X)]
            b = -t1

            # Tangential solve (symmetrized): 0.5*(A_T(X) + A_T(X)^*)[T] = b
            b_flat = b.ravel()
            dim = n * p
            def matvec(v_flat: Array) -> Array:
                V = v_flat.reshape(n, p)
                AV = AT_action(X, V, grad_f, hess_f, XTX=XTX, G=G, XtG=XtG)
                return AV.ravel()
                # AstarV = AT_adjoint_action(X, V, grad_f, hess_f, XTX=XTX)
                # return (0.5 * (AV + AstarV)).ravel()

            def rmatvec(v_flat: Array) -> Array:
                V = v_flat.reshape(n, p)
                AstarV = AT_adjoint_action(X, V, grad_f, hess_f, XTX=XTX, G=G, XtG=XtG)
                return AstarV.ravel()

            Aop = LinearOperator((dim, dim), matvec=matvec, rmatvec=rmatvec, dtype=float)

            if self.linear_rtol is None:
                b_norm = float(np.linalg.norm(b_flat))
                print(f"b_norm:{b_norm}")
                if b_norm == 0.0:
                    rtol = 0.0
                else:
                    rtol = min(self.zeta_max, b_norm ** self.theta)
            else:
                rtol = float(self.linear_rtol)
            print(f"rtol:{rtol}")
            log["rtol"].append(float(rtol))


            T_flat, info = self._solve_linear_system(Aop, b_flat, rtol)
            
            # T_flat, info = tfqmr(Aop, b_flat, rtol=rtol, maxiter=self.linear_maxiter)

            T = T_flat.reshape(n, p)


            # Try full step first
            X_trial = X + self.eta * T
            ortho_trial = float(np.linalg.norm(X_trial.T @ X_trial - np.eye(p), ord="fro"))
            if ortho_trial <= self.epsilon:
                step = self.eta
                X = X_trial
            else:
                # Safeguard
                g_norm = float(np.linalg.norm(T, ord="fro"))
                step = eta_safe(ortho, g_norm, self.epsilon, lam = 0.0)
                X = X + step * T

            log["step_size"].append(float(step))





        if self.verbosity >= 1:
            print(
                f"Alternating Second-Order Landing finished: {stopping_reason} after {len(log['cost'])} iterations."
            )

        if log["time"]:
            t0 = log["time"][0]
            log["time"] = [t - t0 for t in log["time"]]

        return SOLResult(X=X, iterations=len(log["cost"]), stopping_reason=stopping_reason, log=log)


class AltSecondOrderLandingSymmetric:
    r"""Alternating Second-order landing based on the symmetric tangential system.

    The tangential subproblem is solved by boundary-step truncated CG (tCG).
    If negative curvature is detected, tCG returns a boundary step instead of
    discarding the direction. An optional MINRES fallback is only used when
    tCG breaks down numerically.
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.75,
        eta: float = 1.0,
        tol: float = 1e-12,
        max_iter: int = 200,
        linear_rtol: Optional[float] = None,
        theta: float = 1.0,
        zeta_max: float = 1e-2,
        linear_maxiter: int = 200,
        tcg_delta_factor: float = 10.0,
        tcg_delta_min: float = 1e-16,
        fallback_to_minres: bool = True,
        verbosity: int = 1,
    ):
        if not (0 < epsilon <= 0.75):
            raise ValueError("epsilon is too big.")
        if theta <= 0:
            raise ValueError("theta must be positive.")
        if not (0.0 < zeta_max < 1.0):
            raise ValueError("zeta_max must be in (0, 1).")
        if tcg_delta_factor <= 0.0:
            raise ValueError("tcg_delta_factor must be positive.")
        if tcg_delta_min <= 0.0:
            raise ValueError("tcg_delta_min must be positive.")

        self.epsilon = float(epsilon)
        self.eta = float(eta)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.linear_rtol = linear_rtol
        self.theta = float(theta)
        self.zeta_max = float(zeta_max)
        self.linear_maxiter = int(linear_maxiter)
        self.tcg_delta_factor = float(tcg_delta_factor)
        self.tcg_delta_min = float(tcg_delta_min)
        self.fallback_to_minres = bool(fallback_to_minres)
        self.verbosity = int(verbosity)

    def run(
        self,
        *,
        n: int,
        p: int,
        cost: Callable[[Array], float],
        grad_f: Callable[[Array], Array],
        hess_f: Callable[[Array, Array], Array],
        X0: Optional[Array] = None,
        manifold: Optional[Any] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable[[Array, int], Any]] = None,
    ) -> SOLResult:
        rng = np.random.default_rng(seed)
        if X0 is None:
            if manifold is not None and hasattr(manifold, "random_point"):
                X = manifold.random_point()
            else:
                X = rng.standard_normal((n, p))
        else:
            X = np.array(X0, dtype=float, copy=True)

        log: Dict[str, list] = {
            "time": [],
            "cost": [],
            "ortho_error": [],
            "tangent_residual": [],
            "landing_residual": [],
            "step_size": [],
            "solver_info": [],
            "solver_flag": [],
            "rtol": [],
            "rhs_norm": [],
            "delta_t": [],
        }
        t_start = time.perf_counter()
        stopping_reason = "max_iter"

        def landing_field_first_order(Xk: Array) -> Array:
            return T1(Xk, grad_f) + 0.5 * grad_N(Xk)

        for k in range(1, self.max_iter + 1):
            # normal step
            gn = grad_N(X)
            N = -0.5 * gn
            X = X + N

            XTX = X.T @ X
            chol_G = cho_factor(XTX, overwrite_a=False, check_finite=False)
            Ginv = cho_solve(chol_G, np.eye(p))
            XGinv = X @ Ginv
            G = grad_f(X)
            XtG = X.T @ G

            t1 = T1(X, grad_f)
            t1_norm = float(np.linalg.norm(t1, ord="fro"))
            ortho = float(np.linalg.norm(XTX - np.eye(p), ord="fro"))
            lam1 = landing_field_first_order(X)
            lam1_norm = float(np.linalg.norm(lam1, ord="fro"))

            elapsed = float(time.perf_counter() - t_start)
            log["time"].append(elapsed)
            log["cost"].append(float(cost(X)))
            log["ortho_error"].append(ortho)
            log["tangent_residual"].append(t1_norm)
            log["landing_residual"].append(lam1_norm)

            if self.verbosity >= 2:
                print(
                    f"[AltSOL-sym {k:4d}] t={elapsed:8.3f}s  cost={log['cost'][-1]:.6e}  "
                    f"||T1||={t1_norm:.3e}  ||X^T X-I||={ortho:.3e}  ||Λ1||={lam1_norm:.3e}"
                )

            if callable(callback):
                try:
                    cb_ret = callback(X, k)
                except Exception as ex:
                    raise RuntimeError(
                        f"Callback raised an exception at iteration {k}: {ex}"
                    ) from ex
                if cb_ret is True or cb_ret == "stop":
                    stopping_reason = "callback"
                    break

            if lam1_norm <= self.tol:
                stopping_reason = "tol"
                break



            b = -t1
            # b = proj_tangent_cached(X, b, Ginv=Ginv, XGinv=XGinv)




            # T_flat, info = minres(
            #     Aop,
            #     rhs_flat,
            #     rtol=rtol,
            #     maxiter=self.linear_maxiter,
            # )
            #
            #
            # print("minres info:", info)
            #
            #
            #
            # T = T_flat.reshape(n, p)
            # T = proj_tangent_cached(X, T, Ginv=Ginv, XGinv=XGinv)

            ##################### CG or MINRES with the metric g #############

            def Hessop(V: Array) -> Array:
                HessV = exact_tangent_hessian_action_cached(X, V, grad_f, hess_f, XTX=XTX, Ginv=Ginv, XGinv=XGinv, G=G,
                                                            XtG=XtG)
                return HessV

            b_norm = float(np.linalg.norm(b, ord="fro"))

            if self.linear_rtol is None:
                if b_norm == 0.0:
                    rtol = 0.0
                else:
                    rtol = min(self.zeta_max, b_norm ** self.theta)
            else:
                rtol = float(self.linear_rtol)

            log["rtol"].append(rtol)
            T, info = g_metric_cg(
                X,
                Hessop,
                b,
                rtol=rtol,
                atol=1e-14,
                maxiter = self.linear_maxiter,
            )

            ##################### CG or MINRES with the metric g #############

            X_trial = X + self.eta * T
            ortho_trial = float(
                np.linalg.norm(X_trial.T @ X_trial - np.eye(p), ord="fro")
            )
            if ortho_trial <= self.epsilon:
                step = self.eta
                X = X_trial
            else:
                g_norm = float(np.linalg.norm(T, ord="fro"))
                step = eta_safe(ortho, g_norm, self.epsilon, lam = 0.0)
                X = X + step * T

            log["step_size"].append(float(step))




        if self.verbosity >= 1:
            print(
                f"AltSecondOrderLandingSymmetric finished: {stopping_reason} after {len(log['cost'])} iterations."
            )

        if log["time"]:
            t0 = log["time"][0]
            log["time"] = [t - t0 for t in log["time"]]

        return SOLResult(
            X=X,
            iterations=len(log["cost"]),
            stopping_reason=stopping_reason,
            log=log,
        )
