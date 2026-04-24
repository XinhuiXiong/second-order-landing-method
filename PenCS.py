"""
PenCS (Algorithm 2) for PenC (Xiao et al., 2022).

Logs per-iteration:
    time, cost, ortho_error, tangent_residual, step_size

User provides:
    - cost_f(X): scalar objective f(X) (optional but recommended for logging/line-search)
    - grad_f(X): gradient ∇f(X)
    - hess_f(X, V): Hessian action ∇^2 f(X)[V]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any
import time

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve, factorized
from scipy.sparse.linalg import LinearOperator, gmres, cg, minres

Array = np.ndarray


# ----------------------------- helpers -----------------------------

def sym(A: Array) -> Array:
    """Φ(A) = (A + A^T)/2."""
    return 0.5 * (A + A.T)


def fro_norm(A: Array) -> float:
    return float(np.linalg.norm(A, ord="fro"))


def inner(A: Array, B: Array) -> float:
    """Frobenius inner product <A,B> = tr(A^T B)."""
    return float(np.tensordot(A, B, axes=([0, 1], [0, 1])))


def project_to_ball(X: Array, K: float) -> Array:
    """Project onto Frobenius ball {||X||_F <= K} by radial scaling."""
    nrm = fro_norm(X)
    if nrm <= K or nrm == 0.0:
        return X
    return (K / nrm) * X


# -------------------- paper-specific operators ---------------------

def Lambda_from_grad(X: Array, grad_f_X: Array) -> Array:
    """Λ(X) = Φ(∇f(X)^T X)."""
    return sym(grad_f_X.T @ X)


def ortho_error_val(X: Array) -> float:
    p = X.shape[1]
    return fro_norm(X.T @ X - np.eye(p, dtype=X.dtype))


def tangent_residual_val(X: Array, grad_f_X: Array) -> float:
    """
    Paper uses substationarity ||∇f(X) - XΛ(X)||_F in experiments.
    """
    Lam = Lambda_from_grad(X, grad_f_X)
    return fro_norm(grad_f_X - X @ Lam)


def h_value(
    X: Array,
    beta: float,
    cost_f: Callable[[Array], float],
    grad_f: Callable[[Array], Array],
) -> float:
    """
    Merit function h(X) = f(X) - 1/2 tr(Λ(X)(X^T X - I)) + (β/4)||X^T X - I||_F^2.
    """
    p = X.shape[1]
    G = grad_f(X)
    Lam = Lambda_from_grad(X, G)
    XtX_minus_I = X.T @ X - np.eye(p, dtype=X.dtype)

    term1 = float(cost_f(X))
    term2 = 0.5 * float(np.sum(Lam * XtX_minus_I))
    term3 = (beta / 4.0) * float(np.sum(XtX_minus_I * XtX_minus_I))
    return term1 - term2 + term3


def grad_h(
    X: Array,
    beta: float,
    grad_f: Callable[[Array], Array],
    hess_f: Callable[[Array, Array], Array],
) -> Array:
    """
    Gradient of h(X) from paper (page 7, before Proposition 2.1):
        ∇h(X) = ∇f(X) - XΛ(X)
              - 1/2 ∇f(X)(X^T X - I)
              - 1/2 ∇^2 f(X)[ X(X^T X - I) ]
              + β X(X^T X - I)
    """
    n, p = X.shape
    G = grad_f(X)
    Lam = Lambda_from_grad(X, G)
    XtX_minus_I = X.T @ X - np.eye(p, dtype=X.dtype)

    HX_term = hess_f(X, X @ XtX_minus_I)
    return (
        G
        - X @ Lam
        - 0.5 * (G @ XtX_minus_I)
        - 0.5 * HX_term
        + beta * (X @ XtX_minus_I)
    )


def W_action(
    X: Array,
    D: Array,
    beta: float,
    grad_f: Callable[[Array], Array],
    hess_f: Callable[[Array, Array], Array],
) -> Array:
    """
    Approximate Hessian operator W(X)[D] from paper (5.1)-(5.3):
        W(X)[D] = ∇^2 f(X)[D] - DΛ(X)
                  - X( Φ(D^T∇f(X)) + Φ(X^T∇^2 f(X)[D]) + 2βΦ(X^T D) )
                  - ( ∇f(X)Φ(D^T X) + ∇^2 f(X)[ X Φ(X^T D) ] )
    """
    G = grad_f(X)
    Lam = Lambda_from_grad(X, G)
    HD = hess_f(X, D)

    # (5.1)
    out = HD - D @ Lam

    # (5.2)
    term52 = sym(D.T @ G) + sym(X.T @ HD) + 2.0 * beta * sym(X.T @ D)
    out = out - X @ term52

    # (5.3)
    term53a = G @ sym(D.T @ X)
    term53b = hess_f(X, X @ sym(X.T @ D))
    out = out - (term53a + term53b)

    return out


# ------------------------- linear solve ----------------------------

def solve_inexact_newton(
    X: Array,
    rhs: Array,
    beta: float,
    grad_f: Callable[[Array], Array],
    hess_f: Callable[[Array, Array], Array],
    *,
    reg_mu: float = 0.0,
    rtol: float = 1e-8,
    maxiter: int = 200,
) -> Tuple[Array, int]:
    """
    Approximately solve (W(X) + μ I) D = rhs via GMRES.
    Returns (D, info). info==0 means converged.
    """
    n, p = X.shape
    dim = n * p

    def matvec(v: Array) -> Array:
        D = v.reshape(n, p)
        WD = W_action(X, D, beta, grad_f, hess_f)
        if reg_mu != 0.0:
            WD = WD + reg_mu * D
        return WD.reshape(dim)

    A = LinearOperator((dim, dim), matvec=matvec, dtype=float)
    b = rhs.reshape(dim)

    sol, info = cg(A, b, rtol=rtol, maxiter=maxiter)
    D = sol.reshape(n, p)
    return D, int(info)


# --------------------------- optimizer -----------------------------

@dataclass
class PenCSResult:
    X: Array
    iterations: int
    stopping_reason: str
    log: Dict[str, list]


class PenCS:
    """
    PenCS solver (Algorithm 2).

    Parameters:
        beta: penalty parameter β
        K: radius of Frobenius ball B_K (K > sqrt(p))
        eta0: default step size (paper experiments often use 1)
        tol: stopping tolerance on tangent_residual = ||∇f - XΛ||_F
        ortho_tol: optional feasibility tolerance on ||X^T X - I||_F
        reg_mu: Tikhonov regularization for W(X) when GMRES is unstable
        backtracking: enable Armijo backtracking on h(X) (requires cost_f)
    """

    def __init__(
        self,
        *,
        beta: float,
        K: float,
        eta0: float = 1.0,
        tol: float = 1e-12,
        ortho_tol: Optional[float] = None,
        max_iter: int = 200,
        linear_rtol: Optional[float] = None,
        linear_maxiter: int = 200,
        theta: float = 1.0,
        zeta_max: float = 1e-2,
        reg_mu: float = 1e-8,
        backtracking: bool = False,
        armijo_c1: float = 1e-4,
        bt_shrink: float = 0.5,
        min_eta: float = 1e-16,
        verbosity: int = 1,
    ):
        if beta <= 0:
            raise ValueError("beta must be positive.")
        if K <= 0:
            raise ValueError("K must be positive.")
        if eta0 <= 0:
            raise ValueError("eta0 must be positive.")
        self.beta = float(beta)
        self.K = float(K)
        self.eta0 = float(eta0)
        self.tol = float(tol)
        self.ortho_tol = ortho_tol if ortho_tol is None else float(ortho_tol)
        self.max_iter = int(max_iter)
        self.linear_rtol = linear_rtol
        self.linear_maxiter = int(linear_maxiter)
        self.theta = float(theta)
        self.zeta_max = float(zeta_max)
        self.reg_mu = float(reg_mu)
        self.backtracking = bool(backtracking)
        self.armijo_c1 = float(armijo_c1)
        self.bt_shrink = float(bt_shrink)
        self.min_eta = float(min_eta)
        self.verbosity = int(verbosity)

    def run(
            self,
            *,
            n: int,
            p: int,
            grad_f: Callable[[Array], Array],
            hess_f: Callable[[Array, Array], Array],
            cost_f: Optional[Callable[[Array], float]] = None,
            X0: Optional[Array] = None,
            seed: Optional[int] = None,
    ) -> PenCSResult:

        rng = np.random.default_rng(seed)
        if X0 is None:
            X = rng.standard_normal((n, p))
        else:
            X = np.array(X0, dtype=float, copy=True)

        # Ensure inside ball
        X = project_to_ball(X, self.K)

        log: Dict[str, list] = {
            "time": [],
            "cost": [],
            "ortho_error": [],
            "tangent_residual": [],
            "step_size": [],
        }

        t_start = time.perf_counter()
        stopping_reason = "max_iter"

        if self.backtracking and cost_f is None:
            raise ValueError("backtracking=True requires cost_f to evaluate h(X).")

        for k in range(1, self.max_iter + 1):
            G = grad_f(X)
            tr_res = tangent_residual_val(X, G)
            ortho = ortho_error_val(X)

            # --- Logging ---
            elapsed = float(time.perf_counter() - t_start)
            f_val = float(cost_f(X)) if cost_f is not None else float("nan")

            log["time"].append(elapsed)
            log["cost"].append(f_val)
            log["ortho_error"].append(ortho)
            log["tangent_residual"].append(tr_res)

            if self.verbosity >= 2:
                print(
                    f"[PenCS {k:4d}] t={elapsed:8.3f}s  f={f_val:.6e}  "
                    f"tangent={tr_res:.3e}  ortho={ortho:.3e}"
                )

            # --- Stopping Criteria ---
            stop_tr = (tr_res <= self.tol)
            stop_ortho = (True if self.ortho_tol is None else (ortho <= self.ortho_tol))
            if stop_tr and stop_ortho:
                stopping_reason = "tol"
                break

            # --- Compute Gradient of Merit Function ---
            gh = grad_h(X, self.beta, grad_f, hess_f)

            # --- Linear Solver Setup (Corrected) ---
            n, p = X.shape
            dim = n * p

            def matvec(v_flat: Array) -> Array:
                D_in = v_flat.reshape(n, p)
                WD = W_action(X, D_in, self.beta, grad_f, hess_f)
                # [FIX 1] Apply regularization: (W + mu*I) D
                if self.reg_mu != 0.0:
                    WD = WD + self.reg_mu * D_in
                return WD.ravel()

            A_op = LinearOperator((dim, dim), matvec=matvec, dtype=float)
            b_flat = gh.ravel()
            b_norm = float(np.linalg.norm(b_flat))

            # Adaptive tolerance for inexact Newton
            if self.linear_rtol is None:
                if b_norm == 0.0:
                    rtol = 0.0
                else:
                    rtol = min(self.zeta_max, b_norm ** self.theta)
            else:
                rtol = float(self.linear_rtol)

            # Solve (W + mu*I) D = grad_h
            # Using MINRES because W might be indefinite
            sol_flat, info = minres(A_op, b_flat, rtol=rtol, maxiter=self.linear_maxiter)
            D = sol_flat.reshape(n, p)

            # [FIX 2] Robustness check: Fallback to gradient if solver failed
            if info != 0 or not np.all(np.isfinite(D)):
                if self.verbosity >= 2:
                    print(f"    Linear solver failed (info={info}), using gradient.")
                D = gh

            # [FIX 3] Ensure Descent Direction
            # If (W+mu*I) is PD, slope = <g, (W+mu*I)^-1 g> > 0.
            # We want to perform X_new = X - eta * D.
            # So we need D to be a descent direction for h(X), i.e., <grad_h, -D> < 0 => <grad_h, D> > 0.
            slope = inner(gh, D)
            if slope <= 1e-16:  # Slightly larger than 0 to avoid numerical noise
                if self.verbosity >= 2:
                    print("    Direction is not descent (slope <= 0), using gradient.")
                D = gh
                slope = inner(gh, gh)

            # --- Step-size Selection (Backtracking) ---
            eta = self.eta0
            if self.backtracking:
                h0 = h_value(X, self.beta, cost_f, grad_f)
                while eta >= self.min_eta:
                    X_trial = X - eta * D
                    X_trial = project_to_ball(X_trial, self.K)
                    h1 = h_value(X_trial, self.beta, cost_f, grad_f)

                    # Armijo condition
                    if h1 <= h0 - self.armijo_c1 * eta * slope:
                        break
                    eta *= self.bt_shrink

                # If backtracking failed to find step, take a small gradient step
                if eta < self.min_eta:
                    if self.verbosity >= 2:
                        print("    Backtracking failed, taking small gradient step.")
                    eta = self.min_eta
                    D = gh

            log["step_size"].append(float(eta))

            # --- Update ---
            X_next = X - eta * D
            X_next = project_to_ball(X_next, self.K)
            X = X_next

        return PenCSResult(
            X=X,
            iterations=len(log["time"]),
            stopping_reason=stopping_reason,
            log=log,
        )


#     def run(
#         self,
#         *,
#         n: int,
#         p: int,
#         grad_f: Callable[[Array], Array],
#         hess_f: Callable[[Array, Array], Array],
#         cost_f: Optional[Callable[[Array], float]] = None,
#         X0: Optional[Array] = None,
#         seed: Optional[int] = None,
#     ) -> PenCSResult:
#
#         rng = np.random.default_rng(seed)
#         if X0 is None:
#             X = rng.standard_normal((n, p))
#         else:
#             X = np.array(X0, dtype=float, copy=True)
#
#         # Ensure inside ball
#         X = project_to_ball(X, self.K)
#
#         log: Dict[str, list] = {
#             "time": [],
#             "cost": [],
#             "ortho_error": [],
#             "tangent_residual": [],
#             "step_size": [],
#         }
#
#         t_start = time.perf_counter()
#         stopping_reason = "max_iter"
#
#         # If line-search is requested, we need cost_f to evaluate h(X)
#         if self.backtracking and cost_f is None:
#             raise ValueError("backtracking=True requires cost_f to evaluate h(X).")
#
#         for k in range(1, self.max_iter + 1):
#             G = grad_f(X)
#             tr_res = tangent_residual_val(X, G)
#             ortho = ortho_error_val(X)
#
#             # --- Logging ---
#             elapsed = float(time.perf_counter() - t_start)
#             f_val = float(cost_f(X)) if cost_f is not None else float("nan")
#
#             log["time"].append(elapsed)
#             log["cost"].append(f_val)
#             log["ortho_error"].append(ortho)
#             log["tangent_residual"].append(tr_res)
#
#             if self.verbosity >= 2:
#                 print(
#                     f"[PenCS {k:4d}] t={elapsed:8.3f}s  f={f_val:.6e}  "
#                     f"tangent={tr_res:.3e}  ortho={ortho:.3e}"
#                 )
#
#             # stopping
#             stop_tr = (tr_res <= self.tol)
#             stop_ortho = (True if self.ortho_tol is None else (ortho <= self.ortho_tol))
#             if stop_tr and stop_ortho:
#                 stopping_reason = "tol"
#                 break
#
#             # Compute ∇h(X)
#             gh = grad_h(X, self.beta, grad_f, hess_f)
#
#
#             n, p = X.shape
#             dim = n * p
#             def matvec(v_flat: Array) -> Array:
#                 D = v_flat.reshape(n, p)
#                 WD = W_action(X, D, self.beta, grad_f, hess_f)
#                 return WD.ravel()
#
#             A = LinearOperator((dim, dim), matvec=matvec, dtype=float)
#             b_flat = gh.ravel()
#
#             if self.linear_rtol is None:
#                 b_norm = float(np.linalg.norm(b_flat))
#                 print(f"b_norm:{b_norm}")
#                 if b_norm == 0.0:
#                     rtol = 0.0
#                 else:
#                     print(f"self.theta:{self.theta}")
#                     rtol = min(self.zeta_max, b_norm ** self.theta)
#             else:
#                 rtol = float(self.linear_rtol)
#             print(f"rtol:{rtol}")
#             sol, info = minres(A, b_flat, rtol=rtol, maxiter=self.linear_maxiter)
#             print(info)
#             D = sol.reshape(n, p)
#
#
# ########
#
#             # # If solver fails badly, fall back to gradient step (robustness)
#             # if info != 0 or not np.all(np.isfinite(D)):
#             #     D = gh
#
#             # Ensure descent direction for step X <- X - η D
#             # slope = inner(gh, D)
#             # if slope <= 0:
#             #     D = gh
#             #     slope = inner(gh, gh)
#
#             # Step-size selection
#             eta = self.eta0
#             if self.backtracking:
#                 h0 = h_value(X, self.beta, cost_f, grad_f)
#                 while eta >= self.min_eta:
#                     X_trial = X - eta * D
#                     X_trial = project_to_ball(X_trial, self.K)
#                     h1 = h_value(X_trial, self.beta, cost_f, grad_f)
#                     if h1 <= h0 - self.armijo_c1 * eta * slope:
#                         break
#                     eta *= self.bt_shrink
#
#             log["step_size"].append(float(eta))
#
#             # Update + projection to B_K (Algorithm 2 lines 5-10)
#             X_next = X - eta * D
#             X_next = project_to_ball(X_next, self.K)
#             X = X_next
#
#         return PenCSResult(
#             X=X,
#             iterations=len(log["time"]),
#             stopping_reason=stopping_reason,
#             log=log,
#         )


# --------------------------- example -------------------------------

def _example_quadratic():
    # """
    # Example:
    #     min 0.5 tr(X^T A X) s.t. X^T X = I
    # for symmetric A. Global minimizers are eigenvectors of smallest eigenvalues.
    # """
    # n, p = 200, 5
    # rng = np.random.default_rng(0)
    # A = rng.standard_normal((n, n))
    # A = 0.5 * (A + A.T) + 1e-2 * np.eye(n)
    #
    # def cost_f(X: Array) -> float:
    #     return 0.5 * float(np.trace(X.T @ A @ X))
    #
    # def grad_f(X: Array) -> Array:
    #     return A @ X
    #
    # def hess_f(X: Array, V: Array) -> Array:
    #     # Hessian action for quadratic objective is constant
    #     return A @ V
    #
    # solver = PenCS(
    #     beta=200.0,
    #     K=np.sqrt(p) + 1.0,
    #     eta0=1.0,
    #     tol=1e-10,
    #     # ortho_tol=1e-8,
    #     max_iter=100,
    #     backtracking=True,
    #     verbosity=2,
    # )
    # res = solver.run(n=n, p=p, grad_f=grad_f, hess_f=hess_f, cost_f=cost_f, seed=42)
    # print("Stopping:", res.stopping_reason, "iters:", res.iterations)
    # print("final tangent_residual:", res.log["tangent_residual"][-1])
    # print("final ortho_error:", res.log["ortho_error"][-1])
    # Set parameters
    n = 1000
    p = 20
    alpha = 1

    L = diags(np.array([-1, 2, -1]), np.array([1, 0, -1]), shape=(n, n)).tocsc()

    solve_L = factorized(L)

    def obj_fun(X):
        LX = L @ X
        rho = np.sum(X * X, 1)
        Lrho = spsolve(L, rho)
        fval = 0.5 * np.sum(X * LX) + (alpha / 4) * np.sum(rho * Lrho)
        return fval

    def obj_grad(X):
        LX = L @ X
        rho = np.sum(X * X, 1)
        Lrho = spsolve(L, rho)
        grad = LX + alpha * Lrho[:, np.newaxis] * X
        return grad

    def obj_hvp_fast(X, V, L, alpha, solve_L):
        rho  = np.sum(X * X, axis=1)
        u    = solve_L(rho)
        drho = 2.0 * np.sum(X * V, axis=1)
        du   = solve_L(drho)
        return (L @ V) + alpha * (du[:, None] * X + u[:, None] * V)

    hess_f = lambda X, V: obj_hvp_fast(X, V, L=L, alpha=alpha, solve_L=solve_L)



    penCS = PenCS(
        beta=200,
        K=np.sqrt(p) + 1.0,
        eta0=1.0,
        tol=1e-10,
        max_iter=100,
        reg_mu=1e-8,
        theta=1.0,
        zeta_max=1e-2,
        linear_maxiter= 200,
        backtracking=False,
        verbosity=2,
    )
    penCS_res = penCS.run(n=n, p=p, cost_f=obj_fun, grad_f=obj_grad, hess_f=hess_f, X0=X0)


if __name__ == "__main__":
    _example_quadratic()
