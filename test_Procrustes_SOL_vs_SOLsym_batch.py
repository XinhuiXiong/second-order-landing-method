#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orthogonal Procrustes experiment: SOL vs SOL-sym.

Supports:
  1) Single run with --n, --d, --noise
  2) Batch runs over parameter grids with --n-list, --d-list, --noise-list
  3) Automatic output directories grouped by parameter values
  4) Per-run summary.json and top-level batch_summary.{json,csv}

Examples
--------
Single run
    python test_Procrustes_SOL_vs_SOLsym_batch.py --n 5000 --d 200 --noise 0.02

Single run with explicit output directory
    python test_Procrustes_SOL_vs_SOLsym_batch.py \
        --n 5000 --d 200 --noise 0.02 \
        --out results/procrustes/custom_run

Batch run
    python test_Procrustes_SOL_vs_SOLsym_batch.py \
        --n-list 2000 5000 \
        --d-list 100 200 500 \
        --noise-list 0.01 0.02 0.05 \
        --out-root results/procrustes

Run in background
    nohup python -u test_Procrustes_SOL_vs_SOLsym_batch.py \
        --n-list 2000 5000 \
        --d-list 100 200 \
        --noise-list 0.01 0.05 \
        --out-root results/procrustes \
        > procrustes_batch.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pymanopt
from pymanopt import function
from pymanopt.manifolds import Stiefel

from second_order_landing import (
    FirstOrderLanding,
    SecondOrderLanding,
    SecondOrderLandingSymmetric,
)

Array = np.ndarray



def procrustes_svd(X: Array, Y: Array) -> Array:
    """Closed-form R_* = argmin_{R∈O(d)} ||X R - Y||_F^2."""
    U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    return U @ Vt


def cost_fn(X: Array, Y: Array, R: Array) -> float:
    E = X @ R - Y
    return float(0.5 * np.mean(np.sum(E * E, axis=1)))


def grad_fn(X: Array, Y: Array, R: Array) -> Array:
    n = float(X.shape[0])
    return (X.T @ (X @ R - Y)) / n


def hess_fn(X: Array, R: Array, V: Array) -> Array:
    _ = R
    n = float(X.shape[0])
    return (X.T @ (X @ V)) / n


def ortho_error(R: Array) -> float:
    d = R.shape[0]
    return float(np.linalg.norm(R.T @ R - np.eye(d), ord="fro"))


def dist_to_optimum(R: Array, R_star: Array) -> float:
    return float(np.linalg.norm(R - R_star, ord="fro"))


def noise_to_tag(noise: float) -> str:
    """Stable string for directory/file naming."""
    return format(float(noise), ".12g").replace("-", "m").replace(".", "p")


def build_output_dir(base_root: str | Path, n_samples: int, d: int, noise: float) -> Path:
    base = Path(base_root)
    return base / f"n_{n_samples}" / f"d_{d}" / f"noise_{noise_to_tag(noise)}"


def to_jsonable(obj: Any) -> Any:
    """Convert nested objects to JSON-serializable data."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def save_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def final_metric_or_nan(arr: np.ndarray) -> float:
    return float(arr[-1]) if arr.size > 0 else float("nan")


# -------------------------
# Main experiment
# -------------------------
def generate_plots(
    output_dir: str | Path,
    *,
    n_samples: int = 5000,
    d: int = 200,
    noise: float = 0.02,
    seed: int = 0,
    run_AltSOL: bool = True,
) -> dict[str, object]:
    """Run one Procrustes experiment and save SOL vs SOL-sym figures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Build synthetic Procrustes data
    # -------------------------
    rng = np.random.default_rng(seed)

    X = rng.standard_normal((n_samples, d)).astype(np.float64)

    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    R_true = Q

    Y = (X @ R_true + noise * rng.standard_normal((n_samples, d))).astype(np.float64)

    manifold = Stiefel(d, d)

    def cost(R: Array) -> float:
        return cost_fn(X, Y, R)

    def grad_f(R: Array) -> Array:
        return grad_fn(X, Y, R)

    def hess_f(R: Array, V: Array) -> Array:
        return hess_fn(X, R, V)

    # Keep pymanopt decorators in case you want to reuse later
    @function.numpy(manifold)
    def cost_pm(R: Array) -> float:
        return cost(R)

    @function.numpy(manifold)
    def egrad_pm(R: Array) -> Array:
        return grad_f(R)

    @function.numpy(manifold)
    def ehess_pm(R: Array, V: Array) -> Array:
        return hess_f(R, V)

    _ = cost_pm, egrad_pm, ehess_pm, pymanopt

    R_star = procrustes_svd(X, Y)
    f_star = float(cost(R_star))

    print("\n================== Procrustes SOL vs SOL-sym summary =================")
    print(f"output_dir: {out}")
    print(f"seed:      {seed}")
    print(f"n_samples: {n_samples}")
    print(f"d:         {d}")
    print(f"noise:     {noise}")
    print(f"f*:        {f_star:.6e}")
    print(f"det(R*):   {float(np.linalg.det(R_star)):+.6f}")
    print("=====================================================================\n")

    # -------------------------
    # Per-iteration distance trackers
    # -------------------------
    sol_dist_hist: List[float] = []
    sym_dist_hist: List[float] = []

    # Random feasible init
    R_init, _ = np.linalg.qr(rng.standard_normal((d, d)))

    # -------------------------
    # Stage 0: warm start
    # -------------------------
    warm = FirstOrderLanding(
        epsilon=0.75,
        lam=5,
        eta=0.1,
        tol=0.01,
        max_iter=10000,
        verbosity=2,
    )
    warm_res = warm.run(n=d, p=d, grad_f=grad_f, cost=cost, X0=R_init)
    R0 = 1.01 * warm_res.X
    # R0 = warm_res.X 

    # -------------------------
    # Stage 1: short FOL
    # -------------------------
    fol = FirstOrderLanding(
        epsilon=0.75,
        lam=1,
        eta=0.1,
        tol=1e-13,
        max_iter=200,      
        verbosity=2,
    )
    fol_res = fol.run(n=d, p=d, grad_f=grad_f, cost=cost, X0=R0)

    # -------------------------
    # Stage 2: SOL
    # -------------------------
    sol = SecondOrderLanding(
        epsilon=0.75,
        eta=1.0,
        tol=1e-13,
        max_iter=5,
        linear_rtol=None,
        theta=1.0,
        zeta_max=1e-1,
        linear_solver="bicgstab",
        linear_solver_options={"atol": 1e-14},
        linear_maxiter=1000,
        verbosity=2,
    )

    def _sol_cb(R: Array, k: int) -> bool:
        _ = k
        sol_dist_hist.append(dist_to_optimum(R, R_star))
        return False

    sol_res = sol.run(
        n=d,
        p=d,
        cost=cost,
        grad_f=grad_f,
        hess_f=hess_f,
        NS_order=1, 
        X0=R0,
        callback=_sol_cb,
    )

    # -------------------------
    # Stage 3: SOL-sym
    # -------------------------
    sol_sym = SecondOrderLandingSymmetric(
        epsilon=0.75,
        eta=1.0,
        tol=1e-12,
        max_iter=10,
        theta=1.0,
        zeta_max=1e-1,
        linear_maxiter=1000,
        fallback_to_minres=True,
        verbosity=2,
    )

    def _sym_cb(R: Array, k: int) -> bool:
        _ = k
        sym_dist_hist.append(dist_to_optimum(R, R_star))
        return False

    sol_sym_res = sol_sym.run(
        n=d,
        p=d,
        cost=cost,
        grad_f=grad_f,
        hess_f=hess_f,
        X0=R0,
        callback=_sym_cb,
    )




    # -------------------------
    # Unify logs/series
    # -------------------------
    def abs_gap(cost_list: List[float]) -> np.ndarray:
        arr = np.asarray(cost_list, dtype=float)
        return np.abs(arr - f_star)

    sol_log = sol_res.log
    sol_t = np.asarray(sol_log["time"], dtype=float)
    sol_it = np.arange(0, len(sol_t))
    sol_gap = abs_gap([float(x) for x in sol_log["cost"]])
    sol_ortho = np.asarray(sol_log["ortho_error"], dtype=float)
    sol_t1 = np.asarray(sol_log["tangent_residual"], dtype=float)
    sol_dist = np.asarray(sol_dist_hist[: len(sol_t)], dtype=float)

    sym_log = sol_sym_res.log
    sym_t = np.asarray(sym_log["time"], dtype=float)
    sym_it = np.arange(0, len(sym_t))
    sym_gap = abs_gap([float(x) for x in sym_log["cost"]])
    sym_ortho = np.asarray(sym_log["ortho_error"], dtype=float)
    sym_t1 = np.asarray(sym_log["tangent_residual"], dtype=float)
    sym_dist = np.asarray(sym_dist_hist[: len(sym_t)], dtype=float)

    fol_log = fol_res.log
    fol_t = np.asarray(fol_log["time"], dtype=float)
    fol_it = np.arange(0, len(fol_t))
    fol_gap = abs_gap([float(x) for x in fol_log["cost"]])
    fol_ortho = np.asarray(fol_log["ortho_error"], dtype=float)
    fol_t1 = np.asarray(fol_log["tangent_residual"], dtype=float)


    # -------------------------
    # Plot styling
    # -------------------------
    SIAM_BASE_FONTSIZE = 10
    SIAM_COL_WIDTH_IN = 3.25
    fig_w_in, fig_h_in = SIAM_COL_WIDTH_IN, 2.25

    SIAM_RCPARAMS = {
        "font.size": SIAM_BASE_FONTSIZE,
        "axes.titlesize": SIAM_BASE_FONTSIZE,
        "axes.labelsize": SIAM_BASE_FONTSIZE,
        "xtick.labelsize": SIAM_BASE_FONTSIZE - 1,
        "ytick.labelsize": SIAM_BASE_FONTSIZE - 1,
        "legend.fontsize": SIAM_BASE_FONTSIZE - 1,
        "figure.titlesize": SIAM_BASE_FONTSIZE,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    def plot_curves(
        curves: list[tuple[np.ndarray, np.ndarray, str, str]],
        *,
        xlabel: str,
        ylabel: str,
        title: str = None,
        filename: str,
        logy: bool = False,
        xmax: float | None = None,
        legend_loc="best",
    ) -> None:

        STYLE_MAP: dict[str, dict[str, str]] = {
            "SOL": {"marker": "s", "color": "tab:blue"},
            "SOL-sym": {"marker": "^", "color": "tab:orange"},
            "Landing": {"marker": "o", "color": "tab:brown"},
        }

        with plt.rc_context(SIAM_RCPARAMS):
            fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
            for x, y, label, marker in curves:
                style = STYLE_MAP.get(label, {"marker": "o", "color": None})
                ax.plot(x, y, label=label, marker=style["marker"], color=style["color"], linewidth=1.5, markersize=4)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if title is not None:
                plt.title(title)
            if logy:
                ax.set_yscale("log")
            if xmax is not None:
                ax.set_xlim(left=0, right=float(xmax))
            ax.legend(loc=legend_loc, frameon=False)
            ax.tick_params(axis="both", which="both")
            fig.tight_layout(pad=0.15)
            fig.savefig(out / filename, dpi=300, bbox_inches="tight", pad_inches=0.01)
            if Path(filename).suffix.lower() != ".pdf":
                fig.savefig(out / (Path(filename).stem + ".pdf"), bbox_inches="tight", pad_inches=0.01)
            plt.close(fig)


    def last_or_zero(arr: np.ndarray) -> float:
        return float(arr[-1]) if arr.size > 0 else 0.0

    xmax_iter_ref = max(last_or_zero(sol_it), last_or_zero(sym_it))
    xmax_time_ref = max(last_or_zero(sol_t), last_or_zero(sym_t))
    # -------------------------
    # Plots
    # -------------------------
    plot_curves(
        [
            (sol_it, sol_gap, "SOL", "s"),
            (sym_it, sym_gap, "SOL-sym", "^"),
            (fol_it, fol_gap, "Landing", "o")
          ],
        xlabel="Iteration",
        ylabel="Objective gap",
        filename="Procrustes_gap_vs_iter.png",
        logy=True,
        xmax=xmax_iter_ref * 1.1,
    )

    plot_curves(
        [
            (sol_t, sol_gap, "SOL", "s"),
            (sym_t, sym_gap, "SOL-sym", "^"),
            (fol_t, fol_gap, "Landing", "o")
        ],
        xlabel="Time (s)",
        ylabel="Objective gap",
        filename="Procrustes_gap_vs_time.png",
        logy=True,
        xmax=xmax_time_ref * 1.1,
    )

    plot_curves(
        [
            (sol_it, sol_ortho, "SOL", "s"), 
            (sym_it, sym_ortho, "SOL-sym", "^"),
            (fol_it, fol_ortho, "Landing", "o"),
        ],
        xlabel="Iteration",
        ylabel=r"$\|X^\top X -I_p\|_{\mathrm{F}}$",
        filename="Procrustes_ortho_vs_iter.png",
        logy=True,
        legend_loc="right",
        xmax=xmax_iter_ref * 1.1,
    )

    plot_curves(
        [
            (sol_t, sol_ortho, "SOL", "s"), 
            (sym_t, sym_ortho, "SOL-sym", "^"),
            (fol_t, fol_ortho, "Landing", "o"),
        ],
        xlabel="Time (s)",
        ylabel=r"$\|X^\top X -I_p\|_{\mathrm{F}}$",
        filename="Procrustes_ortho_vs_time.png",
        logy=True,
        xmax=xmax_time_ref * 1.1,
        legend_loc="right"
    )

    plot_curves(
        [
            (sol_it, sol_t1, "SOL", "s"),
            (sym_it, sym_t1, "SOL-sym", "^"),
            (fol_it, fol_t1, "Landing", "o"),
        ],
        xlabel="Iteration",
        ylabel=r"$\|\operatorname{grad} f(X)\|_{\mathrm{F}}$",
        filename="Procrustes_tangent_vs_iter.png",
        logy=True,
        xmax=xmax_iter_ref * 1.1,
    )

    plot_curves(
        [
        (sol_t, sol_t1, "SOL", "s"), 
        (sym_t, sym_t1, "SOL-sym", "^"),
        (fol_t, fol_t1, "Landing", "o"),
        ],
        xlabel="Time (s)",
        ylabel=r"$\|\operatorname{grad} f(X)\|_{\mathrm{F}}$",
        filename="Procrustes_tangent_vs_time.png",
        logy=True,
        xmax=xmax_time_ref * 1.1,
    )

    plot_curves(
        [
            (sol_it[: len(sol_dist)], sol_dist, "SOL", "s"),
            (sym_it[: len(sym_dist)], sym_dist, "SOL-sym", "^"),
        ],
        xlabel="Iteration",
        ylabel="Distance to optimum",
        filename="Procrustes_dist_vs_iter.png",
        logy=True,
        xmax=xmax_iter_ref * 1.1,
    )

    plot_curves(
        [
            (sol_t[: len(sol_dist)], sol_dist, "SOL", "s"),
            (sym_t[: len(sym_dist)], sym_dist, "SOL-sym", "^"),
        ],
        xlabel="Time (s)",
        ylabel="Distance to optimum",
        filename="Procrustes_dist_vs_time.png",
        logy=True,
        xmax=xmax_time_ref * 1.1,
    )

    # -------------------------
    # Final summary
    # -------------------------


    results: dict[str, object] = {
        "f_star": f_star,
        "R_star": R_star,
        "warm": warm_res,
        "fol": fol_res,
        "sol": sol_res,
        "sol_sym": sol_sym_res,
        "plots_dir": str(out),
        "sol_dist_hist": sol_dist_hist,
        "sol_sym_dist_hist": sym_dist_hist,
    }
    return results


def expand_grid(
    n_values: Sequence[int],
    d_values: Sequence[int],
    noise_values: Sequence[float],
) -> list[tuple[int, int, float]]:
    return [(int(n), int(d), float(noise)) for n, d, noise in product(n_values, d_values, noise_values)]


def run_batch(
    *,
    out_root: str | Path,
    n_values: Sequence[int],
    d_values: Sequence[int],
    noise_values: Sequence[float],
    seed: int = 0,
    continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    grid = expand_grid(n_values, d_values, noise_values)
    batch_rows: list[dict[str, Any]] = []

    print("\n================== Batch configuration ==================")
    print(f"out_root:   {out_root}")
    print(f"seed:       {seed}")
    print(f"n_values:   {list(map(int, n_values))}")
    print(f"d_values:   {list(map(int, d_values))}")
    print(f"noise_vals: {[float(v) for v in noise_values]}")
    print(f"num_runs:   {len(grid)}")
    print("=========================================================\n")

    for idx, (n_samples, d, noise) in enumerate(grid, start=1):
        run_dir = build_output_dir(out_root, n_samples, d, noise)
        print(f"[{idx}/{len(grid)}] Running n={n_samples}, d={d}, noise={noise} -> {run_dir}")

        row: dict[str, Any] = {
            "status": "pending",
            "n_samples": int(n_samples),
            "d": int(d),
            "noise": float(noise),
            "seed": int(seed),
            "output_dir": str(run_dir),
        }

        try:
            results = generate_plots(
                output_dir=run_dir,
                n_samples=n_samples,
                d=d,
                noise=noise,
                seed=seed,
            )
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            save_json(Path(run_dir) / "summary.json", row)
            if not continue_on_error:
                batch_rows.append(row)
                raise

        batch_rows.append(row)

    save_json(out_root / "batch_summary.json", batch_rows)
    save_csv(out_root / "batch_summary.csv", batch_rows)
    return batch_rows


def run_without_plot() -> dict[str, object]:
    return generate_plots(
        output_dir="procrustes_plots_sol_vs_sym",
        n_samples=800,
        d=80,
        noise=0.02,
        seed=0,
    )


def test_procrustes_pipeline_runs() -> None:
    results = run_without_plot()
    sol = results["sol"]
    sym = results["sol_sym"]

    sol_log = sol.log
    sym_log = sym.log

    assert len(sol_log["cost"]) > 0
    assert len(sym_log["cost"]) > 0
    assert float(sol_log["ortho_error"][-1]) < 1e-3
    assert float(sym_log["ortho_error"][-1]) < 1e-3
    assert Path(results["plots_dir"]).exists()
    assert Path(results["plots_dir"]).joinpath("summary.json").exists()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Orthogonal Procrustes experiment: SOL vs SOL-sym.")

    # Single-run args
    ap.add_argument("--out", type=str, default=None, help="Explicit output directory for a single run.")
    ap.add_argument("--n", type=int, default=5000, help="Number of samples (rows of X/Y) for a single run.")
    ap.add_argument("--d", type=int, default=500, help="Dimension (orthogonal matrix is d x d) for a single run.")
    ap.add_argument("--noise", type=float, default=0.05, help="Noise std in Y = X R_true + noise for a single run.")

    # Batch/grid args
    ap.add_argument("--out-root", type=str, default="results/procrustes", help="Root output directory for auto-grouped runs.")
    ap.add_argument("--n-list", type=int, nargs="*", default=None, help="Batch values for n.")
    ap.add_argument("--d-list", type=int, nargs="*", default=None, help="Batch values for d.")
    ap.add_argument("--noise-list", type=float, nargs="*", default=None, help="Batch values for noise.")

    # Misc
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="In batch mode, continue remaining runs even if one configuration fails.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    batch_requested = any(
        values is not None and len(values) > 0
        for values in (args.n_list, args.d_list, args.noise_list)
    )

    if batch_requested:
        n_values = args.n_list if args.n_list else [args.n]
        d_values = args.d_list if args.d_list else [args.d]
        noise_values = args.noise_list if args.noise_list else [args.noise]

        run_batch(
            out_root=args.out_root,
            n_values=n_values,
            d_values=d_values,
            noise_values=noise_values,
            seed=args.seed,
            continue_on_error=args.continue_on_error,
        )
        return

    if args.out is not None:
        out_dir = Path(args.out)
    else:
        out_dir = build_output_dir(args.out_root, args.n, args.d, args.noise)

    generate_plots(
        output_dir=out_dir,
        n_samples=args.n,
        d=args.d,
        noise=args.noise,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
