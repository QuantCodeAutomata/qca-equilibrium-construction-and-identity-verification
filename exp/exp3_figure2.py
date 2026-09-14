"""Experiment 3: Figure 2 Reproduction.

Replicates Figure 2 of the paper using the exact calibration:
  - lambda_keep=0.7, lambda_ubi=0.2, I=2
  - alpha_i=0.2, rho_i=1 (identical agents for risk/discounting)
  - Heterogeneous utility: beta_1=-0.02, beta_2=-0.01, gamma_1=-0.8, gamma_2=-1
  - Wage SDE coefficients: mu_w=[0.1, 0.2], sigma_w=[0.1, -0.05]
  - delta in {-2, 0, 4, 8}
  - Wage grid: w in [1, 8] with 300 points, w_1=w_2=w

For each (delta, w), computes equilibrium labor L_i, the market price of risk
kappa, and the interest-rate coefficient r, then plots kappa(w) and r(w).

Outputs
-------
results/exp3_figure2.png : kappa(w) and r(w) for all four delta values
results/exp3_data.csv    : full tabular results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.labor_solver import (
    solve_labor, compute_effective_lambda,
    compute_alpha_sigma, compute_rho_sigma,
)
from src.equilibrium import compute_aggregate_coefficients

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 2 calibration (exact)
# ---------------------------------------------------------------------------
LAMBDA_KEEP = 0.7
LAMBDA_UBI = 0.2
I = 2
ALPHAS = np.array([0.2, 0.2])
RHOS = np.array([1.0, 1.0])
BETAS = np.array([-0.02, -0.01])
GAMMAS = np.array([-0.8, -1.0])
MU_W = np.array([0.1, 0.2])
SIGMA_W = np.array([0.1, -0.05])
MU_D = 0.1
SIGMA_D = 0.2
DELTA_VALUES = [-2, 0, 4, 8]
W_GRID = np.linspace(1.0, 8.0, 300)

ALPHA_SIGMA = compute_alpha_sigma(ALPHAS)   # 0.1
RHO_SIGMA = compute_rho_sigma(ALPHAS, RHOS) # 1.0


def compute_kappa_r_at_w(w, delta):
    """Compute equilibrium kappa and r at a given common wage w and delta.

    Parameters
    ----------
    w : float, common wage level (w_1 = w_2 = w)
    delta : float, social influence parameter

    Returns
    -------
    tuple (kappa, r, L1, L2)
    """
    wages = np.array([w, w])
    lambda_eff = compute_effective_lambda(I, LAMBDA_KEEP, LAMBDA_UBI, delta)

    labors = np.array([
        solve_labor(ALPHAS[i], BETAS[i], GAMMAS[i], lambda_eff, wages[i])
        for i in range(I)
    ])

    agents = [
        dict(alpha=ALPHAS[i], beta=BETAS[i], gamma=GAMMAS[i],
             lambda_eff=lambda_eff, L=labors[i],
             mu_w=MU_W[i], sigma_w=SIGMA_W[i], rho=RHOS[i])
        for i in range(I)
    ]
    agg = compute_aggregate_coefficients(agents, LAMBDA_KEEP, LAMBDA_UBI, MU_D, SIGMA_D)
    return agg['kappa'], agg['r'], labors[0], labors[1]


def count_sign_changes(arr):
    """Count number of sign changes in np.diff(arr), ignoring near-zero differences."""
    d = np.diff(arr)
    d_nz = d[np.abs(d) > 1e-10]
    if len(d_nz) < 2:
        return 0
    return int(np.sum(np.diff(np.sign(d_nz)) != 0))


def main():
    """Compute kappa(w) and r(w) grids, check non-monotonicity, plot and save."""
    records = []
    # kappa_grid[delta_idx, w_idx], r_grid[delta_idx, w_idx]
    kappa_grid = np.zeros((len(DELTA_VALUES), len(W_GRID)))
    r_grid = np.zeros((len(DELTA_VALUES), len(W_GRID)))

    for di, delta in enumerate(DELTA_VALUES):
        lambda_eff = compute_effective_lambda(I, LAMBDA_KEEP, LAMBDA_UBI, delta)
        print(f"delta={delta:+d}: lambda_eff={lambda_eff:.4f}", end='')

        for wi, w in enumerate(W_GRID):
            kappa, r, L1, L2 = compute_kappa_r_at_w(w, delta)
            kappa_grid[di, wi] = kappa
            r_grid[di, wi] = r
            records.append({
                'delta': delta,
                'w': w,
                'lambda_eff': lambda_eff,
                'L1': L1,
                'L2': L2,
                'kappa': kappa,
                'r': r,
            })

        n_kappa_changes = count_sign_changes(kappa_grid[di])
        n_r_changes = count_sign_changes(r_grid[di])
        print(f"  |  kappa sign-changes: {n_kappa_changes}, "
              f"r sign-changes: {n_r_changes}")

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, 'exp3_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # ------------------------------------------------------------------
    # Plot: kappa(w) and r(w) for all four delta values
    # ------------------------------------------------------------------
    colors = {-2: '#e41a1c', 0: '#ff7f00', 4: '#377eb8', 8: '#4daf4a'}
    linestyles = {-2: '--', 0: ':', 4: '-.', 8: '-'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for di, delta in enumerate(DELTA_VALUES):
        lbl = f'$\\delta={delta}$'
        c = colors[delta]
        ls = linestyles[delta]
        axes[0].plot(W_GRID, kappa_grid[di], color=c, linestyle=ls,
                     linewidth=2, label=lbl)
        axes[1].plot(W_GRID, r_grid[di], color=c, linestyle=ls,
                     linewidth=2, label=lbl)

    axes[0].axhline(0, color='k', linewidth=0.8, linestyle='-')
    axes[0].set_xlabel('Wage $w$')
    axes[0].set_ylabel(r'$\kappa(w)$')
    axes[0].set_title('Market Price of Risk $\\kappa(w)$ — Figure 2')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].axhline(0, color='k', linewidth=0.8, linestyle='-')
    axes[1].set_xlabel('Wage $w$')
    axes[1].set_ylabel('$r(w)$')
    axes[1].set_title('Interest-Rate Coefficient $r(w)$ — Figure 2')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, 'exp3_figure2.png')
    plt.savefig(png_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved {png_path}")

    # ------------------------------------------------------------------
    # Non-monotonicity report
    # ------------------------------------------------------------------
    print('\nNon-monotonicity in kappa and r (sign changes in finite differences):')
    for di, delta in enumerate(DELTA_VALUES):
        nk = count_sign_changes(kappa_grid[di])
        nr = count_sign_changes(r_grid[di])
        print(f"  delta={delta:+d}: kappa changes={nk}, r changes={nr}")

    return df


if __name__ == '__main__':
    df = main()
    print(f'\nTotal records: {len(df)}')
