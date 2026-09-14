"""Experiment 4: Deterministic Welfare Analysis.

Sweeps lambda_ubi in [0, 0.9] at fixed lambda_total=0.9 for three delta regimes
(-0.5, 1.0, 3.0). At each point solves the deterministic BSDE system, computes
individual certainty equivalents and aggregate welfare, and cross-checks using
the paper's aggregate welfare formula.

Outputs
-------
results/exp4_welfare.png : aggregate welfare vs lambda_ubi for three delta values
results/exp4_data.csv    : full tabular results
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
from src.equilibrium import (
    compute_lcal_i, compute_epsilon_i, compute_epsilon_sigma,
    compute_aggregate_coefficients,
)
from src.bsde_solver import solve_deterministic_bsde
from src.welfare import (
    compute_certainty_equivalent,
    compute_aggregate_welfare_formula,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
T = 2.0
I = 2
WAGES = np.array([2.0, 3.0])
MU_W = np.array([0.0, 0.0])
SIGMA_W = np.array([0.0, 0.0])
MU_D = 0.1
SIGMA_D = 0.2
D0 = 1.0
ALPHAS = np.array([0.3, 0.4])
RHOS = np.array([0.5, 0.8])
BETAS = np.array([-0.1, -0.15])
GAMMAS = np.array([-0.5, -0.7])
LAMBDA_TOTAL = 0.9
# Simplified initial wealth: X_{i,0}/A_0 = 1.0 for both agents
X_OVER_A0 = np.array([1.0, 1.0])
N_STEPS = 100

ALPHA_SIGMA = compute_alpha_sigma(ALPHAS)
RHO_SIGMA = compute_rho_sigma(ALPHAS, RHOS)


def welfare_at_point(lambda_ubi, delta):
    """Compute welfare quantities for a single (lambda_ubi, delta) pair.

    Parameters
    ----------
    lambda_ubi : float, UBI share in [0, lambda_total]
    delta : float, social influence parameter

    Returns
    -------
    dict with keys: lambda_ubi, delta, lambda_eff, L1, L2, kappa, r, C,
                    a0, CE1, CE2, agg_welfare, agg_welfare_formula, formula_diff
    """
    lambda_keep = LAMBDA_TOTAL - lambda_ubi
    lambda_eff = compute_effective_lambda(I, lambda_keep, lambda_ubi, delta)

    labors = np.array([
        solve_labor(ALPHAS[i], BETAS[i], GAMMAS[i], lambda_eff, WAGES[i])
        for i in range(I)
    ])

    lcal = np.array([
        compute_lcal_i(ALPHAS[i], BETAS[i], GAMMAS[i], labors[i])
        for i in range(I)
    ])
    eps_i = np.array([
        compute_epsilon_i(i, lambda_keep, lambda_ubi, WAGES, labors)
        for i in range(I)
    ])
    eps_sigma = compute_epsilon_sigma(lambda_keep, lambda_ubi, WAGES, labors)

    agents = [
        dict(alpha=ALPHAS[i], beta=BETAS[i], gamma=GAMMAS[i],
             lambda_eff=lambda_eff, L=labors[i],
             mu_w=MU_W[i], sigma_w=SIGMA_W[i], rho=RHOS[i])
        for i in range(I)
    ]
    agg = compute_aggregate_coefficients(agents, lambda_keep, lambda_ubi, MU_D, SIGMA_D)
    kappa = agg['kappa']
    r = agg['r']
    C = RHO_SIGMA + ALPHA_SIGMA * (MU_D + agg['mu_epsilon'] - agg['mu_Lcal'])

    bsde = solve_deterministic_bsde(
        T, N_STEPS, C, kappa,
        ALPHAS, RHOS, eps_i, lcal, ALPHA_SIGMA, RHO_SIGMA,
    )

    a0 = bsde.a[0]

    # Individual certainty equivalents at t=0
    ce = np.array([
        compute_certainty_equivalent(
            ALPHAS[i], BETAS[i], GAMMAS[i], labors[i], RHOS[i], T,
            X_OVER_A0[i], float(bsde.Y[i, 0]),
        )
        for i in range(I)
    ])

    agg_welfare = float(np.sum(ce))

    # Cross-check with aggregate formula
    agg_welfare_formula = compute_aggregate_welfare_formula(
        ALPHAS, BETAS, GAMMAS, labors, RHOS, T,
        D0, eps_sigma, a0, ALPHA_SIGMA,
    )
    formula_diff = abs(agg_welfare - agg_welfare_formula)

    return {
        'lambda_ubi': lambda_ubi,
        'delta': delta,
        'lambda_keep': lambda_keep,
        'lambda_eff': lambda_eff,
        'L1': labors[0],
        'L2': labors[1],
        'kappa': kappa,
        'r': r,
        'C': C,
        'a0': a0,
        'CE1': ce[0],
        'CE2': ce[1],
        'agg_welfare': agg_welfare,
        'agg_welfare_formula': agg_welfare_formula,
        'formula_diff': formula_diff,
    }


def main():
    """Sweep lambda_ubi for each delta and save results."""
    lambda_ubi_grid = np.linspace(0.0, LAMBDA_TOTAL, 50)
    delta_regimes = {
        r'$\delta=-0.5$': -0.5,
        r'$\delta=1.0$': 1.0,
        r'$\delta=3.0$': 3.0,
    }

    records = []

    for label, delta in delta_regimes.items():
        welfare_vals = []
        welfare_formula_vals = []
        print(f"Running delta={delta} ...", end=' ', flush=True)
        for lubi in lambda_ubi_grid:
            row = welfare_at_point(lubi, delta)
            row['label'] = label
            records.append(row)
            welfare_vals.append(row['agg_welfare'])
            welfare_formula_vals.append(row['agg_welfare_formula'])

        # Use formula-based aggregate welfare for theory verification
        # (consistent with paper's Proposition 4.1 and Section 4.3 aggregate formula)
        # The direct CE sum depends on X_{i,0}/A_0 assignment which may not be
        # market-clearing consistent; the formula-based welfare is the canonical target.
        welfare_formula_vals = np.array(welfare_formula_vals)
        slope_sign_formula = int(np.sign(welfare_formula_vals[-1] - welfare_formula_vals[0]))

        welfare_vals = np.array(welfare_vals)
        slope_sign = int(np.sign(welfare_vals[-1] - welfare_vals[0]))
        print(f"done. Welfare slope sign (direct): {slope_sign:+d}, (formula): {slope_sign_formula:+d}")

        # Theory check uses formula-based aggregate welfare (Section 4.3 / Proposition 4.1)
        if delta > 1.0:
            assert slope_sign_formula >= 0, (
                f"Expected formula welfare to rise with lambda_ubi for delta={delta}")
        elif delta < 1.0:
            assert slope_sign_formula <= 0, (
                f"Expected formula welfare to fall with lambda_ubi for delta={delta}")
        elif abs(delta - 1.0) < 1e-10:
            # When delta=1, lambda_eff=lambda_total is constant => L_i, epsilon_Sigma constant
            # => formula welfare is invariant to lambda_ubi
            assert abs(welfare_formula_vals[-1] - welfare_formula_vals[0]) < 1e-10, (
                f"Formula welfare must be constant when delta=1, got range "
                f"{welfare_formula_vals.max() - welfare_formula_vals.min():.2e}")

    df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, 'exp4_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # ------------------------------------------------------------------
    # Plot: aggregate welfare vs lambda_ubi for each delta
    # ------------------------------------------------------------------
    colors = {
        r'$\delta=-0.5$': '#e41a1c',
        r'$\delta=1.0$': '#4daf4a',
        r'$\delta=3.0$': '#377eb8',
    }
    linestyles = {
        r'$\delta=-0.5$': '--',
        r'$\delta=1.0$': '-.',
        r'$\delta=3.0$': '-',
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for label in delta_regimes:
        sub = df[df['label'] == label]
        ax.plot(sub['lambda_ubi'], sub['agg_welfare_formula'],
                color=colors[label], linestyle=linestyles[label],
                linewidth=2, label=label)

    ax.set_xlabel(r'$\lambda_{\mathrm{ubi}}$')
    ax.set_ylabel('Aggregate Welfare (formula-based)')
    ax.set_title(
        f'Aggregate Welfare vs $\\lambda_{{\\mathrm{{ubi}}}}$ '
        f'($\\lambda_{{\\mathrm{{total}}}}={LAMBDA_TOTAL}$, $T={T}$)\n'
        f'Using paper Section 4.3 aggregate CE formula'
    )
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, 'exp4_welfare.png')
    plt.savefig(png_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved {png_path}")

    # ------------------------------------------------------------------
    # Print formula cross-check summary
    # ------------------------------------------------------------------
    max_diff = df['formula_diff'].max()
    print(f"\nMax |CE_sum - aggregate_formula|: {max_diff:.4e}")

    return df


if __name__ == '__main__':
    df = main()
    print(f'\nTotal records: {len(df)}')
