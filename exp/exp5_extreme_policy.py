"""Experiment 5: Extreme Policy Comparison.

Compares two polar fiscal regimes across a grid of delta values:
  - Pure Capitalism (PC): lambda_keep=1, lambda_ubi=0
    → lambda_eff_PC = 1 (independent of delta)
  - Social Market Economy (SM): lambda_keep=0, lambda_ubi=1
    → lambda_eff_SM = 1/I + delta*(I-1)/I

For I=2: lambda_eff_SM = 0.5 + 0.5*delta.  SM dominates PC in aggregate
labor income iff delta > 1 (equivalently, lambda_eff_SM > 1).

Two wage configurations are tested:
  - Heterogeneous: w = [2.0, 3.0]
  - Homogeneous:   w = [2.5, 2.5]

Outputs
-------
results/exp5_extreme_policy.png : income difference (SM − PC) vs delta for both configs
results/exp5_data.csv           : full tabular results from extreme_policy_comparison
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.comparative_statics import extreme_policy_comparison
from src.labor_solver import compute_effective_lambda

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Fixed agent parameters
# ---------------------------------------------------------------------------
I = 2
ALPHAS = np.array([0.3, 0.4])
BETAS = np.array([-0.1, -0.15])
GAMMAS = np.array([-0.5, -0.7])
DELTA_GRID = np.linspace(-3.0, 5.0, 81)

WAGE_CONFIGS = {
    'Heterogeneous (w=[2.0, 3.0])': np.array([2.0, 3.0]),
    'Homogeneous (w=[2.5, 2.5])': np.array([2.5, 2.5]),
}


def analytical_crossover(I=2):
    """Return the delta at which lambda_eff_SM = lambda_eff_PC = 1.

    lambda_eff_SM = 1/I + delta*(I-1)/I = 1  =>  delta = 1.
    """
    return 1.0


def verify_lambda_eff_sm_crossover(delta_grid, I=2):
    """Verify analytically that lambda_eff_SM > 1 iff delta > 1 for each point."""
    for delta in delta_grid:
        leff_sm = compute_effective_lambda(I, 0.0, 1.0, delta)
        leff_pc = compute_effective_lambda(I, 1.0, 0.0, delta)
        assert abs(leff_pc - 1.0) < 1e-12, "PC effective lambda must always equal 1"
        if delta > 1.0 + 1e-10:
            assert leff_sm > 1.0, f"SM lambda_eff should exceed 1 for delta={delta}"
        elif delta < 1.0 - 1e-10:
            assert leff_sm < 1.0, f"SM lambda_eff should be below 1 for delta={delta}"


def main():
    """Run extreme policy comparison for both wage configurations."""
    verify_lambda_eff_sm_crossover(DELTA_GRID)
    crossover_delta = analytical_crossover(I)
    print(f"Analytical crossover: lambda_eff_SM = 1 iff delta = {crossover_delta}")

    all_dfs = []

    for config_label, wages in WAGE_CONFIGS.items():
        df = extreme_policy_comparison(I, ALPHAS, BETAS, GAMMAS, wages, DELTA_GRID)
        df['config'] = config_label
        all_dfs.append(df)

        # Check that SM dominates iff delta > 1
        n_agree = 0
        n_total = 0
        for _, row in df.iterrows():
            delta = row['delta']
            sm_dom = row['SM_dominates']
            if abs(delta - 1.0) > 0.05:  # skip the crossover neighbourhood
                n_total += 1
                if delta > 1.0:
                    if sm_dom:
                        n_agree += 1
                else:
                    if not sm_dom:
                        n_agree += 1
        pct = 100 * n_agree / n_total if n_total > 0 else 0.0
        print(f"{config_label}: SM-dominance direction correct in "
              f"{n_agree}/{n_total} points ({pct:.0f}%)")

    combined = pd.concat(all_dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, 'exp5_data.csv')
    combined.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # ------------------------------------------------------------------
    # Plot: income difference (SM − PC) vs delta
    # ------------------------------------------------------------------
    colors = {
        'Heterogeneous (w=[2.0, 3.0])': '#e41a1c',
        'Homogeneous (w=[2.5, 2.5])': '#377eb8',
    }
    linestyles = {
        'Heterogeneous (w=[2.0, 3.0])': '-',
        'Homogeneous (w=[2.5, 2.5])': '--',
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for config_label in WAGE_CONFIGS:
        sub = combined[combined['config'] == config_label]
        ax.plot(sub['delta'], sub['delta_agg_income'],
                color=colors[config_label], linestyle=linestyles[config_label],
                linewidth=2, label=config_label)

    ax.axvline(x=crossover_delta, color='k', linestyle=':', linewidth=1.5,
               label=f'Crossover $\\delta = {crossover_delta}$')
    ax.axhline(y=0, color='k', linewidth=0.8)
    ax.fill_betweenx(
        [ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -0.1,
         max(combined['delta_agg_income'].max() * 1.05, 0.1)],
        crossover_delta, DELTA_GRID[-1],
        alpha=0.08, color='blue', label='SM-dominant region ($\\delta > 1$)',
    )

    ax.set_xlabel('$\\delta$ (social influence parameter)')
    ax.set_ylabel('Aggregate Income Difference (SM $-$ PC)')
    ax.set_title(
        'Extreme Policy Comparison: Social Market vs Pure Capitalism\n'
        r'$\Delta\sum_i w_i L_i = \sum_i w_i L_i^{SM} - \sum_i w_i L_i^{PC}$'
    )
    ax.legend(fontsize=9)
    ax.grid(True)

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, 'exp5_extreme_policy.png')
    plt.savefig(png_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved {png_path}")

    # ------------------------------------------------------------------
    # Summary: key values at delta = {-2, 0, 1, 2, 4}
    # ------------------------------------------------------------------
    print('\nKey delta slices (heterogeneous wages):')
    sub_het = combined[combined['config'] == 'Heterogeneous (w=[2.0, 3.0])']
    for target in [-2.0, 0.0, 1.0, 2.0, 4.0]:
        idx = (sub_het['delta'] - target).abs().idxmin()
        row = sub_het.loc[idx]
        print(f"  delta={row['delta']:.2f}: lambda_eff_SM={row['lambda_eff_SM']:.3f}, "
              f"delta_income={row['delta_agg_income']:.4f}, "
              f"SM_dominates={row['SM_dominates']}")

    return combined


if __name__ == '__main__':
    df = main()
    print(f'\nTotal records: {len(df)}')
