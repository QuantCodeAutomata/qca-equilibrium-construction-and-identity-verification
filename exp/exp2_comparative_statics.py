"""Experiment 2: Labor-Market Comparative Statics.

Sweeps lambda_ubi in [0, lambda_total] while holding lambda_total=0.9 fixed,
for three values of delta (negative, one, positive). Computes and plots
equilibrium labor choices and aggregate labor income as functions of lambda_ubi,
and verifies the predicted monotonicity direction.

Outputs
-------
results/exp2_comparative_statics.png : L_1, L_2, and agg_income vs lambda_ubi
results/exp2_data.csv                : full tabular sweep results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.comparative_statics import labor_sweep

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_sweep(I, alphas, betas, gammas, wages, lambda_total,
              lambda_ubi_grid, delta, label):
    """Run labor_sweep for one delta value and attach a label column.

    Parameters
    ----------
    I : int, number of agents
    alphas, betas, gammas, wages : arrays (I,)
    lambda_total : float, fixed total labor-tax share
    lambda_ubi_grid : array, values of lambda_ubi to sweep
    delta : float, social influence parameter
    label : str, regime label added to the DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = labor_sweep(I, alphas, betas, gammas, wages, lambda_total,
                     lambda_ubi_grid, delta)
    df['delta'] = delta
    df['label'] = label
    return df


def compute_monotonicity(df):
    """Compute finite-difference sign of dL_i/d(lambda_ubi) and d(agg_income)/d(lambda_ubi).

    Returns dict with 'sign_L1', 'sign_L2', 'sign_agg', 'consistent_L1',
    'consistent_L2', 'consistent_agg' (True if sign is uniform across grid).
    """
    dL1 = np.diff(df['L_1'].values)
    dL2 = np.diff(df['L_2'].values)
    dagg = np.diff(df['agg_labor_income'].values)
    nonzero = lambda x: x[np.abs(x) > 1e-14]
    sign = lambda x: int(np.sign(np.mean(nonzero(x)))) if len(nonzero(x)) > 0 else 0
    consistent = lambda x: len(np.unique(np.sign(nonzero(x)))) <= 1 if len(nonzero(x)) > 0 else True
    return {
        'sign_L1': sign(dL1),
        'sign_L2': sign(dL2),
        'sign_agg': sign(dagg),
        'consistent_L1': consistent(dL1),
        'consistent_L2': consistent(dL2),
        'consistent_agg': consistent(dagg),
    }


def main():
    """Run comparative-statics sweep and produce plot and CSV."""
    I = 2
    alphas = np.array([0.3, 0.4])
    betas = np.array([-0.1, -0.15])
    gammas = np.array([-0.5, -0.7])
    wages = np.array([2.0, 3.0])
    lambda_total = 0.9

    # Grid: lambda_ubi in [0, lambda_total] with 100 points
    lambda_ubi_grid = np.linspace(0.0, lambda_total, 100)

    delta_regimes = {
        'delta_neg (δ=-0.5)': -0.5,
        'delta_one (δ=1.0)': 1.0,
        'delta_pos (δ=3.0)': 3.0,
    }

    all_dfs = []
    mono_rows = []

    for label, delta in delta_regimes.items():
        df = run_sweep(I, alphas, betas, gammas, wages, lambda_total,
                       lambda_ubi_grid, delta, label)
        all_dfs.append(df)

        mono = compute_monotonicity(df)
        mono_rows.append({
            'label': label,
            'delta': delta,
            **mono,
        })

        print(f"{label}: sign dL_1/dλ_ubi={mono['sign_L1']:+d}, "
              f"sign dL_2/dλ_ubi={mono['sign_L2']:+d}, "
              f"sign d(agg)/dλ_ubi={mono['sign_agg']:+d}")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Theory check: d(agg_income)/d(lambda_ubi) > 0 iff delta > 1
    for row in mono_rows:
        delta = row['delta']
        sign_agg = row['sign_agg']
        if delta > 1.0:
            assert sign_agg >= 0, (
                f"Expected positive slope for delta={delta}, got {sign_agg}")
        elif delta < 1.0:
            assert sign_agg <= 0, (
                f"Expected negative slope for delta={delta}, got {sign_agg}")
        # delta==1: slope should be ~0; no strict assertion needed

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, 'exp2_data.csv')
    combined.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # ------------------------------------------------------------------
    # Plot: 3 subplots (L_1, L_2, agg_income) vs lambda_ubi
    # ------------------------------------------------------------------
    colors = {'delta_neg (δ=-0.5)': '#e41a1c',
              'delta_one (δ=1.0)': '#4daf4a',
              'delta_pos (δ=3.0)': '#377eb8'}
    linestyles = {'delta_neg (δ=-0.5)': '--',
                  'delta_one (δ=1.0)': '-.',
                  'delta_pos (δ=3.0)': '-'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for label in delta_regimes:
        sub = combined[combined['label'] == label]
        c = colors[label]
        ls = linestyles[label]
        axes[0].plot(sub['lambda_ubi'], sub['L_1'], color=c, linestyle=ls,
                     linewidth=2, label=label)
        axes[1].plot(sub['lambda_ubi'], sub['L_2'], color=c, linestyle=ls,
                     linewidth=2, label=label)
        axes[2].plot(sub['lambda_ubi'], sub['agg_labor_income'], color=c,
                     linestyle=ls, linewidth=2, label=label)

    axes[0].set_xlabel(r'$\lambda_{\mathrm{ubi}}$')
    axes[0].set_ylabel(r'$L_1$')
    axes[0].set_title(r'Agent 1 Labor $L_1$ vs $\lambda_{\mathrm{ubi}}$')
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].set_xlabel(r'$\lambda_{\mathrm{ubi}}$')
    axes[1].set_ylabel(r'$L_2$')
    axes[1].set_title(r'Agent 2 Labor $L_2$ vs $\lambda_{\mathrm{ubi}}$')
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    axes[2].set_xlabel(r'$\lambda_{\mathrm{ubi}}$')
    axes[2].set_ylabel(r'$\sum_i w_i L_i$')
    axes[2].set_title(r'Aggregate Labor Income vs $\lambda_{\mathrm{ubi}}$')
    axes[2].legend(fontsize=8)
    axes[2].grid(True)

    plt.suptitle(
        r'Labor-Market Comparative Statics ($\lambda_{\mathrm{total}}=0.9$)',
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    png_path = os.path.join(RESULTS_DIR, 'exp2_comparative_statics.png')
    plt.savefig(png_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved {png_path}")

    # Print monotonicity summary
    mono_df = pd.DataFrame(mono_rows)
    print('\nMonotonicity summary:')
    print(mono_df.to_string(index=False))

    return combined


if __name__ == '__main__':
    df = main()
    print(f'\nTotal rows in combined data: {len(df)}')
