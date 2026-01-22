import numpy as np
from matplotlib import pyplot as plt

def plot_sorted_percentage_diff(sim_values, obs_values, region_names, title="Model Error"):
    """
    Plots the percentage difference sorted from largest to smallest.
    Better for identifying outliers like the Occipital lobe.
    """
    # 1. Calculate Percentage Difference (Relative Error)
    # Adding specific small epsilon to avoid div by zero if obs is 0 (unlikely for SUVR but good practice)
    # diff = ((sim_values - obs_values) / (obs_values + 1e-9)) * 100
    diff = sim_values - obs_values

    # 2. Sort indices based on the difference (descending order)
    sorted_indices = np.argsort(diff)[::-1]

    # 3. Reorder data
    sorted_diff = diff[sorted_indices]
    sorted_names = [region_names[i] for i in sorted_indices]

    # 4. Plot
    fig, ax = plt.subplots(figsize=(18, 8))

    # Color logic: Red for Overestimation (>0), Blue for Underestimation (<0)
    colors = ['red' if v >= 0 else 'blue' for v in sorted_diff]

    ax.bar(sorted_names, sorted_diff, color=colors, alpha=0.8)

    # Add a horizontal line at 0 for reference
    ax.axhline(0, color='black', linewidth=1, linestyle='--')

    ax.set_ylabel('Difference (%)\n(Sim - Obs)', fontsize=12)
    ax.set_title(f"{title} - Sorted by Deviation", fontsize=14)

    # Rotate labels 90 degrees
    ax.set_xticklabels(sorted_names, rotation=90, fontsize=9)
    ax.set_xlim(-1, len(region_names))

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig