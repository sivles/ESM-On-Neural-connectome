import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from load import load_pet_table, extract_pet_matrix, load_atlas

# Import your simulation/data variables
from braak_stage import BRAAK_ROI_GROUPS_6
from plot_public import corr_with_pet, compute_prob_per_region, find_best_model_time_per_subject


# -----------------------------------------------------------------------------
# 1. Helper: Map Region Name -> Braak Color
# -----------------------------------------------------------------------------
def get_braak_colors(region_names):
    """
    Returns a list of colors for each region based on its Braak stage.
    """
    # Define colors for stages I-VI (matching standard schemes or the paper)
    # Paper uses: Blue, Orange, Green, Red, Purple, Brown
    stage_colors = {
        "I": "#1f77b4",  # Blue
        "II": "#ff7f0e",  # Orange
        "III": "#2ca02c",  # Green
        "IV": "#d62728",  # Red
        "V": "#9467bd",  # Purple
        "VI": "#8c564b"  # Brown
    }

    colors = []
    labels = []

    for r in region_names:
        found = False
        for stage, rois in BRAAK_ROI_GROUPS_6.items():
            # Check if region 'r' is in this stage list (handling .L/.R suffix)
            if r in rois:
                colors.append(stage_colors[stage])
                labels.append(f"stage_{stage}")
                found = True
                break
        if not found:
            colors.append("#7f7f7f")  # Grey for unassigned
            labels.append("Unassigned")

    return colors, labels, stage_colors

def plot_model_vs_pet(model_map: np.ndarray,
                      pet_map: np.ndarray,
                      title: str,
                      type: str = "SUVR"):
    # --- clean inputs (drop NaNs / infs) ---
    x = np.asarray(model_map, dtype=float)
    y = np.asarray(pet_map, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # --- compute correlation and OLS fit ---
    r, p = corr_with_pet(x, y)  # your existing helper
    lr = stats.linregress(x, y)  # slope, intercept, rvalue, pvalue, stderr

    # --- plot scatter and fit line ---
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y)
    if x.size >= 2:
        xx = np.linspace(x.min(), x.max(), 100)
        yy = lr.intercept + lr.slope * xx
        plt.plot(xx, yy, linewidth=2, label=f"Fit: y = {lr.slope:.3g}x + {lr.intercept:.3g}")
        plt.legend(loc="best")

    # --- labels and annotation ---
    plt.xlabel("Simulated misfolded tau")
    plt.ylabel(f"Observed misfolded tau ({type})")
    plt.text(
        0.05, 0.95, f"r = {r:.3f}\np = {p:.1e}",
        transform=plt.gca().transAxes,
        va="top", ha="left"
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return lr  # gives access to slope/intercept/etc.


def plot_scatter_with_braak_color(model_map: np.ndarray,
                                  pet_map: np.ndarray,
                                  region_names,
                                  title: str,
                                  add_legend = True,
                                  params_desc = "",
                                  type: str = "SUVR"):

    # --- clean inputs (drop NaNs / infs) ---
    mean_pred = np.asarray(model_map, dtype=float)
    mean_obs = np.asarray(pet_map, dtype=float)
    mask = np.isfinite(mean_obs) & np.isfinite(mean_pred)
    mean_pred, mean_obs = mean_pred[mask], mean_obs[mask]

    # Get colors
    dot_colors, dot_labels, color_map = get_braak_colors(region_names)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(7.5, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.4])

    # 2. Scatter Plot (Middle/Right)
    # The paper shows multiple groups, we will plot the main "CN Ab-" group here
    ax1 = fig.add_subplot(gs[0])


    # Scatter points colored by Braak Stage
    for stg, color in color_map.items():
        # Find indices for this stage
        indices = [i for i, label in enumerate(dot_labels) if label == f"stage_{stg}"]

        if add_legend:
            label = f"Stage {stg}"
        else:
            label = None

        if indices:
            ax1.scatter(mean_pred[indices], mean_obs[indices],
                        c=color, label=label, s=20, alpha=0.8)

    # Linear Fit Line
    slope, intercept, r_val, p_val, std_err = stats.linregress(mean_pred, mean_obs)
    r, p = corr_with_pet(mean_pred, mean_obs)

    x_line = np.linspace(0, max(mean_pred) * 1.1, 100)
    ax1.plot(x_line, slope * x_line + intercept, color='gray', alpha=0.5, linewidth=1.5)

    # Shaded confidence interval (approximate)
    ax1.fill_between(x_line,
                     (slope - std_err) * x_line + intercept,
                     (slope + std_err) * x_line + intercept,
                     color='gray', alpha=0.1)

    # Annotation stats
    ax1.text(0.55, 0.1, f"{params_desc}\n\n$r^2 = {r_val ** 2:.2f}$ \n $r = {r:.2f}$", transform=ax1.transAxes, fontsize=12)

    ax1.set_xlabel(f"ESM-predicted tau {type}", fontsize=12)
    ax1.set_ylabel(f"Observed tau {type}", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 3. Legend (Right Panel)
    if (add_legend):
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        handles, labels = ax1.get_legend_handles_labels()
        ax2.legend(handles, labels, title="Braak Stage", loc='center left', frameon=False, fontsize=12)

    # plt.suptitle("Figure 5 Reproduction: Model Performance in A$\\beta$- Subjects", fontsize=16)
    # plt.tight_layout()
    # plt.show()

    return fig


# -----------------------------------------------------------------------------
# 2. Main Plotting Function
# -----------------------------------------------------------------------------
def plot_figure_5_reproduction(region_names):
    """
    Reproduces Figure 5:
    a) Histogram of individual R^2 model fits.
    b) Scatter plots of Predicted vs Observed Tau (colored by Braak Stage).
    """

    # --- A. Load & Filter Data (Ab- Negatives) ---
    print("Loading Ab- Negative Data...")

    # Load separate groups
    try:
        df_cn_neg = load_pet_table("TAU", "CN_neg")
        X_cn_neg = extract_pet_matrix(df_cn_neg, region_names)

        # You can add MCI_neg if you have that file, otherwise use CN_neg
        # df_mci_neg = load_pet_table("TAU", "MCI_neg")
        # ...

        # For this example, we use all CN_neg as the main "Abeta- group"
        X_target = X_cn_neg
        print(f"Target Sample: {X_target.shape[0]} subjects (CN Ab-)")

    except Exception as e:
        print(f"Data loading error: {e}")
        return

    # --- B. Compute Tau Probabilities (TPP) ---
    # We need the TPP (0-1 probability) for the scatter plots [cite: 53]
    tpp_obs, _ = compute_prob_per_region(X_target)

    # --- C. Get Model Predictions ---
    # Find best fit time point for each subject in this specific group
    # M_hist is your global simulation history from sir.py
    from run import M_hist
    X_pred = find_best_model_time_per_subject(tpp_obs, M_hist)

    # --- D. Panel A: Histogram of Individual Fits ---
    print("Calculating individual fits...")
    r_squared_values = []
    for i in range(X_target.shape[0]):
        # Correlation between Subject Observed vs Subject Predicted
        # Only use regions with valid data
        obs = tpp_obs[i, :]
        pred = X_pred[i, :]
        if np.std(obs) > 1e-5 and np.std(pred) > 1e-5:
            r, _ = stats.pearsonr(obs, pred)
            r_squared_values.append(r ** 2)

    # --- E. Panel B: Scatter Plots (Group Means) ---
    # Average across all subjects in the group
    mean_obs = np.nanmean(tpp_obs, axis=0)
    mean_pred = np.nanmean(X_pred, axis=0)

    # Get colors
    dot_colors, dot_labels, color_map = get_braak_colors(region_names)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])

    # 1. Histogram (Left)
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(r_squared_values, bins=15, color='gray', alpha=0.7, edgecolor='none')
    ax1.set_xlabel(r"Within-subject $R^2$", fontsize=12)
    ax1.set_ylabel("N (Subjects)", fontsize=12)
    ax1.set_title("Individual Model Fit (CN Ab-)", fontsize=14)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 2. Scatter Plot (Middle/Right)
    # The paper shows multiple groups, we will plot the main "CN Ab-" group here
    ax2 = fig.add_subplot(gs[1])

    # Scatter points colored by Braak Stage
    for stg, color in color_map.items():
        # Find indices for this stage
        indices = [i for i, label in enumerate(dot_labels) if label == f"stage_{stg}"]
        if indices:
            ax2.scatter(mean_pred[indices], mean_obs[indices],
                        c=color, label=f"Stage {stg}", s=20, alpha=0.8)

    # Linear Fit Line
    slope, intercept, r_val, p_val, std_err = stats.linregress(mean_pred, mean_obs)
    x_line = np.linspace(0, max(mean_pred) * 1.1, 100)
    ax2.plot(x_line, slope * x_line + intercept, color='gray', alpha=0.5, linewidth=1.5)

    # Shaded confidence interval (approximate)
    ax2.fill_between(x_line,
                     (slope - std_err) * x_line + intercept,
                     (slope + std_err) * x_line + intercept,
                     color='gray', alpha=0.1)

    # Annotation stats
    ax2.text(0.55, 0.1, f"$r^2 = {r_val ** 2:.2f}$", transform=ax2.transAxes, fontsize=12)

    ax2.set_xlabel("ESM-predicted tau probability", fontsize=12)
    ax2.set_ylabel("Observed tau probability", fontsize=12)
    ax2.set_title("CN A$\\beta$- Group", fontsize=14)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 3. Legend (Right Panel)
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    handles, labels = ax2.get_legend_handles_labels()
    ax3.legend(handles, labels, title="Braak Stage", loc='center left', frameon=False, fontsize=12)

    plt.suptitle("Figure 5 Reproduction: Model Performance in A$\\beta$- Subjects", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Ensure atlas is loaded
    region_names = load_atlas()
    plot_figure_5_reproduction(region_names)