import numpy as np
import matplotlib.pyplot as plt
from braak_stage import BRAAK_ROI_GROUPS_6
from load import group_mean_pet, load_dti_group_sc
from plot_public import compute_prob_per_region2


def _get_braak_stats(data_matrix, region_names, braak_groups):
    """
    Helper: Computes Mean and SEM for each Braak stage (I-VI).
    data_matrix: (n_subjects, n_regions)
    """
    stages = ["I", "II", "III", "IV", "V", "VI"]
    means = []
    sems = []

    for st in stages:
        rois = braak_groups[st]
        idx = [region_names.index(r) for r in rois if r in region_names]

        if not idx:
            means.append(0)
            sems.append(0)
            continue

        # 1. Average across ROIs within the stage for each subject
        # Shape: (n_subjects,)
        stage_scores = np.nanmean(data_matrix[:, idx], axis=1)

        # 2. Average across subjects
        m = np.nanmean(stage_scores)

        # 3. SEM across subjects
        n = np.sum(np.isfinite(stage_scores))
        sd = np.nanstd(stage_scores, ddof=1)
        sem = sd / np.sqrt(n) if n > 0 else 0.0

        means.append(m)
        sems.append(sem)

    return stages, means, sems


def plot_observed_suvr(X_suvr, region_names, braak_groups=BRAAK_ROI_GROUPS_6):
    """Plots the Observed SUVR bar chart separately."""
    stages, means, sems = _get_braak_stats(X_suvr, region_names, braak_groups)

    plt.figure(figsize=(5, 4))

    # Style settings matching your image
    bar_opts = dict(color='#999999', capsize=0, width=0.75)
    err_opts = dict(ecolor='#4d4d4d', elinewidth=2.5, capthick=0)

    plt.bar(stages, means, yerr=sems, error_kw=err_opts, **bar_opts)

    plt.ylabel("Observed SUVR", fontsize=16)
    plt.xlabel("Braak stage", fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.ylim(bottom=1.0)  # Adjust based on your SUVR range (usually > 1)

    plt.tight_layout()
    plt.show()


def plot_observed_tau_p(X_prob, region_names, braak_groups=BRAAK_ROI_GROUPS_6):
    """Plots the Observed tau-P bar chart separately."""
    stages, means, sems = _get_braak_stats(X_prob, region_names, braak_groups)

    plt.figure(figsize=(5, 4))

    # Style settings matching your image
    bar_opts = dict(color='#999999', capsize=0, width=0.75)
    err_opts = dict(ecolor='#4d4d4d', elinewidth=2.5, capthick=0)

    plt.bar(stages, means, yerr=sems, error_kw=err_opts, **bar_opts)

    plt.ylabel("Observed tau-P", fontsize=16)
    plt.xlabel("Braak stage", fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.ylim(bottom=0.0)

    plt.tight_layout()
    plt.show()


def plot_observed_tau_p_with_threshold(X_prob, region_names, braak_groups=BRAAK_ROI_GROUPS_6):
    """Plots the Observed tau-P bar chart separately with thresholds."""
    stages, means, sems = _get_braak_stats(X_prob, region_names, braak_groups)

    fig = plt.figure(figsize=(5, 4))

    # Style settings matching your image
    bar_opts = dict(color='#999999', capsize=0, width=0.75)
    err_opts = dict(ecolor='#4d4d4d', elinewidth=2.5, capthick=0)

    # Plot the bars
    plt.bar(stages, means, yerr=sems, error_kw=err_opts, **bar_opts)

    # --- ADDED: Horizontal Threshold Lines ---
    thresholds = [0.35, 0.25, 0.15, 0.05]

    # Loop through thresholds to add lines
    for th in thresholds:
        plt.axhline(y=th, color='tab:red', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Threshold {th}')

    plt.ylabel("Observed tau-P", fontsize=16)
    plt.xlabel("Braak stage", fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.ylim(bottom=0.0)

    # Add legend to explain thresholds
    # bbox_to_anchor moves it slightly outside if needed, or remove it to keep inside
    plt.legend(loc='upper right', fontsize='small', frameon=True)

    plt.tight_layout()
    plt.show()



W_group, region_names = load_dti_group_sc()

groups_all = ["MCI_pos", "CN_pos", "AD_pos"]
X_tau_all, tau_mean_all = group_mean_pet("TAU", region_names, groups_all)
X_tau_all_prob, tau_all_prob_mean = compute_prob_per_region2(X_tau_all, X_tau_all)

X_tau_ad, tau_ad_mean = group_mean_pet("TAU", region_names, ["AD_pos"])
X_tau_ad_prob, tau_ad_prob_mean = compute_prob_per_region2(X_tau_all, X_tau_ad)

X_tau_mci, tau_mci_mean = group_mean_pet("TAU", region_names, ["MCI_pos"])
X_tau_mci_prob, tau_mci_prob_mean = compute_prob_per_region2(X_tau_all, X_tau_mci)

X_tau_cn, tau_cn_mean = group_mean_pet("TAU", region_names, ["CN_pos"])
X_tau_cn_prob, tau_cn_prob_mean = compute_prob_per_region2(X_tau_all, X_tau_cn)


# Call the new function using the AD group data
# Plot 1: SUVR
plot_observed_suvr(X_tau_all, region_names)
# Plot 2: Tau Probability
plot_observed_tau_p(X_tau_all_prob, region_names)
plot_observed_tau_p_with_threshold(X_tau_all_prob, region_names)


# Call the new function using the AD group data
# Plot 1: SUVR
plot_observed_suvr(X_tau_ad, region_names)
# Plot 2: Tau Probability
plot_observed_tau_p(X_tau_ad_prob, region_names)
plot_observed_tau_p_with_threshold(X_tau_ad_prob, region_names)


# Call the new function using the AD group data
# Plot 1: SUVR
plot_observed_suvr(X_tau_mci, region_names)
# Plot 2: Tau Probability
plot_observed_tau_p(X_tau_mci_prob, region_names)
plot_observed_tau_p_with_threshold(X_tau_mci_prob, region_names)

