# ---------------------------------------------------------------------
# Braak-stage ROI groups (AAL-style labels) — adjust to your atlas
# ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

BRAAK_ROI_GROUPS_6 = {
    "I": [
        "PHG.L", "PHG.R"
    ],  # Transentorhinal (AAL PHG includes Entorhinal)

    "II": [
        "HIP.L", "HIP.R"
    ],  # Hippocampus

    "III": [
        "AMYG.L", "AMYG.R", "FFG.L", "FFG.R", "LING.L", "LING.R",
        "OLF.L", "OLF.R"
    ],  # Limbic (Amygdala, Fusiform, Lingual, Olfactory)

    "IV": [
        "MTG.L", "MTG.R", "ITG.L", "ITG.R", "TPOsup.L", "TPOsup.R",
        "TPOmid.L", "TPOmid.R"
    ],  # Temporal Association (Middle/Inf Temporal, Temporal Pole)

    "V": [
        # Frontal Association
        "SFG.L", "SFG.R", "MFG.L", "MFG.R",
        "IFGoperc.L", "IFGoperc.R", "IFGtriang.L", "IFGtriang.R", "IFGorb .L", "IFGorb .R",
        "SFGmedial.L", "SFGmedial.R", "PFCventmed.L", "PFCventmed.R",
        "REC.L", "REC.R", "OFCmed.L", "OFCmed.R", "OFCant.L", "OFCant.R",
        "OFCpost.L", "OFCpost.R", "OFClat.L", "OFClat.R",
        # Cingulate & Insula
        "ACC.L", "ACC.R", "MCC.L", "MCC.R", "PCC.L", "PCC.R", "INS.L", "INS.R",
        # Parietal Association
        "SPG.L", "SPG.R", "IPG.L", "IPG.R", "SMG.L", "SMG.R", "ANG.L", "ANG.R",
        "PCUN.L", "PCUN.R", "ROL.L", "ROL.R",
        # Occipital Association
        "SOG.L", "SOG.R", "MOG.L", "MOG.R", "IOG.L", "IOG.R", "CUN.L", "CUN.R",
        # Subcortical (Basal Ganglia & Thalamus - often grouped with Assoc or late stage)
        "CAU.L", "CAU.R", "PUT.L", "PUT.R", "PAL.L", "PAL.R", "THA.L", "THA.R",
        "STG.L", "STG.R"
    ],  # Higher Order Association (Frontal, Parietal, Occipital, Subcortical)

    "VI": [
        "PreCG.L", "PreCG.R", "PoCG.L", "PoCG.R",
        "CAL.L", "CAL.R", "HES.L", "HES.R",
        "SMA.L", "SMA.R", "PCL.L", "PCL.R"
    ]  # Primary Sensory/Motor
}


# ---------------------------------------------------------------------
# Optional: keep your original 3-bin merged staging (I-II, III-IV, V-VI)
# by merging the 6-stage dict above.
# ---------------------------------------------------------------------
BRAAK_ROI_GROUPS = {
    "I-II":  BRAAK_ROI_GROUPS_6["I"] + BRAAK_ROI_GROUPS_6["II"],
    "III-IV": BRAAK_ROI_GROUPS_6["III"] + BRAAK_ROI_GROUPS_6["IV"],
    "V-VI":  BRAAK_ROI_GROUPS_6["V"] + BRAAK_ROI_GROUPS_6["VI"],
}

# This dictionary translates your short codes to the specific names used in AAL (SPM12)
ABBREV_TO_AAL_SPM12 = {
    "PreCG": "Precentral",
    "SFG": "Frontal_Sup",         # Note: AAL splits SFG into Sup, Sup_Medial, etc.
    "SFGmed": "Frontal_Sup_Medial",
    "MFG": "Frontal_Mid",
    "IFGoperc": "Frontal_Inf_Oper",
    "IFGtriang": "Frontal_Inf_Tri",
    "IFGorb": "Frontal_Inf_Orb",
    "ROL": "Rolandic_Oper",
    "SMA": "Supp_Motor_Area",
    "OLF": "Olfactory",
    "REC": "Rectus",
    "INS": "Insula",
    "ACG": "Cingulum_Ant",
    "MCG": "Cingulum_Mid",
    "PCG": "Cingulum_Post",
    "HIP": "Hippocampus",
    "PHG": "ParaHippocampal",
    "AMYG": "Amygdala",
    "CAL": "Calcarine",
    "CUN": "Cuneus",
    "LING": "Lingual",
    "SOG": "Occipital_Sup",
    "MOG": "Occipital_Mid",
    "IOG": "Occipital_Inf",
    "FFG": "Fusiform",
    "PostCG": "Postcentral",
    "SPG": "Parietal_Sup",
    "IPL": "Parietal_Inf",
    "SMG": "SupraMarginal",
    "ANG": "Angular",
    "PCUN": "Precuneus",
    "PCL": "Paracentral_Lobule",
    "HES": "Heschl",
    "STG": "Temporal_Sup",
    "TPOsup": "Temporal_Pole_Sup",
    "MTG": "Temporal_Mid",
    "TPOmid": "Temporal_Pole_Mid",
    "ITG": "Temporal_Inf",
    "ENT": "ParaHippocampal", # Fallback: AAL often includes Entorhinal in PHG
}


# ---------------------------------------------------------------------
# Braak-stage analysis: mean simulated tau per stage and onset times
# ---------------------------------------------------------------------
def compute_braak_stage_means(M_hist: np.ndarray,
                              region_names: list[str]):
    """
    Compute mean simulated tau per Braak stage over time.

    M_hist : (T, N) misfolded tau history.
    region_names : list of N atlas labels.
    braak_rois : dict mapping stage name -> list of ROI labels.

    Returns
    -------
    stage_means : dict[stage] -> np.ndarray of shape (T,)
    """
    stage_indices = {}
    for stage, rois in BRAAK_ROI_GROUPS.items():
        idx = [region_names.index(r) for r in rois if r in region_names]
        if idx:
            stage_indices[stage] = np.array(idx, dtype=int)
        else:
            print(f"[Braak] Warning: no ROIs for stage {stage} found in atlas.")

    stage_means = {}
    for stage, idx in stage_indices.items():
        stage_means[stage] = M_hist[:, idx].mean(axis=1)
    return stage_means

def compute_braak_stage_sum(M_hist: np.ndarray,
                              region_names: list[str]):
    """
    Compute mean simulated tau per Braak stage over time.

    M_hist : (T, N) misfolded tau history.
    region_names : list of N atlas labels.
    braak_rois : dict mapping stage name -> list of ROI labels.

    Returns
    -------
    stage_means : dict[stage] -> np.ndarray of shape (T,)
    """
    stage_indices = {}
    for stage, rois in BRAAK_ROI_GROUPS.items():
        idx = [region_names.index(r) for r in rois if r in region_names]
        if idx:
            stage_indices[stage] = np.array(idx, dtype=int)
        else:
            print(f"[Braak] Warning: no ROIs for stage {stage} found in atlas.")

    stage_sum = {}
    for stage, idx in stage_indices.items():
        stage_sum[stage] = M_hist[:, idx].sum(axis=1)
    return stage_sum


def compute_stage_onset_times(stage_means: dict[str, np.ndarray],
                              frac_of_max: float = 0.2):
    """
    For each Braak stage, find the first time step where the mean
    simulated tau reaches a given fraction of its maximum.

    frac_of_max : e.g. 0.2 = 20% of max value.

    Returns
    -------
    onset_times : dict[stage] -> int or None
    """
    onset_times = {}
    for stage, curve in stage_means.items():
        if curve.max() <= 0:
            onset_times[stage] = None
            continue
        thr = frac_of_max * curve.max()
        idx = np.where(curve >= thr)[0]
        onset_times[stage] = int(idx[0]) if idx.size > 0 else None
    return onset_times

def plot_braak_spreading(M_hist: np.ndarray,
                         region_names: list[str]):
    """
    Plot average simulated tau for each Braak stage vs time.
    """
    stage_means = compute_braak_stage_sum(M_hist, region_names)

    # enforce plotting order I-II, III-IV, V-VI if present
    order = [s for s in ["I-II", "III-IV", "V-VI"] if s in stage_means]

    offsets = {"I-II": 0.0, "III-IV": 0.01, "V-VI": 0.02}

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    for stage in order:
        plt.plot(stage_means[stage] + offsets[stage], label=f"Braak {stage}")
    # plot any other stages that might exist
    for stage in stage_means:
        if stage not in order:
            plt.plot(stage_means[stage] + offsets[stage], label=f"Braak {stage}")

    plt.xlabel("Time step")
    plt.ylabel("Mean simulated tau")
    plt.title("Simulated tau spread across Braak stages")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return stage_means


def plot_braak_relative(M_hist, region_names):
    stage_means = compute_braak_stage_means(M_hist, region_names)
    total_mean = M_hist.mean(axis=1)  # (T,)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))

    order = [s for s in ["I-II", "III-IV", "V-VI"] if s in stage_means]
    colors = {"I-II": "tab:blue", "III-IV": "tab:orange", "V-VI": "tab:green"}

    offsets = {"I-II": 0.0, "III-IV": 0.1, "V-VI": 0.2}
    for stage in order:
        frac = stage_means[stage] / (total_mean + 1e-12)
        plt.plot(frac + offsets[stage], label=f"Braak {stage}", color=colors.get(stage, None))

    plt.xlabel("Time step")
    plt.ylabel("Fraction of total simulated tau")
    plt.title("Relative tau burden per Braak stage")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return stage_means



def _as_2d(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    if A.ndim == 1:
        return A[None, :]  # (1, n_regions)
    if A.ndim == 2:
        return A
    raise ValueError(f"Expected 1D or 2D array, got shape {A.shape}")

def _braak_flatten_2d(X2: np.ndarray, Y2: np.ndarray, region_names, braak_rois):
    idx = [region_names.index(r) for r in braak_rois if r in region_names]
    if len(idx) == 0:
        raise ValueError("No Braak ROIs found in region_names for this stage.")
    x = X2[:, idx].ravel()
    y = Y2[:, idx].ravel()
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def plot_braak_three_panels(
    x_raw,                  # (n_regions,) or (n_samples, n_regions)
    y_value,                # same shape as x_raw
    region_names,           # list[str]
    BRAAK_ROI_GROUPS,       # dict like {"I-II":[...], "III-IV":[...], "V-VI":[...]}
    *,
    stages=("I-II", "III-IV", "V-VI"),
    y_label="Model value",
    title="",
    add_binned_curve=False,
    n_bins=30,
    add_linear_fit=True,
    show_fit_text=True,
    scatter_kwargs=None,
    binned_kwargs=None,
    fit_kwargs=None,
):
    X2 = _as_2d(x_raw)
    Y2 = _as_2d(y_value)
    if X2.shape != Y2.shape:
        raise ValueError(f"x_raw shape {X2.shape} must match y_value shape {Y2.shape}")

    scatter_kwargs = scatter_kwargs or dict(s=8, alpha=0.18)
    binned_kwargs = binned_kwargs or dict(linewidth=2)
    fit_kwargs = fit_kwargs or dict(linewidth=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True,
                             constrained_layout=True)

    for ax, st in zip(axes, stages):
        x, y = _braak_flatten_2d(X2, Y2, region_names, BRAAK_ROI_GROUPS[st])
        ax.scatter(x, y, **scatter_kwargs)
        ax.set_title(f"Braak {st}")
        ax.set_xlabel("Raw tau (x)")

        # Optional: binned mean curve
        if add_binned_curve and x.size >= 20 and np.nanmin(x) < np.nanmax(x):
            bins = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
            which = np.digitize(x, bins) - 1
            xc, yc = [], []
            for b in range(n_bins):
                m = which == b
                if np.any(m):
                    xc.append(np.nanmean(x[m]))
                    yc.append(np.nanmean(y[m]))
            if len(xc) >= 2:
                ax.plot(xc, yc, **binned_kwargs)

        # Linear fit line (per panel)
        if add_linear_fit and x.size >= 2 and np.nanmin(x) < np.nanmax(x):
            # polyfit can fail if x is degenerate; guard above helps
            a, b = np.polyfit(x, y, deg=1)
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            yy = a * xx + b
            ax.plot(xx, yy, **fit_kwargs)

            if show_fit_text:
                # R^2
                yhat = a * x + b
                ss_res = np.sum((y - yhat) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

                r, p = stats.pearsonr(x, y)
                ax.text(
                    0.05, 0.95,
                    f"y = {a:.3g}x + {b:.3g}\n$R$ = {r:.3f}",
                    transform=ax.transAxes,
                    va="top", ha="left"
                )

    axes[0].set_ylabel(y_label)

    if title:
        fig.suptitle(title)  # no y=... needed usually
    # DO NOT call fig.tight_layout() when constrained_layout=True
    # plt.show()
    return fig


def plot_braak_bars_6(
    values,                 # (N,) or (n_samples, N)
    region_names,           # list[str] length N
    BRAAK_ROI_GROUPS_6,     # dict: {"I":[...], ..., "VI":[...]}
    *,
    ylabel="Simulated tau-P",
    title="",
):
    """
    Bar plot over Braak stages I..VI with mean ± SEM.

    - If values is (N,), SEM is computed across ROIs within each stage (proxy).
    - If values is (n_samples, N), SEM is computed across samples (recommended).
    """
    V = np.asarray(values, dtype=float)
    if V.ndim == 1:
        V = V[None, :]  # (1, N)

    stages = ["I", "II", "III", "IV", "V", "VI"]
    means, sems, counts = [], [], []

    for st in stages:
        rois = BRAAK_ROI_GROUPS_6[st]
        idx = [region_names.index(r) for r in rois if r in region_names]
        if len(idx) == 0:
            means.append(np.nan); sems.append(np.nan); counts.append(0)
            continue

        # Per-sample stage mean: average across ROIs in this stage
        stage_per_sample = np.nanmean(V[:, idx], axis=1)  # (n_samples,)

        m = float(np.nanmean(stage_per_sample))
        # SEM across samples (if n_samples>1). If n_samples==1, SEM will be 0.
        n = int(np.sum(np.isfinite(stage_per_sample)))
        s = float(np.nanstd(stage_per_sample, ddof=1)) if n > 1 else 0.0
        sem = s / np.sqrt(n) if n > 0 else np.nan

        means.append(m); sems.append(sem); counts.append(len(idx))


    x = np.arange(len(stages))

    fig = plt.figure(figsize=(5, 4))
    # Style settings matching your image
    bar_opts = dict(color='#999999', capsize=0, width=0.75)
    err_opts = dict(ecolor='#4d4d4d', elinewidth=2.5, capthick=0)

    plt.bar(x, means,
            # yerr=sems,
            # error_kw=err_opts,
            **bar_opts)

    # --- NEW: Explicitly set x-ticks to show all 6 numbers ---
    plt.xticks(x, stages)

    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel("Braak stage", fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.ylim(bottom=0.0)

    return fig