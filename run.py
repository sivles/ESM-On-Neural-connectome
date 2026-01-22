"""
tau_sir_spreading.py

Requirements:
    pip install numpy pandas scipy scikit-learn matplotlib
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import stats
import matplotlib.pyplot as plt
from load import load_dti_group_sc, load_pet_table, extract_pet_matrix, group_mean_pet, get_region_idx, \
    load_atlas_with_coords
from braak_stage import plot_braak_spreading, compute_stage_onset_times, BRAAK_ROI_GROUPS, plot_braak_relative, \
    BRAAK_ROI_GROUPS_6, plot_braak_three_panels, plot_braak_bars_6

from plot_braak_snapshot import plot_figure_4_reproduction, plot_figure_4_surface_nilearn, plot_fig4_hybrid
from plot_public import corr_with_pet, compute_prob_per_region, compute_prob_per_region2
from plot_region_diff_bars import plot_sorted_percentage_diff
from plot_residual_surface import plot_surface_with_cbar
from plot_scatter import plot_scatter_with_braak_color
from spread_esm import call_spread
from plot_residual_glass import plot_markers_on_glassbrain_with_cbar

IMG_PATH = "./result_final/"

def plot_corr_over_time(M_hist: np.ndarray,
                        target_map: np.ndarray,
                        label: str):

    corrs = []
    for t in range(M_hist.shape[0]):
        r, _ = corr_with_pet(M_hist[t], target_map)
        corrs.append(r)
    corrs = np.array(corrs)

    fig = plt.figure(figsize=(5, 4))
    plt.plot(corrs)
    plt.xlabel("Time step")
    plt.ylabel("Correlation r")
    plt.title(f"Model vs {label} over time")
    # plt.tight_layout()
    # plt.show()

    best_t = int(corrs.argmax())
    best_r = float(corrs[best_t])
    return best_t, best_r, fig



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


def scale_sim(sim, obs):
    # paper note: "simulated values normalized to observed range"
    sim_norm = (sim - sim.min()) / (sim.max() - sim.min() + 1e-12)
    sim_scaled = obs.min() + sim_norm * (obs.max() - obs.min())
    return sim_scaled



def plot_brain(sim_scaled, obs, group_name, type_name: str, img_path):

    # load ROI coords (must match your region order)
    region_names, coords = load_atlas_with_coords()

    # paper note: "simulated values normalized to observed range"
    # sim_norm = (sim - sim.min()) / (sim.max() - sim.min() + 1e-12)
    # sim_scaled = obs.min() + sim_norm * (obs.max() - obs.min())

    # resid = obs - sim_scaled
    # resid = ((sim_scaled - obs) / (obs + 1e-9)) * 100

    if type_name == "suvr":
        x = "Tau SUVR"
    if type_name == "tpp":
        x = "Tau Probability"
        x = ""

    vmax = obs.max()

    # plot_markers_on_glassbrain_with_cbar
    # Simulated: sequential cmap
    fig = plot_surface_with_cbar(
        # coords,
        sim_scaled,
        region_names,
        None,
        title=f"Simulated {x}",
        # vmax = vmax
        # cmap="inferno",
        # vmin=float(obs.min()), vmax=float(obs.max()),
        # marker_size=60,
        # display_mode="lyrz"
    )
    fig.savefig(img_path + 'brain_sim_' + type_name + '.png'
                , dpi=300, bbox_inches='tight'
                )
    plt.close(fig)

    # Obeserved pet:
    fig = plot_surface_with_cbar(
        # coords,
        obs,
        region_names,
        None,
        title=f"Observed {x}, {group_name}",
        # vmax=vmax
        # cmap="inferno",
        # vmin=float(obs.min()), vmax=float(obs.max()),
        # marker_size=60,
        # display_mode="lyrz"
    )
    fig.savefig(img_path + 'brain_obs_' + type_name + '.png'
                , dpi=300, bbox_inches='tight'
                )
    plt.close(fig)

    # Residuals: diverging cmap, symmetric around 0
    # m = float(np.max(np.abs(resid)))
    fig = plot_surface_with_cbar(
        # coords,
        obs,
        region_names,
        sim_scaled,
        title=f"Residuals (Obs - Sim) {x}, {group_name}",
        # vmax=vmax
        # cmap="coolwarm",
        # vmin=-m, vmax=m,
        # marker_size=60,
        # display_mode="lyrz"
    )
    fig.savefig(img_path + 'brain_residuals_' + type_name + '.png'
                , dpi=300, bbox_inches='tight'
                )
    plt.close(fig)



# ---------------------------------------------------------------------
# 5. Main example pipeline
# ---------------------------------------------------------------------


if __name__ == "__main__":
    # DTI only have CN neg
    W_group, region_names = load_dti_group_sc()

    # ---- PET TAU  ----
    groups_all = ["CN_pos", "MCI_pos", "AD_pos"]
    X_tau_all, tau_mean_all = group_mean_pet("TAU", region_names, groups_all)

    X_tau_ad, tau_ad_mean = group_mean_pet("TAU", region_names, ["AD_pos"])
    X_tau_ad_prob, tau_ad_prob_mean = compute_prob_per_region2(X_tau_all, X_tau_ad)

    X_tau_mci, tau_mci_mean = group_mean_pet("TAU", region_names, ["MCI_pos"])
    X_tau_mci_prob, tau_mci_prob_mean = compute_prob_per_region2(X_tau_all, X_tau_mci)

    X_tau_cn, tau_cn_mean = group_mean_pet("TAU", region_names, ["CN_pos"])
    X_tau_cn_prob, tau_cn_prob_mean = compute_prob_per_region2(X_tau_all, X_tau_cn)

    X_tau_cn_neg, tau_cn_neg_mean = group_mean_pet("TAU", region_names, ["CN_neg"])
    X_tau_cn_neg_prob, tau_cn_neg_prob_mean = compute_prob_per_region2(X_tau_all, X_tau_cn)

    print("\n[TPP-continuum] First 5 regions and their mean TPP:")
    for name, val in zip(region_names[:5], tau_ad_prob_mean[:5]):
        print(f"  {name:10s}: {val:.3f}")

    # ---- PET AMYLOID continuum: CN_pos + MCI_pos + AD_pos (for modulation) ----
    # Modulator: Amyloid Pattern
    X_amy, amy_mean = group_mean_pet("AMYLOID", region_names, ["AD_pos"])
    # *** FIX: Pass X_amy, not X_tau ***
    amy_prob, amy_prob_mean = compute_prob_per_region(X_amy)

    # CN_neg baseline tau map (mean SUVR per region)
    df_tau_cnneg = load_pet_table(tracer="TAU", group="CN_neg")
    X_tau_cnneg = extract_pet_matrix(df_tau_cnneg, region_names)
    tau_mean_cnneg = X_tau_cnneg.mean(axis=0)



    # ---- choose Braak-like epicenters (medial temporal) ----
    braak_I_II_candidates = BRAAK_ROI_GROUPS["I-II"]
    epp = ["ENT.L", "ENT.R", "HIP.L", "HIP.R"]

    # epic_idx, epic_name_list = get_region_idx(braak_I_II_candidates, region_names)
    epic_idx, epic_name_list = get_region_idx(braak_I_II_candidates, region_names)
    print(f"\n[Model] Using epicenters: {epic_name_list}")


    """
    # ---- Braak-stage spreading analysis ----
    print("\n[Braak] Simulated spreading across Braak stages:")

    braak_stage_means = plot_braak_spreading(M_hist, region_names)
    braak_stage_means = plot_braak_relative(M_hist, region_names)

    onset_times = compute_stage_onset_times(braak_stage_means, frac_of_max=0.2)
    for stage, t in onset_times.items():
        if t is None:
            print(f"  Braak {stage}: no clear onset (curve ~ 0)")
        else:
            print(f"  Braak {stage}: reaches 20% of max at time step {t}")
    """


    _, coords = load_atlas_with_coords()
    ad_param = {"beta": 1.5, "delta": 0.375}
    ad_map = [tau_ad_mean, tau_ad_prob_mean, "AD_pos", X_tau_ad, X_tau_ad_prob, ad_param]
    mci_param = {"beta": 2.0, "delta": 0.75}
    mci_map = [tau_mci_mean, tau_mci_prob_mean, "MCI_pos", X_tau_mci, X_tau_mci_prob, mci_param]
    cn_param = {"beta": 1.5, "delta": 1.125}
    cn_map = [tau_cn_mean, tau_cn_prob_mean, "CN_pos", X_tau_cn, X_tau_cn_prob, cn_param]
    list = [ad_map, mci_map, cn_map]

    for group_data in list:
        tau_suvr_mean = group_data[0]
        tau_prob_mean = group_data[1]
        group_name = group_data[2]
        X_tau_suvr = group_data[3]
        X_tau_prob = group_data[4]
        params = group_data[5]
        path = IMG_PATH + group_name + "/"

        # ---- run SIR model ----
        # init_zero_tau_load = np.zeros(W_group.shape[0], dtype=float)
        M_hist = call_spread(W_group, epic_idx, amy_prob_mean, None, params)

        best_t_tpp, best_r_tpp, fig = plot_corr_over_time(
            M_hist, tau_prob_mean, label=f"Tau probability ({group_name})"
        )
        fig.savefig(path + 'r_time_tpp.png'
                    , dpi=300, bbox_inches='tight'
                    )
        plt.close(fig)
        print(f"  Best step (Tau probability) = {best_t_tpp}, r = {best_r_tpp:.3f}")


        print(f"\n[Fit] Correlation vs {group_name} SUVR over time:")
        best_t_suvr, best_r_suvr, fig = plot_corr_over_time(
            M_hist, tau_suvr_mean, label=f"Tau SUVR ({group_name})"
        )
        fig.savefig(path + 'r_time_suvr.png'
                    , dpi=300, bbox_inches='tight'
                    )
        plt.close(fig)
        print(f"  Best step (SUVR, {group_name}) = {best_t_suvr}, r = {best_r_suvr:.3f}")


        # ---- scatter plots at best TPP/SUVR steps ----
        best_pred_tpp = M_hist[best_t_tpp]
        pred_tpp_scaled = scale_sim(best_pred_tpp, tau_prob_mean)

        best_pred_suvr = M_hist[best_t_suvr]
        pred_suvr_scaled = scale_sim(best_pred_suvr, tau_suvr_mean)


        add_legend = False
        if group_name == "CN_pos":
            add_legend = True
        beta = params["beta"]
        delta = params["delta"]
        ratio = beta/delta
        params_desc = f"$\\beta_o={beta}$, $\delta_o={delta}$\n$\\beta_o/\delta_o={ratio:.2f}$"
        fig = plot_scatter_with_braak_color(
            pred_tpp_scaled,
            tau_prob_mean,
            region_names,
            title=f"{group_name}",
            add_legend=add_legend,
            params_desc=params_desc,
            type="Probability"
        )
        fig.savefig(path + 'fit_tpp.png'
                    , dpi=300, bbox_inches='tight'
                    )
        plt.close(fig)


        fig = plot_scatter_with_braak_color(
            pred_suvr_scaled,
            tau_suvr_mean,
            region_names,
            title=f"{group_name}",
        )
        fig.savefig(path + 'fit_suvr.png'
                    , dpi=300, bbox_inches='tight'
                    )
        plt.close(fig)


        fig = plot_scatter_with_braak_color(
            pred_tpp_scaled,
            tau_cn_neg_prob_mean,
            region_names,
            title=f"{group_name}",
        )
        fig.savefig(path + 'fit_tpp_cn_neg.png'
                    , dpi=300, bbox_inches='tight'
                    )
        plt.close(fig)
        plot_brain

        # old version resididuals
        plot_brain(pred_tpp_scaled, tau_prob_mean, group_name, "tpp", path)
        plot_brain(pred_suvr_scaled, tau_suvr_mean, group_name, "suvr", path)

        # ---------------------------------------------------------
        # NEW VISUALIZATIONS
        # ---------------------------------------------------------
        # 2. The Better Sorted Difference Plot
        fig_diff = plot_sorted_percentage_diff(
            pred_tpp_scaled,
            tau_prob_mean,
            region_names,
            title=f"Error ({group_name})"
        )
        fig_diff.savefig(path + 'bar_sorted_error_tpp.png', dpi=300)
        plt.close(fig_diff)


        # New version residuals
        # x-axis: your model output at the best step
        x_raw = pred_tpp_scaled  # (N,)
        # y-axis: your target map (TPP continuum mean)
        y_val = tau_prob_mean  # (N,)
        fig = plot_braak_three_panels(
            x_raw=x_raw,
            y_value=y_val,
            region_names=region_names,
            BRAAK_ROI_GROUPS=BRAAK_ROI_GROUPS,
            y_label="Target tau probability",
            title=f"Braak panels: target tau Probability({group_name}) vs model (step {best_t_tpp})",
            add_linear_fit=True,
            show_fit_text=True,
        )
        fig.savefig(path + 'braak_3fit_tpp.png'
                    , dpi=300, bbox_inches='tight'
                    )
        plt.close(fig)

        x_raw = pred_suvr_scaled  # (N,)
        y_val = tau_suvr_mean  # (N,)
        fig = plot_braak_three_panels(
            x_raw=x_raw,
            y_value=y_val,
            region_names=region_names,
            BRAAK_ROI_GROUPS=BRAAK_ROI_GROUPS,
            y_label="Target SUVR",
            title=f"Braak panels: target SUVR({group_name}) vs model (step {best_t_suvr})",
            add_linear_fit=True,
            show_fit_text=True,
        )
        fig.savefig(path + 'braak_3fit_suvr.png.png'
                    , dpi=300, bbox_inches='tight'
                    )
        plt.close(fig)


        # Example 1: use the model map at best_t_tpp
        fig = plot_braak_bars_6(
            values=pred_tpp_scaled,  # (N,)
            region_names=region_names,
            BRAAK_ROI_GROUPS_6=BRAAK_ROI_GROUPS_6,
            ylabel="Simulated tau-P",
            title=f"Model tau(TPP) by Braak stage ({group_name}, step {best_t_tpp})"
        )
        fig.savefig(path + 'braak_bars_tpp.png'
                    , dpi=300, bbox_inches='tight'
                    )
        plt.close(fig)

        # Example 2: use the model map at best_t_suvr
        fig = plot_braak_bars_6(
            values=pred_suvr_scaled,  # (N,)
            region_names=region_names,
            BRAAK_ROI_GROUPS_6=BRAAK_ROI_GROUPS_6,
            ylabel="Simulated tau-SUVR",
            title=f"Model tau(suvr) by Braak stage ({group_name}, step {best_t_suvr})"
        )
        fig.savefig(path + 'braak_bars_suvr.png'
                    , dpi=300, bbox_inches='tight'
                    )
        plt.close(fig)




        # paper note: "simulated values normalized to observed range"
        # sim_tpp_norm = (best_pred_tpp - best_pred_tpp.min()) / (best_pred_tpp.max() - best_pred_tpp.min() + 1e-12)
        # sim_tpp_scaled = X_tau_prob.min() + sim_tpp_norm * (X_tau_prob.max() - X_tau_prob.min())

        fig = plot_fig4_hybrid(tau_prob_mean, pred_tpp_scaled, region_names, f"Tau Probability ({group_name})")
        fig.savefig(path + 'braak_tpp_threshold.png'
                    , dpi=300, bbox_inches='tight'
                    )
        plt.close(fig)

        # paper note: "simulated values normalized to observed range"
        # sim_suvr_norm = (best_pred_suvr - best_pred_suvr.min()) / (best_pred_suvr.max() - best_pred_suvr.min() + 1e-12)
        # sim_suvr_scaled = X_tau_suvr.min() + sim_suvr_norm * (X_tau_suvr.max() - X_tau_suvr.min())
        # plot_fig4_hybrid(tau_suvr_mean, sim_suvr_scaled, region_names, f"Hybrid Coronal-Slice & Surface Views, SUVR ({group_name})")

        print("\nDone.")
