# DTI only have CN neg
from braak_stage import plot_braak_bars_6, BRAAK_ROI_GROUPS_6
from load import load_dti_group_sc, group_mean_pet, load_pet_table, extract_pet_matrix
from plot_public import compute_prob_per_region2, compute_prob_per_region
from plot_scatter import plot_scatter_with_braak_color

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




plot_scatter_with_braak_color(
        tau_ad_mean,
        amy_mean,
        region_names,
        title=f"CN_neg vs AD_pos SUVR",
)

ad = [tau_ad_mean, tau_ad_prob_mean, "AD_pos"]
mci = [tau_mci_mean, tau_mci_prob_mean, "MCI_pos"]
cn = [tau_cn_mean, tau_cn_prob_mean, "CN_pos"]
list = [ad, mci, cn]

for group in list:

    tau_suvr_mean = group[0]
    tau_prob_mean = group[1]
    group_name = group[2]

    # Example 1: Observed tau, suvr
    plot_braak_bars_6(
        values=tau_suvr_mean,  # (N,)
        region_names=region_names,
        BRAAK_ROI_GROUPS_6=BRAAK_ROI_GROUPS_6,
        ylabel="Model misfolded tau",
        title=f"Observed tau SUVR({group_name}) by Braak stage"
    )

    # Example 1: Observed tau, tpp
    plot_braak_bars_6(
        values=tau_prob_mean,  # (N,)
        region_names=region_names,
        BRAAK_ROI_GROUPS_6=BRAAK_ROI_GROUPS_6,
        ylabel="Model misfolded tau",
        title=f"Observed tau probability({group_name}) by Braak stage"
    )