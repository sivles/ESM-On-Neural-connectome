import numpy as np
import pandas as pd
from tqdm import tqdm  # Recommended for progress tracking

# Helpers (Assumed to exist based on your previous uploads)
from braak_stage import BRAAK_ROI_GROUPS
from plot_public import compute_prob_per_region, corr_with_pet, compute_prob_per_region2
from load import load_dti_group_sc, group_mean_pet, get_region_idx, load_atlas_with_coords

from spread_esm import EpidemicSpreadingModel


def compute_mean(my_list):
    """Computes trimmed mean (removing min and max) to handle outliers."""
    if len(my_list) < 3: return np.mean(my_list)
    array = np.array(my_list)
    sorted_arr = np.sort(array)
    trimmed_arr = sorted_arr[1:-1]
    return np.mean(trimmed_arr)


# ==========================================
# 1. Setup Data
# ==========================================
print("Loading Data...")
W_group, region_names = load_dti_group_sc()
# Note: load_atlas_with_coords usually returns names and coords
# Ensure region_names match between DTI and Atlas
region_names_atlas, coords = load_atlas_with_coords()

# Normalize W to [0, 1] (Max Normalization is correct for ESM)
if W_group.max() > 0:
    W_group = W_group / W_group.max()

# --- Load Targets (The Fix is applied here) ---
print("Loading PET Targets...")
# Target: AD Tau Pattern
X_tau_all, _ = group_mean_pet(tracer="TAU", region_names=region_names, groups=["CN_pos", "MCI_pos", "AD_pos"])
X_tau, _ = group_mean_pet(tracer="TAU", region_names=region_names, groups=["AD_pos"])
tau_prob, target_tau_prob_mean = compute_prob_per_region2(X_tau_all, X_tau)

# Modulator: Amyloid Pattern
X_amy_all, _ = group_mean_pet(tracer="AMYLOID", region_names=region_names, groups=["CN_pos", "MCI_pos", "AD_pos"])
X_amy, amy_mean = group_mean_pet("AMYLOID", region_names, ["CN_pos"])
# *** FIX: Pass X_amy, not X_tau ***
amy_prob, amy_prob_mean = compute_prob_per_region2(X_amy_all, X_amy)

# ==========================================
# 2. Define Parameter Grid
# ==========================================
# Adjust ranges as needed based on prelim results
beta_range = np.linspace(0.5, 2.5, 5)
delta_range = np.linspace(0.0, 1.5, 5)
# kappa_range = np.linspace(0.0, 2.0, 5)  # Amyloid synergy
kappa_range = [0]
noise_range = [0.001]  # Keep noise constant usually to reduce search space, or sweep small range

rep_times = 10  # Reduced for testing, increase to 10-20 for final paper
best_r = -999
best_params = {}

# epic_idx, epic_name_list = get_region_idx(braak_I_II_candidates, region_names)
# epic_idx, epic_name_list = get_region_idx(["ENT.L", "ENT.R", "HIP.L", "HIP.R"], region_names)
# Epicenters: Braak I-II (Entorhinal)
braak_I_II_candidates = BRAAK_ROI_GROUPS["I-II"]
epic_idx, epic_name_list = get_region_idx(braak_I_II_candidates, region_names)
print(f"Epicenters: {epic_name_list}")

# ==========================================
# 3. Sweep Loop
# ==========================================
# Initialize Model ONCE (efficient)
model = EpidemicSpreadingModel(W_group,
                               # region_coords=coords,
                               amy=None,
                               dt=0.05)

print(f"Starting Sweep ({len(beta_range) * len(delta_range) * len(kappa_range)} combinations)...")

# Use tqdm for progress bar
for b_o in tqdm(beta_range, desc="Beta Loop"):
    for d_o in delta_range:
        for k_o in kappa_range:
            for sigma_o in noise_range:

                runs_max_r = []
                runs_best_time = []

                for i in range(rep_times):
                    # Run simulation
                    # Years=40 is usually enough to reach saturation
                    time, P_hist = model.simulate(epic_idx, beta_o=b_o, delta_o=d_o,
                                                  noise_sigma=sigma_o, amy_kappa=k_o,
                                                  years=40,
                                                  # velocity=1500
                                                  )

                    # Pearson Correlation at every time step
                    # Optimization: Vectorized correlation is faster, but loop is fine for now
                    corrs = []
                    for t in range(P_hist.shape[0]):
                        # Comparing Simulation(t) vs Observed(Target)
                        r, _ = corr_with_pet(P_hist[t], target_tau_prob_mean)
                        corrs.append(r)

                    # Find Best Time Point for this specific run
                    max_r_idx = np.argmax(corrs)
                    max_r = corrs[max_r_idx]

                    runs_max_r.append(max_r)
                    runs_best_time.append(time[max_r_idx])

                # Average performance across repetitions
                mean_max_r = compute_mean(runs_max_r)
                mean_time = compute_mean(runs_best_time)

                # Save if best
                if mean_max_r > best_r:
                    best_r = mean_max_r
                    best_params = {
                        'beta': b_o,
                        'delta': d_o,
                        'amy_kappa': k_o,
                        'sigma_noise': sigma_o,
                        'best_time_years': mean_time,
                        'r': mean_max_r
                    }

print(f"\nOptimization Finished!")
print(f"Best Correlation (R): {best_params['r']:.4f}")
print("\n--- Best Parameters ---")
for key, value in best_params.items():
    print(f"{key}: {value}")