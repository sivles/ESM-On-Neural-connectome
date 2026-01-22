import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from braak_stage import BRAAK_ROI_GROUPS
# Import your existing simulation tools
from spread_esm import EpidemicSpreadingModel
from load import load_dti_group_sc, load_atlas_with_coords, group_mean_pet, get_region_idx
from plot_public import compute_prob_per_region, corr_with_pet

import bct  # Brain Connectivity Toolbox


def generate_null_model_shuffle(W_real):
    # --- CREATE NULL NETWORK ---
    # Strategy: Randomly shuffle the weights of the connectivity matrix
    # This preserves the distribution of weights but destroys topology.
    W_null = W_real.flatten()
    np.random.shuffle(W_null)
    W_null = W_null.reshape(W_real.shape)
    return W_null


def generate_null_model_bct(W_real):
    # BCT function: null_model_und_sign
    # It randomizes the matrix while preserving degree and strength distributions.
    # We pass the binary matrix to preserve topology, or weighted for strength.

    # Standard approach: Rewire the network
    # bin_swaps=5 means each edge is swapped approx 5 times
    W_null, _ = bct.null_model_und_sign(W_real, bin_swaps=5, wei_freq=0.1)
    return W_null

def get_max_r2(model, epic_idx, target_pattern, params):
    """
    Runs the model once and returns the best R^2 across all time points.
    """
    _, P_hist = model.simulate(epic_idx, **params)

    # Calculate R (correlation) for every time step
    corrs = [corr_with_pet(P_hist[t], target_pattern)[0] for t in range(P_hist.shape[0])]
    best_r = np.max(corrs)

    # Return R^2 (Coefficient of Determination)
    return best_r ** 2


def run_null_analysis(n_permutations=100):
    print(f"--- Starting Null Model Analysis (N={n_permutations}) ---")

    # 1. Load Real Data
    W_real, region_names = load_dti_group_sc()
    region_names, coords = load_atlas_with_coords()

    # Normalize Real W
    if W_real.max() > 1.0: W_real /= W_real.max()

    # Load Target (Observed Tau)
    X_tau, _ = group_mean_pet("TAU", region_names, ["AD_pos"])
    tau_target, _ = compute_prob_per_region(X_tau)
    obs_mean = np.nanmean(tau_target, axis=0)

    # Load Modulator (Amyloid)
    X_amy, _ = group_mean_pet("AMYLOID", region_names, ["CN_pos"])
    amy_prob, amy_mean = compute_prob_per_region(X_amy)

    # Define Best Parameters (Use values found in your sweep!)
    # CHECK: Update these with your "Best Params" from the previous step
    params = {
        'beta_o': 1.4,
        'delta_o': 1.2,
        'amy_kappa': 0.1,
        'noise_sigma': 0.001,
        'years': 40
        # , 'velocity': 1500
    }

    # Define Seed (Use the "PHG" fix we discussed)
    # seeds = [i for i, r in enumerate(region_names) if "PHG" in r or "Parahippocampal" in r]

    braak_I_II_candidates = BRAAK_ROI_GROUPS["I-II"]
    epp = ["ENT.L", "ENT.R", "HIP.L", "HIP.R"]
    # epic_idx, epic_name_list = get_region_idx(braak_I_II_candidates, region_names)
    epic_idx, epic_name_list = get_region_idx(braak_I_II_candidates, region_names)

    # 2. Get Real Performance
    print("Running Real Model...")
    esm_real = EpidemicSpreadingModel(W_real,
                                      # coords,
                                      amy_mean, dt=0.1)
    real_r2 = get_max_r2(esm_real, epic_idx, obs_mean, params)
    print(f"Real Model R^2: {real_r2:.4f}")

    # 3. Run Null Permutations
    null_r2_scores = []

    print("Running Null Models...")
    for i in tqdm(range(n_permutations)):
        # --- CREATE NULL NETWORK ---
        # Strategy: Randomly shuffle the weights of the connectivity matrix
        # This preserves the distribution of weights but destroys topology.
        W_null = generate_null_model_bct(W_real)

        # Ensure diagonal is 0 (no self-loops in W for calculation)
        np.fill_diagonal(W_null, 0)

        # Run Simulation with Null W
        # Note: We keep coordinates same (assuming geometry is fixed),
        # or you could shuffle coords too for a spatial null.
        # Shuffling weights is standard for "Network Null".
        esm_null = EpidemicSpreadingModel(W_null,
                                          # coords,
                                          amy_mean, dt=0.1)

        null_r2 = get_max_r2(esm_null, epic_idx, obs_mean, params)
        null_r2_scores.append(null_r2)

    # 4. Calculate Statistics
    null_r2_scores = np.array(null_r2_scores)
    mean_null = np.mean(null_r2_scores)
    p_value = np.sum(null_r2_scores >= real_r2) / n_permutations

    # Handling case where p=0 (none were better)
    if p_value == 0:
        p_str = f"< {1 / n_permutations}"
    else:
        p_str = f"= {p_value:.4f}"

    print(f"\n--- Results ---")
    print(f"Real R^2: {real_r2:.4f}")
    print(f"Null Mean R^2: {mean_null:.4f} (SD={np.std(null_r2_scores):.4f})")
    print(f"P-value: {p_str}")

    # 5. Plot Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(null_r2_scores, bins=15, color='gray', alpha=0.7, label='Null Models')
    plt.axvline(real_r2, color='red', linestyle='--', linewidth=2, label=f'Real Model (R²={real_r2:.2f})')
    plt.axvline(mean_null, color='black', linestyle=':', linewidth=2, label=f'Null Mean (R²={mean_null:.2f})')

    plt.xlabel('Model Performance (R²)')
    plt.ylabel('Frequency')
    plt.title(f'Null Model Analysis (N={n_permutations})\np {p_str}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_null_analysis(n_permutations=100)