from typing import Union, Optional, Sequence, Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from load import load_dti_group_sc, load_atlas


class EpidemicSpreadingModel:

    def __init__(self, connectivity_matrix, amy, dt=0.1):
        """
        Initialize the ESM model.

        Args:
            connectivity_matrix (np.ndarray): NxN structural connectivity matrix (Pa_j->i).
                                              Should be normalized [0,1] probabilities.
            region_names (list): List of N region names.
            dt (float): Time step for integration (e.g., 0.1 corresponds to a fraction of a year
                        or day depending on calibration. Paper uses 1 day steps).
        """
        if connectivity_matrix.max() > 1.0:
            connectivity_matrix /= connectivity_matrix.max()
        self.W = connectivity_matrix
            # self._build_neighbor_kernel(connectivity_matrix, True))
        self.N = connectivity_matrix.shape[0]
        self.dt = dt

        if amy is not None:
            amy = np.asarray(amy, dtype=float).reshape(-1)
            if amy.size != self.N:
                raise ValueError(f"amyloid must have length {self.N}, got {amy.size}.")
            # keep bounded; treat as intensity
            amy_mod = np.clip(amy, 0.0, 1.0)
            self.amy_mod = amy_mod
        else:
            self.amy_mod = None

        # Ensure diagonal is 1 for intrinsic self-infection as per paper [cite: 739]
        np.fill_diagonal(self.W, 1.0)

    def _build_neighbor_kernel(self, W: np.ndarray, make_symmetric: bool = True) -> np.ndarray:
        """
        Return a column-stochastic neighbor transition kernel with zero diagonal.
        If make_symmetric is True, first symmetrize W to reduce acquisition asymmetries.
        """
        W = np.asarray(W, float)
        if make_symmetric:
            W = 0.5 * (W + W.T)

        # Remove self-loops for neighbor spread
        np.fill_diagonal(W, 0.0)

        # Column-stochastic: sum over i for each source j
        col_sums = W.sum(axis=0, keepdims=True)
        # Avoid divide-by-zero: only normalize columns with positive sum
        mask = col_sums > 0
        W_norm = np.zeros_like(W)
        W_norm[:, mask[0]] = W[:, mask[0]] / col_sums[:, mask[0]]

        return W_norm


    def _gini_coefficient(self, x):
        """
        Calculate Gini coefficient g(t) to measure inequality of MP burden[cite: 675].
        0 = perfect equality, 1 = perfect inequality.
        """
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(x, x)).mean()
        # Relative mean absolute difference
        rmad = mad / np.mean(x) if np.mean(x) > 0 else 0
        # Gini coefficient
        return 0.5 * rmad

    def _production_rate(self, P, beta_o):
        """
        Eq. 4: Beta_i(t) = 1 - exp(-beta_o * P_i) [cite: 677]
        """
        return 1.0 - np.exp(-beta_o * P)

    def _clearance_rate(self, P, delta_o):
        """
        Eq. 5: Delta_i(t) = exp(-delta_o * P_i) [cite: 685]
        """
        return np.exp(-delta_o * P)

    def simulate(self, epic_idx, beta_o, delta_o, noise_sigma=0.002, amy_kappa=1, years=50, t_unit_scale=365):
        """
        Run the simulation.

        Args:
            epic_idx (list of str): Names of regions to start the infection.
            beta_o (float): Global production constant[cite: 678].
            delta_o (float): Global clearance constant[cite: 687].
            noise_sigma (float): Standard deviation of additive noise[cite: 688].
            years (int): Total simulation duration.
            t_unit_scale (int): Number of steps per 'year' if dt=1 is a day.
                                Adjust depending on desired temporal resolution.
        """
        # Time setup
        steps = int(years * t_unit_scale)
        time_vector = np.linspace(0, years, steps)

        # Initialize Probabilities P [NxSteps]
        P_history = np.zeros((steps, self.N))


        P_current = np.zeros(self.N)
        P_current[epic_idx] = 0.1
        P_history[0] = P_current


        # print(f"[ESM] Simulating {years} years with seeds: {[self.regions[i] for i in epic_idx]}...")

        for t in tqdm(range(1, steps)):
            # 1. Calculate Global Gini g(t) [cite: 675]
            g = self._gini_coefficient(P_current)

            # 2. Calculate local rates based on current burden P
            beta = self._production_rate(P_current, beta_o)  # Total infection rate
            delta = self._clearance_rate(P_current, delta_o)  # Clearance rate

            # Optional amyloid modulation hook (not in baseline ref. 50 tau use-case):
            # scale production term. (Set amy_kappa=0 to disable.)
            if self.amy_mod is not None and amy_kappa != 0.0:
                beta = beta * (1.0 + float(amy_kappa) * self.amy_mod)

            # 3. Split into Intrinsic vs Extrinsic rates
            # Extrinsic: beta_ext = g * beta
            # Intrinsic: beta_int = (1-g) * beta
            beta_ext = g * beta
            beta_int = (1 - g) * beta

            # 4. Calculate Incoming Infection Force (Epsilon) [cite: 665]
            # Sum(Pa_j->i * beta_ext_j * P_j) + Pa_i->i * beta_int_i * P_i
            # Note: We use matrix multiplication for the summation.
            # Term 1 (Neighbors): W @ (beta_ext * P_current)
            # Term 2 (Self): We manually handle this to ensure diagonal W=1 usage logic

            # Exogenous force (from neighbors j to i)
            # We assume W has 0 on diagonal for this operation to strictly follow "j != i"
            # sum in Eq 2, then add intrinsic separately.
            W_nodiag = self.W.copy()
            np.fill_diagonal(W_nodiag, 0)

            # neighbor_input[i] = Sum_j( W[j,i] * beta_ext[j] * P[j] )
            # Assuming W is symmetric or W[j,i] represents j->i
            neighbor_input = W_nodiag @ (beta_ext * P_current)

            # Self input
            self_input = beta_int * P_current  # Assuming Pa_ii = 1

            epsilon = neighbor_input + self_input

            # 5. Add Noise [cite: 660]
            noise = np.random.normal(0, noise_sigma, self.N)

            # 6. Differential Update (Euler Method) for Eq. 1
            # dP/dt = (1 - P)*epsilon - delta*P + noise
            dP = (1 - P_current) * epsilon - delta * P_current + noise

            P_next = P_current + dP * self.dt

            # Clip to valid probability range [0, 1]
            P_next = np.clip(P_next, 0.0, 1.0)

            # Store and update
            P_history[t] = P_next
            P_current = P_next

        return time_vector, P_history

def call_spread(
    W: np.ndarray,
    epic_idx: Union[int, Sequence[int]],
    amy_mod: Optional[Dict[str, Any]] = None,
    init_tau_load = None,
    params = None
) -> np.ndarray:

    if params is not None:
        BETA_O = params["beta"]
        DELTA_O = params["delta"]
    else:
        # 2. Define Parameters
        # Values chosen to mimic dynamics in paper (need fitting for real data)
        # The paper mentions optimizing these per subject, but we use defaults for simulation.
        BETA_O = 1.5  # Production rate constant
        DELTA_O = 0.375 # Clearance rate constant (AD often has low clearance)

    AMY_KAPPA = 0
    NOISE = 0.001  # Model noise

    # 3. Initialize Model
    esm = EpidemicSpreadingModel(W, amy_mod, dt=0.05)

    # 5. Run Simulation
    time, P_history = esm.simulate(epic_idx, beta_o=BETA_O, delta_o=DELTA_O, amy_kappa=AMY_KAPPA, noise_sigma=NOISE, years=40)
    return P_history


# -----------------------------------------------------------
# Execution Block
# -----------------------------------------------------------

if __name__ == "__main__":
    # 1. Load Data using your helper script
    # This automatically loads atlas and computes group SC from 'data/DTI'
    try:
        W_group, region_names = load_dti_group_sc()

        # Preprocessing W:
        # The paper defines W as "Anatomical Connection Probability" [0,1].
        # DTI matrices are often streamline counts. We must normalize them.
        # A common robust normalization is dividing by the max value or row sums.
        # Here we normalize by max to treat the strongest connection as probability ~1.
        if W_group.max() > 1.0:
            print("[ESM] Normalizing DTI matrix to [0,1] range.")
            W_group = W_group / W_group.max()

    except Exception as e:
        print(f"Error loading data: {e}")
        # Create dummy data if load.py fails (for testing)
        region_names = [f"Region_{i}" for i in range(10)]
        W_group = np.random.rand(10, 10)
        np.fill_diagonal(W_group, 1)
        W_group = (W_group + W_group.T) / 2

    # 2. Define Parameters
    # Values chosen to mimic dynamics in paper (need fitting for real data)
    # The paper mentions optimizing these per subject, but we use defaults for simulation.
    BETA_O = 2.0  # Production rate constant
    DELTA_O = 1.5  # Clearance rate constant (AD often has low clearance)
    AMY_KAPPA = 1
    NOISE = 0.001  # Model noise

    # 3. Initialize Model
    esm = EpidemicSpreadingModel(W_group, region_names, dt=0.05)

    # 4. Define Seeds
    # Paper identifies Posterior Cingulate (PCC) and Anterior Cingulate (ACC)
    # as common outbreak regions.
    # We try to find them in your atlas strings.
    seeds = []
    for r in region_names:
        if "Cingulum_Post" in r or "PCC" in r or "Cingulum_Ant" in r:
            seeds.append(r)

    if not seeds:
        seeds = [region_names[0]]  # Fallback

    print(f"Selected seeds: {seeds}")

    # 5. Run Simulation
    time, P_history = esm.simulate(seeds, beta_o=BETA_O, delta_o=DELTA_O, noise_sigma=NOISE, amy_kappa=AMY_KAPPA, years=40)

    # 6. Visualization
    plt.figure(figsize=(12, 6))

    # Plot global mean burden
    mean_burden = P_history.mean(axis=0)
    plt.plot(time, mean_burden, 'k--', linewidth=2, label="Global Mean Burden")

    # Plot specific regions (Seeds vs Others)
    for idx, region in enumerate(region_names):
        if region in seeds:
            plt.plot(time, P_history[idx, :], color='red', alpha=0.8, label=f"Seed: {region}")
        elif idx % 10 == 0:  # Plot a few non-seeds for context
            plt.plot(time, P_history[idx, :], color='blue', alpha=0.1)

    plt.title(f"ESM Simulation: Amyloid Propagation (Seeds: {seeds[0]}...)")
    plt.xlabel("Simulation Time (Years)")
    plt.ylabel("Probability of MP Burden (P)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Optional: Save result
    # pd.DataFrame(P_history.T, columns=region_names, index=time).to_csv("esm_simulation_results.csv")