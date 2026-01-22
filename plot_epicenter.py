import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting


def plot_epicenters_glass(epicenter_names, epic_indices, all_coords, title="Epicenters"):
    """
    Plots specific 'Epicenter' regions as large red spheres on a glass brain.

    Args:
        epicenter_names: List of strings (e.g. ['ENT.L', 'HIP.L'])
        all_region_names: List of all region names in your atlas.
        all_coords: (N, 3) array of coordinates for all regions.
    """
    print(f"Plotting {len(epicenter_names)} epicenters...")

    # 1. Identify Indices and Coords of Epicenters
    epic_coords = []

    for idx in epic_indices:
        epic_coords.append(all_coords[idx])

    if not epic_coords:
        print("Error: No valid epicenters found.")
        return

    epic_coords = np.array(epic_coords)

    # 2. Prepare Plot Data
    # We create a list of colors/sizes.
    # Let's plot ONLY the epicenters to keep it clean.

    # Node colors: 'red' for all epicenters
    node_colors = ['red'] * len(epic_coords)

    # Node size: 100 (adjust as needed)
    node_sizes = [80] * len(epic_coords)

    # 3. Plot Glass Brain
    # We use 'plot_markers' which places spheres on the glass brain
    display = plotting.plot_markers(
        node_values=node_sizes,  # Determines size scalar (can be fixed)
        node_coords=epic_coords,  # XYZ coordinates
        # node_color=node_colors,  # List of colors
        display_mode='lyrz',  # Views: Left, Y-coronal, Right, Z-axial
        title=title,
        node_size=80,  # Base scaling factor
        alpha=0.8  # Transparency
    )

    plt.show()


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    from load import load_atlas_with_coords, get_region_idx
    from braak_stage import BRAAK_ROI_GROUPS_6, BRAAK_ROI_GROUPS

    # 1. Load Atlas
    region_names, coords = load_atlas_with_coords()

    # epic_idx, epic_name_list = get_region_idx(braak_I_II_candidates, region_names)
    # epic_idx, epic_name_list = get_region_idx(["ENT.L", "ENT.R", "HIP.L", "HIP.R"], region_names)
    # Epicenters: Braak I-II (Entorhinal)
    braak_I_II_candidates = BRAAK_ROI_GROUPS["I-II"]
    epic_idx, epic_name_list = get_region_idx(braak_I_II_candidates, region_names)
    print(f"Epicenters: {epic_name_list}")

    # 3. Plot
    plot_epicenters_glass(epic_name_list, epic_idx, coords, title="Modeled Epicenters (Braak I)")