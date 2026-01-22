# =====================================================================
# ADD TO END OF sir.py: Figure 4 Reproduction Code
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, image, surface, plotting
from nilearn.image import new_img_like
import xml.etree.ElementTree as ET

from brainspace.plotting import plot_hemispheres
from brainspace.datasets import load_conte69, load_parcellation
from brainspace.utils.parcellation import map_to_labels

from braak_stage import ABBREV_TO_AAL_SPM12
from load import load_atlas_with_coords
from plot_public import find_best_model_time_per_subject


def plot_figure_4_reproduction(X_tpp_obs, pred_mean, coords, title):
    """
    Reproduces Figure 4: Spatial patterns thresholded at different levels.
    Rows: Thresholds [0.35, 0.25, 0.15, 0.05] (Early -> Late stages)
    Cols: Observed Mean vs. Predicted Mean
    """

    # 1. Calculate Population Means (Group Average)
    obs_mean = np.nanmean(X_tpp_obs, axis=0)
    # pred_mean = np.nanmean(X_tpp_pred, axis=0)

    # Thresholds from the paper (Fig 4 caption)
    # 0.35 (Early/High burden) -> 0.05 (Late/Widespread)
    thresholds = [0.35, 0.25, 0.15, 0.05]

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10, 12))

    # Common plotting parameters
    vmin, vmax = 0.0, 0.6  # Adjust based on your data range
    cmap = 'YlOrRd'  # "Warmer colors"
    node_size = 40

    print("\n[Fig 4] Generating plots for thresholds:", thresholds)

    for i, thr in enumerate(thresholds):
        # --- Prepare Observed Data for this Stage ---
        # Mask values below threshold
        obs_masked = obs_mean.copy()
        obs_masked[obs_masked < thr] = np.nan  # Hide low values

        # --- Prepare Predicted Data for this Stage ---
        pred_masked = pred_mean.copy()
        pred_masked[pred_masked < thr] = np.nan

        # --- Plot Observed (Left Column) ---
        ax_obs = axes[i, 0]
        plotting.plot_markers(
            node_values=obs_masked,
            node_coords=coords,
            node_cmap=cmap,
            node_vmin=vmin,
            node_vmax=vmax,
            node_size=node_size,
            display_mode='lr',  # Left and Right hemispheres (Lateral view)
            axes=ax_obs,
            colorbar=False,
            alpha=0.8
        )

        # --- Plot Predicted (Right Column) ---
        ax_pred = axes[i, 1]
        plotting.plot_markers(
            node_values=pred_masked,
            node_coords=coords,
            node_cmap=cmap,
            node_vmin=vmin,
            node_vmax=vmax,
            node_size=node_size,
            display_mode='lr',
            axes=ax_pred,
            colorbar=(i == 0),  # Only show colorbar on top row
            alpha=0.8
        )

        # Labels
        if i == 0:
            ax_obs.set_title("Observed Pattern", fontsize=14, fontweight='bold')
            ax_pred.set_title("ESM-Predicted Pattern", fontsize=14, fontweight='bold')

        # Add threshold text to the left
        ax_obs.text(-0.1, 0.5, f"Thr > {thr}", transform=ax_obs.transAxes,
                    rotation=90, va='center', fontsize=12, fontweight='bold')

    plt.suptitle(title, y=0.96, fontsize=16)
    # Note: tight_layout handles nilearn plots poorly, so we adjust manually if needed
    # plt.subplots_adjust(hspace=0.0)
    plt.show()


def create_aal_texture2(data_vector, region_names):
    # =========================================================================
    # CHANGE: Load Manually Downloaded Files
    # =========================================================================
    # Update this path to where you extracted the files
    path_to_nii = "data/aal/atlas/AAL.nii"
    path_to_xml = "data/aal/atlas/AAL.xml"

    print(f"Loading local atlas from: {path_to_nii}")

    # 1. Load the Image
    atlas_nii = image.load_img(path_to_nii)
    atlas_data = atlas_nii.get_fdata()

    # 2. Parse the XML to get Labels (Standard AAL list)
    # The XML structure usually has <label><name>Precentral_L</name>...</label>
    try:
        tree = ET.parse(path_to_xml)
        root = tree.getroot()
        # Find all 'name' tags inside 'label' tags
        aal_labels = [item.text for item in root.findall(".//label/name")]
    except Exception as e:
        print(f"Error reading XML labels: {e}")
        # Fallback: If XML fails, you might need to paste the standard list manually
        return None, None

    # =========================================================================
    # The rest of the logic remains the same
    # =========================================================================

    unique_ids = np.unique(atlas_data)
    unique_ids = unique_ids[unique_ids != 0]  # Remove background
    unique_ids = np.sort(unique_ids)

    new_data = np.zeros_like(atlas_data)
    matched_count = 0

    # Map Your Data -> Atlas Regions
    for i, label_std in enumerate(aal_labels):
        if i >= len(unique_ids): break

        region_id = unique_ids[i]
        label_str = str(label_std)

        match_val = 0.0
        for my_idx, my_name in enumerate(region_names):
            if '.' in my_name:
                code, hemi = my_name.split('.')
                # Use the translation dictionary I provided earlier!
                target_base = ABBREV_TO_AAL_SPM12.get(code, code)
                aal_hemi_suffix = "_" + hemi

                if target_base in label_str and aal_hemi_suffix in label_str:
                    match_val = data_vector[my_idx]
                    if match_val != 0:
                        matched_count += 1
                    break

        if match_val != 0:
            new_data[atlas_data == region_id] = match_val

    print(f"--> Successfully matched {matched_count} regions from local AAL atlas.")

    # Project to Surface
    stat_img = new_img_like(atlas_nii, new_data)
    fsaverage = datasets.fetch_surf_fsaverage()
    texture_lat = surface.vol_to_surf(stat_img, fsaverage.pial_left)

    return fsaverage, texture_lat


def get_stat_img():
    path_to_nii = "data/aal/atlas/AAL.nii"
    path_to_xml = "data/aal/atlas/AAL.xml"

    print(f"Loading local atlas from: {path_to_nii}")

    # 1. Load the Image
    atlas_nii = image.load_img(path_to_nii)
    atlas_data = atlas_nii.get_fdata()

    new_data = np.zeros_like(atlas_data)

    stat_img = new_img_like(atlas_nii, new_data)
    fsaverage = datasets.fetch_surf_fsaverage()
    texture_lat = surface.vol_to_surf(stat_img, fsaverage.pial_left)
    surface.load_surf_data()
    return stat_img, fsaverage, texture_lat



def plot_figure_4_surface_nilearn(X_tpp_obs, pred_mean, region_names, coords):
    # 1. Prepare Data
    print("Preparing prediction data...")
    #X_tpp_pred = find_best_model_time_per_subject(tau_TPP, M_hist)
    obs_mean = np.nanmean(X_tpp_obs, axis=0)
    #pred_mean = np.nanmean(X_tpp_pred, axis=0)

    # 2. Create Surface Textures
    print("Projecting AAL volume to Surface mesh (this may take a moment)...")
    mesh, tex_obs = create_aal_texture2(obs_mean, region_names)
    _, tex_pred = create_aal_texture2(pred_mean, region_names)

    # 3. Plot Grid
    thresholds = [0.35, 0.25, 0.15, 0.05]
    # We want 4 rows. Each row has Obs(Lat), Obs(Med), Pred(Lat), Pred(Med)
    # But to keep it simple and fit on screen, let's do 4 rows x 2 cols (Lateral View only)
    # or use the subplots argument in plot_surf_roi

    fig, axes = plt.subplots(nrows=4, ncols=2, subplot_kw={'projection': '3d'}, figsize=(8, 12))

    print("Rendering surfaces...")
    for i, thr in enumerate(thresholds):
        # Plot Observed (Left Col)
        # We use 'threshold=thr' to hide low values!
        plotting.plot_surf_roi(
            mesh.infl_left, roi_map=tex_obs,
            hemi='left', view='lateral',
            bg_map=mesh.sulc_left, bg_on_data=True, darkness=.5,
            threshold=thr,  # <--- THIS IS THE MAGIC KEY
            cmap='YlOrRd', vmin=0, vmax=0.6,
            axes=axes[i, 0], title=None
        )

        # Plot Predicted (Right Col)
        plotting.plot_surf_roi(
            mesh.infl_left, roi_map=tex_pred,
            hemi='left', view='lateral',
            bg_map=mesh.sulc_left, bg_on_data=True, darkness=.5,
            threshold=thr,  # <--- Apply threshold here
            cmap='YlOrRd', vmin=0, vmax=0.6,
            axes=axes[i, 1], title=None
        )

        # Add text label manually
        axes[i, 0].text2D(-0.1, 0.5, f"Thr > {thr}", transform=axes[i, 0].transAxes,
                          rotation=90, va='center', fontweight='bold')

    axes[0, 0].set_title("Observed (Lateral)", fontsize=14)
    axes[0, 1].set_title("Predicted (Lateral)", fontsize=14)

    plt.suptitle("Figure 4: Progression (Surface Render)", y=0.95, fontsize=16)
    plt.show()


def create_aal_volume_and_surface(data_vector, region_names):
    """
    1. Loads AAL Nifti (Volume).
    2. Maps your data to it (to get the Coronal Slice).
    3. Projects it to fsaverage (to get the Medial Surface).
    """
    # PATHS TO YOUR MANUAL DOWNLOAD
    path_to_nii = "data/aal/atlas/AAL.nii"
    path_to_xml = "data/aal/atlas/AAL.xml"

    print(f"Loading atlas from: {path_to_nii}")
    atlas_nii = image.load_img(path_to_nii)
    atlas_data = atlas_nii.get_fdata()

    # Parse XML for labels
    try:
        tree = ET.parse(path_to_xml)
        root = tree.getroot()
        aal_labels = [item.text for item in root.findall(".//label/name")]
    except Exception as e:
        print(f"XML Error: {e}")
        return None, None

    # Map Data to Volume
    unique_ids = np.unique(atlas_data)
    unique_ids = np.sort(unique_ids[unique_ids != 0])

    new_data = np.zeros_like(atlas_data)

    for i, label_std in enumerate(aal_labels):
        if i >= len(unique_ids): break
        region_id = unique_ids[i]
        label_str = str(label_std)

        match_val = 0.0
        for my_idx, my_name in enumerate(region_names):
            if '.' in my_name:
                code, hemi = my_name.split('.')
                target_base = ABBREV_TO_AAL_SPM12.get(code, code)

                # Check match (Name + Hemisphere)
                if target_base in label_str and ("_" + hemi) in label_str:
                    match_val = data_vector[my_idx]
                    break

        if match_val != 0:
            new_data[atlas_data == region_id] = match_val

    # Create the Volume Image (for Coronal View)
    vol_img = new_img_like(atlas_nii, new_data)

    # Project to Surface (for Medial View)
    fsaverage = datasets.fetch_surf_fsaverage()
    tex_left = surface.vol_to_surf(vol_img, fsaverage.pial_left)

    return vol_img, fsaverage, tex_left


def plot_fig4_hybrid_old(obs_mean, pred_mean, region_names, title):
    # 1. Calculate Prediction
    # X_tpp_pred = find_best_model_time_per_subject(tau_TPP, M_hist)
    # pred_mean = np.nanmean(X_tpp_pred, axis=0)
    # obs_mean = np.nanmean(X_tpp_obs, axis=0)

    # 2. Generate Maps (Volume & Surface)
    print("Generating Observed Maps...")
    vol_obs, mesh, tex_obs = create_aal_volume_and_surface(obs_mean, region_names)
    print("Generating Predicted Maps...")
    vol_pred, _, tex_pred = create_aal_volume_and_surface(pred_mean, region_names)

    # 3. Plotting Configuration
    thresholds = [0.35, 0.25, 0.15, 0.05]

    # Grid: 4 Rows (Thresholds) x 4 Columns
    # Col 1: Obs Coronal | Col 2: Obs Medial | Col 3: Pred Coronal | Col 4: Pred Medial
    fig = plt.figure(figsize=(12, 12))

    # Use GridSpec for control
    gs = fig.add_gridspec(4, 4)

    print("Rendering hybrid views...")

    for i, thr in enumerate(thresholds):
        # --- COLUMN 1: Observed Coronal Slice (Volume) ---
        ax1 = fig.add_subplot(gs[i, 0])
        plotting.plot_stat_map(
            vol_obs,
            display_mode='y',  # 'y' = Coronal view only
            cut_coords=[-15],  # Cut at y=-15 (Hippocampus/Entorhinal area)
            threshold=thr,  # Apply threshold
            cmap='YlOrRd', vmin=0, vmax=0.6,
            colorbar=False, axes=ax1, annotate=False,
            bg_img=datasets.load_mni152_template(),  # Background brain
            dim=-0.5  # Dim background
        )

        # --- COLUMN 2: Observed Medial Surface ---
        ax2 = fig.add_subplot(gs[i, 1], projection='3d')
        plotting.plot_surf_roi(
            mesh.infl_left, roi_map=tex_obs,
            hemi='left', view='medial',
            bg_map=mesh.sulc_left, bg_on_data=True, darkness=.5,
            threshold=thr, cmap='YlOrRd', vmin=0, vmax=0.6,
            axes=ax2, title=None
        )

        # --- COLUMN 3: Predicted Coronal Slice ---
        ax3 = fig.add_subplot(gs[i, 2])
        plotting.plot_stat_map(
            vol_pred,
            display_mode='y',
            cut_coords=[-15],
            threshold=thr,
            cmap='YlOrRd', vmin=0, vmax=0.6,
            colorbar=False, axes=ax3, annotate=False,
            bg_img=datasets.load_mni152_template(),
            dim=-0.5
        )

        # --- COLUMN 4: Predicted Medial Surface ---
        ax4 = fig.add_subplot(gs[i, 3], projection='3d')
        plotting.plot_surf_roi(
            mesh.infl_left, roi_map=tex_pred,
            hemi='left', view='medial',
            bg_map=mesh.sulc_left, bg_on_data=True, darkness=.5,
            threshold=thr, cmap='YlOrRd', vmin=0, vmax=0.6,
            axes=ax4, title=None
        )

        # Add Threshold Label
        ax1.text(-0.2, 0.5, f"Thr > {thr}", transform=ax1.transAxes,
                 rotation=90, va='center', fontweight='bold', fontsize=12)

    # Headers
    fig.text(0.25, 0.92, "Observed Pattern", ha='center', fontsize=16, fontweight='bold')
    fig.text(0.75, 0.92, "Predicted Pattern", ha='center', fontsize=16, fontweight='bold')

    # Sub-headers
    fig.text(0.16, 0.89, "Coronal", ha='center', fontsize=10)
    fig.text(0.36, 0.89, "Medial", ha='center', fontsize=10)
    fig.text(0.63, 0.89, "Coronal", ha='center', fontsize=10)
    fig.text(0.84, 0.89, "Medial", ha='center', fontsize=10)

    plt.suptitle(title, y=0.98, fontsize=18)
    # plt.tight_layout() # Tight layout often breaks mixed 2D/3D plots, allow manual spacing
    # plt.show()
    return fig


# =====================================================================
# MODIFIED FUNCTION FOR plot_braak_snapshot.py
# =====================================================================

def plot_fig4_hybrid(obs_mean, pred_mean, region_names, title):
    """
    Plots a hybrid 4x4 grid:
    Rows: Threshold levels [0.35, 0.25, 0.15, 0.05]
    Cols: Obs(Coronal), Obs(Medial), Pred(Coronal), Pred(Medial)

    Modifications:
    - Added explicit colorbars with value ticks.
    - Ensures consistent range (vmin/vmax) across the row.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    from nilearn import plotting, datasets

    # 1. Generate Maps (Volume & Surface)
    # We create the texture maps once; masking happens during plotting
    print("Generating Observed Maps...")
    vol_obs, mesh, tex_obs = create_aal_volume_and_surface(obs_mean, region_names)
    print("Generating Predicted Maps...")
    vol_pred, _, tex_pred = create_aal_volume_and_surface(pred_mean, region_names)

    # 2. Plotting Configuration
    thresholds = [0.35, 0.25, 0.15, 0.05]

    # Common Scale for all plots to ensure comparability
    vmin, vmax = 0.0, 0.6
    cmap = 'YlOrRd'

    # Create a ScalarMappable for the colorbars
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Dummy array for the mappable

    # Grid: 4 Rows (Thresholds) x 4 Columns
    fig = plt.figure(figsize=(14, 12))  # Slightly wider to accommodate colorbars
    gs = fig.add_gridspec(4, 5, width_ratios=[1, 1, 1, 1, 0.1])
    # Added 5th narrow column for the row-specific colorbar

    print("Rendering hybrid views with value-labeled colorbars...")

    for i, thr in enumerate(thresholds):
        # --- COLUMN 1: Observed Coronal Slice (Volume) ---
        ax1 = fig.add_subplot(gs[i, 0])
        plotting.plot_stat_map(
            vol_obs,
            display_mode='y',
            cut_coords=[-15],
            threshold=thr,
            cmap=cmap, vmin=vmin, vmax=vmax,
            colorbar=False,  # We add manual colorbar later
            axes=ax1, annotate=False,
            bg_img=datasets.load_mni152_template(),
            dim=-0.5
        )

        # --- COLUMN 2: Observed Medial Surface ---
        ax2 = fig.add_subplot(gs[i, 1], projection='3d')
        plotting.plot_surf_roi(
            mesh.infl_left, roi_map=tex_obs,
            hemi='left', view='medial',
            bg_map=mesh.sulc_left, bg_on_data=True, darkness=.5,
            threshold=thr,
            cmap=cmap, vmin=vmin, vmax=vmax,
            colorbar=False,  # Disable default nilearn bar
            axes=ax2, title=None
        )

        # --- COLUMN 3: Predicted Coronal Slice ---
        ax3 = fig.add_subplot(gs[i, 2])
        plotting.plot_stat_map(
            vol_pred,
            display_mode='y',
            cut_coords=[-15],
            threshold=thr,
            cmap=cmap, vmin=vmin, vmax=vmax,
            colorbar=False,
            axes=ax3, annotate=False,
            bg_img=datasets.load_mni152_template(),
            dim=-0.5
        )

        # --- COLUMN 4: Predicted Medial Surface ---
        ax4 = fig.add_subplot(gs[i, 3], projection='3d')
        plotting.plot_surf_roi(
            mesh.infl_left, roi_map=tex_pred,
            hemi='left', view='medial',
            bg_map=mesh.sulc_left, bg_on_data=True, darkness=.5,
            threshold=thr,
            cmap=cmap, vmin=vmin, vmax=vmax,
            colorbar=False,
            axes=ax4, title=None
        )

        # --- COLUMN 5: Shared Colorbar for the Row ---
        ax5 = fig.add_subplot(gs[i, 4])
        # Add the colorbar to this specific axis
        cbar = fig.colorbar(sm, cax=ax5, orientation='vertical')

        # Add explicit ticks (values)
        # We can show 0, max, and maybe the threshold?
        # Or just standard intervals: 0.0, 0.2, 0.4, 0.6
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6])
        cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6'])
        cbar.ax.tick_params(labelsize=10)

        # Add Threshold Label to the far left (Column 1)
        ax1.text(-0.2, 0.5, f"Thr > {thr}", transform=ax1.transAxes,
                 rotation=90, va='center', fontweight='bold', fontsize=12)

    # Headers
    fig.text(0.28, 0.92, "Observed Pattern", ha='center', fontsize=16, fontweight='bold')
    fig.text(0.68, 0.92, "Predicted Pattern", ha='center', fontsize=16, fontweight='bold')

    # Sub-headers
    # Adjusted positions slightly for the new grid layout
    fig.text(0.18, 0.89, "Coronal", ha='center', fontsize=10)
    fig.text(0.38, 0.89, "Medial", ha='center', fontsize=10)
    fig.text(0.58, 0.89, "Coronal", ha='center', fontsize=10)
    fig.text(0.78, 0.89, "Medial", ha='center', fontsize=10)

    plt.suptitle(title, y=0.98, fontsize=18)

    return fig

# ==========================================
# Execution of Fig 4 code
# ==========================================
if __name__ == "__main__":
    # Ensure we have the necessary data variables from previous steps
    # We need: X_tpp_obs (Observed TPP), X_tpp_pred (Predicted TPP), and coords

    try:
        # Load coords if not already in scope
        if 'coords' not in locals():
            _, coords = load_atlas_with_coords()

        # Check if predictions exist (from Fig 2 step); if not, generate them
        if 'X_tpp_pred' not in locals():
            print("[Fig 4] Generating subject predictions...")
            X_tpp_pred = find_best_model_time_per_subject(tau_TPP, M_hist)

        # Run plotting
        plot_figure_4_reproduction(tau_TPP, X_tpp_pred, region_names, coords)

    except NameError as e:
        print(f"Missing data for Figure 4: {e}")
        print("Ensure you have run the main model loading/simulation blocks first.")