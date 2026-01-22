import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting

import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
from plot_braak_snapshot import create_aal_volume_and_surface


def plot_surface_with_cbar_old(obs_vector, region_names, pred_vector=None,
                           title="Surface Plot", kind='auto', threshold=None, vmax=None):
    """
    Plots surface maps with smart transparency for zero values.

    Args:
        obs_vector: (N,) Array of Observed values (or single data vector if pred_vector is None).
        region_names: List of region names.
        pred_vector: (N,) Array of Predicted values (Optional).
                     - If None: Plots obs_vector using Standard logic (Zero = Transparent).
                     - If provided: Plots (Obs - Pred) using Residual logic:
                         * Both 0: Transparent (Brain Texture).
                         * Identical (Obs==Pred!=0): Median Color (White).
        title: Title of figure.
        kind: 'auto', 'standard', or 'residuals' (overrides automatic detection).
    """

    # --- 1. Prepare Data & Logic ---
    data_to_plot = obs_vector.copy()

    # Check if we are doing a comparison (Residuals) or a single plot
    if pred_vector is not None:
        # --- RESIDUAL MODE ---
        is_residual = True

        # Calculate Difference
        residuals = obs_vector - pred_vector
        data_to_plot = residuals

        # LOGIC:
        # 1. "Both Zero" -> Transparent (NaN)
        # 2. "Perfect Fit" (Resid=0 but values!=0) -> Keep as 0 (White/Median color)

        # Create mask where BOTH are effectively zero
        inactive_mask = (np.abs(obs_vector) < 1e-5) & (np.abs(pred_vector) < 1e-5)

        # Set inactive regions to NaN (Transparent)
        # Note: We do this AFTER mapping to surface usually, or map NaNs carefully.
        # Here we prepare the vector with NaNs.
        data_to_plot[inactive_mask] = np.nan

        # Set Plotting Parameters
        if kind == 'auto' or kind == 'residuals':
            cmap = 'coolwarm'  # Blue-White-Red

            # Symmetric Limits
            limit = np.nanmax(np.abs(data_to_plot)) if vmax is None else vmax
            val_min, val_max = -limit, limit

            print(f"Residual Mode: [{-limit:.3f}, {limit:.3f}]")
            print("  - Both=0 -> Transparent")
            print("  - Obs=Pred -> Median Color (White)")

    else:
        # --- STANDARD MODE (Single Vector) ---
        is_residual = False
        if kind == 'residuals': is_residual = True  # Force manual override

        # LOGIC:
        # 1. Zero -> Transparent (NaN)
        data_to_plot[np.abs(data_to_plot) < 1e-5] = np.nan

        # Set Plotting Parameters
        if kind == 'auto' or kind == 'standard':
            cmap = 'YlOrRd'
            val_min = 0
            val_max = np.nanmax(data_to_plot) if vmax is None else vmax
            print(f"Standard Mode: [0, {val_max:.3f}] (Zeros -> Transparent)")
        else:
            # Fallback if user forced 'residuals' style on single vector
            cmap = 'coolwarm'
            limit = np.nanmax(np.abs(data_to_plot))
            val_min, val_max = -limit, limit

    # --- 2. Map Data to Texture ---
    # We use your imported helper.
    # NOTE: vol_to_surf sometimes converts NaNs to 0s depending on interpolation.
    # We will re-apply the NaN mask on the texture if needed, but usually this works.
    _, mesh, tex_map = create_aal_volume_and_surface(data_to_plot, region_names)

    # --- 3. Plotting ---
    fig = plt.figure(figsize=(12, 6))

    # Lateral View
    ax1 = plt.subplot(1, 2, 1, projection='3d')
    plotting.plot_surf_roi(
        mesh.infl_left, roi_map=tex_map, hemi='left', view='lateral',
        bg_map=mesh.sulc_left, bg_on_data=True, darkness=0.5,
        threshold=threshold, cmap=cmap, vmin=val_min, vmax=val_max,
        axes=ax1, title="Lateral View"
    )

    # Medial View
    ax2 = plt.subplot(1, 2, 2, projection='3d')
    plotting.plot_surf_roi(
        mesh.infl_left, roi_map=tex_map, hemi='left', view='medial',
        bg_map=mesh.sulc_left, bg_on_data=True, darkness=0.5,
        threshold=threshold, cmap=cmap, vmin=val_min, vmax=val_max,
        axes=ax2, title="Medial View"
    )

    # --- 4. Colorbar ---
    cax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(val_min, val_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)

    if is_residual:
        cbar.set_label('Difference (Obs - Pred)')
        # Add interpretation text
        plt.figtext(0.99, 0.80, "Red: Underestimated\n(Obs > Pred)", ha='right', color='darkred', fontsize=9)
        plt.figtext(0.99, 0.15, "Blue: Overestimated\n(Obs < Pred)", ha='right', color='darkblue', fontsize=9)
    else:
        cbar.set_label('Value / Probability')

    plt.suptitle(title, fontsize=16, y=0.95)
    # plt.show()
    return fig


def plot_surface_with_cbar(obs_vector, region_names, pred_vector=None,
                           title="Surface Plot", kind='auto', threshold=None, vmax=None):
    """
    Plots surface maps with 2 views (Lateral, Medial) and a single shared colorbar.
    - Title position lowered.
    - Aspect ratio fixed.
    """

    # --- 1. Prepare Data & Logic ---
    data_to_plot = obs_vector.copy()

    if pred_vector is not None:
        is_residual = True

        # --- Choose your Math ---
        # Option A: Standard Difference (Recommended)
        residuals = obs_vector - pred_vector

        # Option B: Relative Difference (If you prefer this)
        # safe_obs = obs_vector.copy()
        # safe_obs[np.abs(safe_obs) < 1e-5] = np.nan
        # residuals = (obs_vector - pred_vector) / safe_obs

        data_to_plot = residuals

        # Mask Zeros
        inactive_mask = (np.abs(obs_vector) < 1e-5) & (np.abs(pred_vector) < 1e-5)
        data_to_plot[inactive_mask] = np.nan

        if kind == 'auto' or kind == 'residuals':
            cmap = 'coolwarm'
            limit = np.nanmax(np.abs(data_to_plot)) if vmax is None else vmax
            val_min, val_max = -limit, limit
    else:
        is_residual = False
        data_to_plot[np.abs(data_to_plot) < 1e-5] = np.nan

        if kind == 'auto' or kind == 'standard':
            cmap = 'YlOrRd'
            val_min = 0
            val_max = np.nanmax(data_to_plot) if vmax is None else vmax
        else:
            cmap = 'coolwarm'
            limit = np.nanmax(np.abs(data_to_plot))
            val_min, val_max = -limit, limit

    # --- 2. Map Data to Texture ---
    _, mesh, tex_map = create_aal_volume_and_surface(data_to_plot, region_names)

    # --- 3. Plotting ---
    fig = plt.figure(figsize=(8, 4))

    plot_args = {
        'surf_mesh': mesh.infl_left,
        'roi_map': tex_map,
        'hemi': 'left',
        'bg_map': mesh.sulc_left,
        'bg_on_data': True,
        'darkness': 0.5,
        'threshold': threshold,
        'cmap': cmap,
        'vmin': val_min,
        'vmax': val_max,
        'colorbar': False
    }

    def fix_aspect_and_zoom(ax, view_name):
        ax.set_box_aspect((1.0, 1.25, 0.9))
        # Title for individual views
        ax.set_title(view_name, fontsize=14, y=1.0, pad=-15)

    # 1. Lateral View
    ax1 = plt.subplot(1, 2, 1, projection='3d')
    plotting.plot_surf_roi(**plot_args, view='lateral', axes=ax1, title=None)
    fix_aspect_and_zoom(ax1, "Lateral View")

    # 2. Medial View
    ax2 = plt.subplot(1, 2, 2, projection='3d')
    plotting.plot_surf_roi(**plot_args, view='medial', axes=ax2, title=None)
    fix_aspect_and_zoom(ax2, "Medial View")

    # --- Adjust Layout ---
    # top=0.80 creates space at the top for the main title
    plt.subplots_adjust(wspace=0.0, left=0.05, right=0.88, top=0.80)

    # --- 4. Colorbar ---
    cax = fig.add_axes([0.90, 0.25, 0.02, 0.5])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(val_min, val_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)

    if is_residual:
        cbar.set_label('Difference (Obs - Pred)')
        plt.figtext(0.99, 0.80, "Underestimated\n(Obs > Sim)", ha='right', color='darkred', fontsize=10)
        plt.figtext(0.99, 0.15, "Overestimated\n(Obs < Sim)", ha='right', color='darkblue', fontsize=10)
    else:
        cbar.set_label('Value / Probability')

    # --- 5. Main Title Position ---
    # y=0.90 moves the title down (Default was ~0.98)
    # If it is still too high, try 0.85
    plt.suptitle(title, fontsize=18, y=0.85)

    return fig


# ==========================================
# Example Usage Scenarios
# ==========================================
if __name__ == "__main__":
    from run import X_tau_ad_prob, region_names, M_hist, find_best_model_time_per_subject

    # 1. Observed Mean (Standard Plot)
    obs_mean = np.nanmean(X_tau_ad_prob, axis=0)
    plot_surface_with_cbar(
        obs_mean, region_names,
        title="Observed Tau (Standard)",
        kind='standard',
        threshold=0.05,
        vmax=0.6
    )

    # 2. Residuals (Diverging Plot)
    X_pred = find_best_model_time_per_subject(X_tau_ad_prob, M_hist)
    pred_mean = np.nanmean(X_pred, axis=0)
    residuals = obs_mean - pred_mean

    plot_surface_with_cbar(
        residuals, region_names,
        title="Residuals (Obs - Pred)",
        kind='residuals'
        # No threshold needed for residuals usually
    )


# --- Example Usage ---
if __name__ == "__main__":
    # Assuming you have your calculated 'tau_TPP' variable
    from run import X_tau_ad_prob, region_names

    obs_mean = np.nanmean(X_tau_ad_prob, axis=0)

    # Replace your old call:
    # plot_markers_on_glassbrain_with_cbar(obs_mean, coords)

    # With this new call:
    plot_surface_with_cbar(obs_mean, region_names, title="Observed Pattern (Textured)", threshold=0.05)