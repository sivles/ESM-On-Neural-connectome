# ---------------------------------------------------------------------
# 4. Correlation & plotting
# ---------------------------------------------------------------------
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture
import matplotlib.patches as patches


def corr_with_pet(model_map: np.ndarray, pet_map: np.ndarray):
    """Pearson correlation between model vector and PET vector."""
    r, p = stats.pearsonr(model_map, pet_map)
    return r, p


# ---------------------------------------------------------------------
# 2. TPP (tau-positive probability) via 2-component GMM
# ---------------------------------------------------------------------


def compute_prob_per_region(X: np.ndarray,
                            n_components: int = 2,
                            random_state: int = 0):
    """
    Compute TPP (tau-positive probability) for each region.

    X : (n_subjects, N) PET SUVR matrix (tau).
    Returns
    -------
    TPP : (n_subjects, N) posterior P(tau-positive) per subject/region.
    tpp_mean : (N,)      average TPP per region (tau presence map).
    """
    n_subj, n_regions = X.shape
    TPP = np.zeros_like(X, dtype=float)

    for r in range(n_regions):
        y = X[:, r].reshape(-1, 1)

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state
        )
        gmm.fit(y)
        means = gmm.means_.flatten()
        pos_comp = np.argmax(means)  # higher-mean = tau-positive

        post = gmm.predict_proba(y)
        TPP[:, r] = post[:, pos_comp]

    tpp_mean = TPP.mean(axis=0)
    print("[TPP] Computed tau-positive probabilities.")
    return TPP, tpp_mean


# ---------------------------------------------------------------------
# 2. TPP (tau-positive probability) via 2-component GMM
# ---------------------------------------------------------------------


def compute_prob_per_region2(X_all: np.ndarray,
                            X: np.ndarray,
                            n_components: int = 2,
                            random_state: int = 0):
    """
    Compute TPP (tau-positive probability) for each region.

    X : (n_subjects, N) PET SUVR matrix (tau).
    Returns
    -------
    TPP : (n_subjects, N) posterior P(tau-positive) per subject/region.
    tpp_mean : (N,)      average TPP per region (tau presence map).
    """
    n_subj, n_regions = X.shape
    TPP = np.zeros_like(X, dtype=float)

    for r in range(n_regions):
        y_all = X_all[:, r].reshape(-1, 1)

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=random_state
        )
        gmm.fit(y_all)
        means = gmm.means_.flatten()
        pos_comp = np.argmax(means)  # higher-mean = tau-positive

        y = X[:, r].reshape(-1, 1)
        post = gmm.predict_proba(y)
        TPP[:, r] = post[:, pos_comp]

    tpp_mean = TPP.mean(axis=0)
    print("[TPP] Computed tau-positive probabilities.")
    return TPP, tpp_mean


def find_best_model_time_per_subject(X_obs, M_hist):
    """
    For each subject (row in X_obs), find the time step 't' in the simulation
    (M_hist) that has the highest correlation with that subject's pattern.
    Returns a matrix X_pred of shape (n_subjects, n_regions).
    """
    n_subj, n_roi = X_obs.shape
    X_pred = np.zeros_like(X_obs)

    # Pre-normalize simulation history for faster correlation
    # M_hist shape: (Time, Regions)
    M_centered = M_hist - M_hist.mean(axis=1, keepdims=True)
    # Avoid div by zero
    M_std = np.sqrt((M_centered ** 2).sum(axis=1, keepdims=True)) + 1e-12
    M_norm = M_centered / M_std

    for i in range(n_subj):
        row = X_obs[i]

        # Skip empty/NaN rows
        if np.isnan(row).all():
            continue

        # Normalize subject row
        y_centered = row - np.nanmean(row)
        y_std = np.sqrt(np.nansum(y_centered ** 2)) + 1e-12
        y_norm = y_centered / y_std

        # Calculate correlation with every time point in simulation
        # Dot product of normalized vectors = Cosine similarity ~ Correlation
        corrs = M_norm @ y_norm.T

        # Find best time point
        best_t = np.argmax(corrs)
        X_pred[i, :] = M_hist[best_t, :]

    return X_pred


def draw_gradient_circle(ax, center_x, center_y, radius, intensity):
    """
    Draws a circle with a vertical gradient fill to mimic the 3D/glossy look.
    intensity: 0.0 (White) to 1.0 (Dark Red)
    """
    # 1. Define Colors
    # We interpolate between White and a Deep Red based on intensity
    base_red = np.array([139, 0, 0]) / 255.0  # Dark Red (RGB)
    white = np.array([1.0, 1.0, 1.0])

    # The 'target' color for the bottom of this specific circle
    # If intensity is 0, target is white. If 1, target is dark red.
    target_color = (1 - intensity) * white + intensity * base_red

    # 2. Create Gradient Data
    # We create a small N x N grid for the gradient
    N = 100
    # Vertical gradient: 0 at top, 1 at bottom
    gradient = np.linspace(0, 3, N).reshape(N, 1)
    gradient = np.tile(gradient, (1, N))

    # Map the gradient to colors (White -> Target Color)
    # This gives the "glossy" look where the light hits the top
    img_data = np.zeros((N, N, 4))  # RGBA
    for i in range(3):  # RGB channels
        # Start at White (top), fade to target_color (bottom)
        img_data[:, :, i] = 1.0 * (1 - gradient) + target_color[i] * gradient

    img_data[:, :, 3] = 1.0  # Alpha

    # 3. Mask the Gradient into a Circle
    y, x = np.ogrid[:N, :N]
    center = N / 2
    dist_from_center = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = dist_from_center > center
    img_data[mask] = [1, 1, 1, 0]  # Make outside transparent

    # 4. Display the Image
    # We figure out the extent to place it correctly on the plot
    extent = [center_x - radius, center_x + radius, center_y - radius, center_y + radius]
    ax.imshow(img_data, extent=extent, zorder=1)

    # 5. Add the Maroon Outline (Stroke)
    circle = patches.Circle(
        (center_x, center_y), radius=radius,
        edgecolor='#8B0000',  # Dark Red border
        facecolor='none',  # Transparent fill (we used the image for fill)
        linewidth=2.5,
        zorder=2
    )
    ax.add_patch(circle)