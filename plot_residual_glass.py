# brain_vis.py
from pathlib import Path
from typing import Sequence, Tuple, Optional
from nilearn import plotting, datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import imageio.v2 as imageio

from load import load_atlas_with_coords


# -----------------------------------------------------------
# 2. PET loading using your existing helpers
# -----------------------------------------------------------

def group_tag_to_pet_group(tag: str) -> str:
    """
    Map BRAPH-style tags to your Excel group names.
    """
    tag = tag.upper()
    if tag == "CNn".upper():
        return "CN_neg"
    if tag == "CNp".upper():
        return "CN_pos"
    if tag == "MCIp".upper():
        return "MCI_pos"
    if tag == "ADp".upper():
        return "AD_pos"
    raise ValueError(f"Unknown group tag: {tag}")


def load_pet_matrix_for_group(
    data_dir: Path,
    tracer: str,
    pet_group_tag: str,
    region_names: Sequence[str],
) -> np.ndarray:
    """
    Use the same logic as your load_pet_table + extract_pet_matrix.
    """
    tracer_up = tracer.upper()
    if tracer_up == "TAU":
        pet_dir = data_dir / "PET TAU"
    elif tracer_up == "AMYLOID":
        pet_dir = data_dir / "PET AMYLOID"
    else:
        raise ValueError("tracer must be 'TAU' or 'AMYLOID'")

    group_name = group_tag_to_pet_group(pet_group_tag)
    path = pet_dir / f"{group_name}.xlsx"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_excel(path)

    # same as extract_pet_matrix
    missing = [c for c in region_names if c not in df.columns]
    if missing:
        raise ValueError(f"PET table missing columns for regions: {missing[:5]}...")

    X = df[region_names].to_numpy(float)  # (n_subj, N)
    return X


# -----------------------------------------------------------
# 3. 2D brain projections + scatter plotting
# -----------------------------------------------------------

def _project_coords(coords: np.ndarray, view: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple 2D projections for 3 standard views.
    view:
      'AD' -> anterior-dorsal (x vs y)
      'CA' -> coronal-axial (y vs z)
      'SL' -> sagittal-lateral (x vs z)
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    view = view.upper()
    if view == "AD":
        return x, y
    if view == "CA":
        return y, z
    if view == "SL":
        return x, z
    raise ValueError(f"Unknown view: {view}")


def plot_brain_values(
    coords: np.ndarray,
    values: np.ndarray,
    view: str,
    cutoff: float,
    factor: float,
    color: Tuple[float, float, float],
    title: str,
    out_path: Optional[Path] = None,
):
    """
    Make a 2D scatter plot of brain regions for a given view.

    values > cutoff are scaled by 'factor' in size; others kept tiny.
    """
    if values.shape[0] != coords.shape[0]:
        raise ValueError("values and coords must have same length")

    # sphere "radii" (actually marker sizes)
    radii = np.ones_like(values) * 10.0  # baseline
    mask = values > cutoff
    radii[mask] = radii[mask] * factor

    Xv, Yv = _project_coords(coords, view)

    plt.figure(figsize=(5, 5))
    plt.scatter(
        Xv,
        Yv,
        s=radii,
        c=[color],
        alpha=0.9,
        edgecolors="k",
        linewidths=0.2,
    )
    plt.gca().set_aspect("equal", "box")
    plt.axis("off")
    plt.title(f"{title} ({view} view)")
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300)
    plt.close()


# -----------------------------------------------------------
# 4. High-level function: visualize PET group (like MATLAB)
# -----------------------------------------------------------

def visualize_pet_group_py(
    atlas_path: Path,
    data_dir: Path,
    modality: str,        # 'Tau' or 'Amyloid'
    group_tag: str,       # 'CNn','CNp','MCIp','ADp'
    cutoff: float,
    factor: float,
    facecolor=(0.52, 0.08, 0.08),
    out_dir: Optional[Path] = None,
):
    """
    Python version of visualize_pet_group from the MATLAB script.

    It will:
      1) load atlas names + coords
      2) load PET matrix for the given modality/group
      3) average SUVR over subjects
      4) make 3 PNGs: *_AD.png, *_CA.png, *_SL.png
    """
    atlas_path = Path(atlas_path)
    data_dir = Path(data_dir)
    if out_dir is None:
        out_dir = data_dir  # default: put images in data root dir
    out_dir = Path(out_dir)

    region_names, coords = load_atlas_with_coords(atlas_path)
    X = load_pet_matrix_for_group(data_dir, modality, group_tag, region_names)
    avg_suvr = X.mean(axis=0)

    tag = group_tag  # just reuse e.g. ADp
    title = f"{modality} {group_tag}"

    for view in ["SL", "AD", "CA"]:
        out_name = f"{modality}_{tag}_{view}.png"
        out_path = out_dir / out_name
        plot_brain_values(
            coords,
            avg_suvr,
            view=view,
            cutoff=cutoff,
            factor=factor,
            color=facecolor,
            title=title,
            out_path=out_path,
        )
        print(f"[viz] Wrote {out_path}")


# -----------------------------------------------------------
# 5. GIF creation from PNGs
# -----------------------------------------------------------

def make_gif_from_frames(
    frame_paths: Sequence[Path],
    out_gif: Path,
    delay: float = 0.6,
    ping_pong: bool = False,
):
    """
    Build an animated GIF from a list of image paths.
    """
    paths = [Path(p) for p in frame_paths]
    if ping_pong and len(paths) > 2:
        paths = list(paths) + list(reversed(paths[1:-1]))

    imgs = []
    for p in paths:
        if not p.exists():
            print(f"[gif] Missing frame (skipped): {p}")
            continue
        imgs.append(imageio.imread(p))

    if not imgs:
        raise RuntimeError("No valid frames for GIF.")

    imageio.mimsave(out_gif, imgs, duration=delay)
    print(f"[gif] Wrote {out_gif}")


def build_modality_gifs(
    out_dir: Path,
    modality: str,
    views=("AD", "CA", "SL"),
    states=("CNn", "CNp", "MCIp", "ADp"),
    delay: float = 0.6,
    ping_pong: bool = False,
):
    """
    Reproduce the MATLAB loop that creates GIFs across states for each view.
    Assumes PNGs like Tau_ADp_AD.png etc. already exist in out_dir.
    """
    out_dir = Path(out_dir)
    for view in views:
        frame_paths = [
            out_dir / f"{modality}_{state}_{view}.png" for state in states
        ]
        out_gif = out_dir / f"{modality}_{view}.gif"
        make_gif_from_frames(frame_paths, out_gif, delay=delay, ping_pong=ping_pong)


def plot_brain_colormap(
    coords: np.ndarray,
    values: np.ndarray,
    view: str,
    title: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "inferno",
    out_path: Optional[Path] = None,
    s: float = 40.0,
):
    """
    2D projection scatter where color encodes values.
    views: 'SL' (x vs z), 'AD' (x vs y), 'CA' (y vs z)
    """
    if values.shape[0] != coords.shape[0]:
        raise ValueError("values and coords must have same length")

    # projection (same logic as your _project_coords)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    view = view.upper()
    if view == "AD":
        Xv, Yv = x, y
    elif view == "CA":
        Xv, Yv = y, z
    elif view == "SL":
        Xv, Yv = x, z
    else:
        raise ValueError(f"Unknown view: {view}")

    plt.figure(figsize=(5, 5))
    sc = plt.scatter(Xv, Yv, c=values, s=s, cmap=cmap, vmin=vmin, vmax=vmax,
                     edgecolors="k", linewidths=0.2)
    plt.gca().set_aspect("equal", "box")
    plt.axis("off")
    plt.title(f"{title} ({view})")
    plt.colorbar(sc, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def plot_markers_on_glassbrain(coords, values, title,
                               cmap="viridis", vmin=None, vmax=None,
                               marker_size=60, output_file=None):
    coords = np.asarray(coords, float)
    values = np.asarray(values, float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N,3), got {coords.shape}")
    if values.ndim != 1 or values.shape[0] != coords.shape[0]:
        raise ValueError(f"values must be (N,), got {values.shape}, coords {coords.shape}")

    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))

    display = plotting.plot_markers(
        values,
        coords,
        title=title,
        node_size=marker_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        colorbar=True,
        display_mode="lyrz",
        output_file=output_file,   # set path to save directly, or None to show
    )

    # If not saving to file, show the matplotlib window
    if output_file is None:
        plt.show()

    return display

def plot_markers_on_glassbrain_with_cbar(
    coords,
    values,
    title,
    cmap="viridis",
    vmin=None,
    vmax=None,
    marker_size=60,
    cbar_label=None,
    output_file=None,
    display_mode="lr",
):
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must be (N,3), got {coords.shape}")
    if values.ndim != 1 or values.shape[0] != coords.shape[0]:
        raise ValueError(f"values must be (N,), got {values.shape}, coords {coords.shape}")

    if vmin is None:
        vmin = float(np.nanmin(values))
    if vmax is None:
        vmax = float(np.nanmax(values))

    # Map values -> RGBA using matplotlib (version-robust; avoids nilearn cmap kwargs)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = cm.get_cmap(cmap)
    rgba = cmap_obj(norm(values))

    # Create glass brain
    display = plotting.plot_glass_brain(
        None,
        display_mode=display_mode,
        title=title,
        colorbar=False,  # we add our own stable colorbar
    )

    # Add colored markers
    display.add_markers(
        marker_coords=coords.tolist(),
        marker_color=rgba,
        marker_size=marker_size,
    )

    # Get the underlying matplotlib figure
    fig = display.frame_axes.figure

    # --- Create a horizontal colorbar axis under the brain panels ---
    # Make room at the bottom for the colorbar
    # (Nilearn uses custom axes; we add our own axis explicitly)
    cax = fig.add_axes([0.35, 0.08, 0.30, 0.04])  # [left, bottom, width, height]

    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    if cbar_label:
        cbar.set_label(cbar_label)

    # Optional: nicer tick label size
    cbar.ax.tick_params(labelsize=10)

    if output_file is not None:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return display

