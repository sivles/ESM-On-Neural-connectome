# ---------------------------------------------------------------------
# 1. Data loading helpers
# ---------------------------------------------------------------------
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ---- paths ----
root = Path(".")       # run script from folder that contains /data
data_dir = root / "data"
# ---- atlas & SC ----
atlas_path = data_dir / "aal_selected_atlas.xlsx"
dti_dir = data_dir / "DTI" / "CN_neg"


def load_atlas(aal_selected_path: Path = atlas_path) -> List[str]:
    """
    Load selected AAL atlas and return the ordered list of region names.
    Assumes 'AAL Atlas' column and first three rows are header/meta.
    """
    atlas = pd.read_excel(aal_selected_path)
    if "AAL Atlas" not in atlas.columns:
        raise ValueError("Column 'AAL Atlas' not found in atlas file.")

    region_names = atlas["AAL Atlas"].iloc[3:].tolist()
    region_names = [r for r in region_names if isinstance(r, str) and r.strip() != ""]
    print(f"[Atlas] Loaded {len(region_names)} region names.")
    return region_names


# -----------------------------------------------------------
# Atlas loading (names + coordinates)
# -----------------------------------------------------------

def load_atlas_with_coords(aal_path: Path = atlas_path):
    """
    Robust loader for your aal_selected_atlas.xlsx:
    - Finds the first real ROI row automatically (e.g., 'PreCG.L')
    - Accepts unnamed coord columns (e.g., 'Unnamed: 2/3/4') or uses C/D/E by position
    """
    df = pd.read_excel(aal_path, sheet_name=0)

    # 1) Which column contains ROI names?
    name_col = "AAL Atlas" if "AAL Atlas" in df.columns else df.columns[0]

    # 2) Find first ROI row (your ROI names look like 'XXX.L' or 'XXX.R')
    s = df[name_col].astype(str)

    start_idx = 3

    # ROI names
    region_names = df.loc[start_idx:, name_col].astype(str).tolist()

    # 3) Find coordinate columns
    cand_cols = [
        ("x", "y", "z"),
        ("X", "Y", "Z"),
        ("MNI_x", "MNI_y", "MNI_z"),
        ("Unnamed: 2", "Unnamed: 3", "Unnamed: 4"),  # common when Excel headers are blank
        ("Unnamed: 3", "Unnamed: 4", "Unnamed: 5"),  # sometimes shifted
    ]

    coord_cols = None
    for xs, ys, zs in cand_cols:
        if xs in df.columns and ys in df.columns and zs in df.columns:
            coord_cols = (xs, ys, zs)
            break

    if coord_cols is not None:
        coords_df = df.loc[start_idx:, list(coord_cols)]
    else:
        # Fallback: assume coordinates are columns C/D/E -> positions 2,3,4
        # IMPORTANT: pandas uses .iloc for positional indexing
        if df.shape[1] < 5:
            raise ValueError("Atlas file has too few columns to contain C/D/E coordinates.")
        coords_df = df.iloc[start_idx:, 2:5]

    # Convert to numeric
    coords_df = coords_df.apply(pd.to_numeric, errors="coerce")
    coords = coords_df.to_numpy(float)

    # Drop any rows with missing coords
    valid = np.isfinite(coords).all(axis=1)
    region_names = [r for r, v in zip(region_names, valid) if v]
    coords = coords[valid]

    if coords.shape[0] != len(region_names):
        raise ValueError(f"Atlas parsing mismatch: {len(region_names)} names vs {coords.shape[0]} coords.")

    return region_names, coords


def load_dti_group_0(region_names_all: List[str], dti_group_dir: Path = dti_dir):
    """
    Load all subject DTI matrices in a group and compute group-average SC.

    Each .xlsx is assumed to be a pure 86x86 numeric matrix (no headers).
    """
    xlsx_files = sorted(dti_group_dir.glob("sub-*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"No DTI subject files found in {dti_group_dir}")

    mats = []
    for f in xlsx_files:
        df = pd.read_excel(f, header=None)
        A = df.to_numpy(dtype=float)
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"{f} is not square: shape={A.shape}")
        A = 0.5 * (A + A.T)  # enforce symmetry
        mats.append(A)

    mats = np.stack(mats, axis=0)
    W_group = mats.mean(axis=0)

    N = W_group.shape[0]
    if len(region_names_all) < N:
        raise ValueError(
            f"Atlas has only {len(region_names_all)} names but DTI has {N} nodes."
        )
    region_names = region_names_all[:N]
    print(f"[DTI] Group SC shape: {W_group.shape} using first {N} atlas regions.")
    return W_group, region_names


def load_dti_group_sc(dti_dir: Path = dti_dir):
    region_names_all = load_atlas(atlas_path)
    W_group, region_names = load_dti_group_0(region_names_all, dti_dir)

    return W_group, region_names


def load_pet_table(tracer: str, group: str, data_dir: Path = data_dir) -> pd.DataFrame:
    """
    Load PET table for a given tracer ('TAU' or 'AMYLOID') and group.
    """
    tracer_up = tracer.upper()
    if tracer_up == "TAU":
        pet_dir = data_dir / "PET TAU"
    elif tracer_up == "AMYLOID":
        pet_dir = data_dir / "PET AMYLOID"
    else:
        raise ValueError("tracer must be 'TAU' or 'AMYLOID'")

    path = pet_dir / f"{group}.xlsx"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_excel(path)
    print(f"[PET {tracer_up}] Loaded {len(df)} subjects from {path.name}")
    return df


def extract_pet_matrix(df_pet: pd.DataFrame,
                       region_names_subset: List[str]) -> np.ndarray:
    """
    Extract SUVR matrix (subjects x regions) from PET DataFrame.
    """
    missing = [c for c in region_names_subset if c not in df_pet.columns]
    if missing:
        raise ValueError(f"PET table missing columns for regions: {missing[:5]}...")

    X = df_pet[region_names_subset].to_numpy(dtype=float)
    print(f"[PET] Extracted SUVR matrix of shape {X.shape}")
    return X


def group_mean_pet(tracer: str, region_names, groups = ["CN_pos", "MCI_pos", "AD_pos"]):
    X_list = []
    for g in groups:
        try:
            df_g = load_pet_table(tracer=tracer, group=g)
            X_g = extract_pet_matrix(df_g, region_names)
            X_list.append(X_g)
            print(f"{tracer} Added {g}: {X_g.shape[0]} subjects")
        except FileNotFoundError:
            print(f"{tracer} {g} not found, skipping.")

    if not X_list:
        raise RuntimeError(f"No {tracer} PET groups found for continuum TPP.")

    X_all = np.vstack(X_list)
    mean_all = X_all.mean(axis=0)
    return X_all, mean_all

def get_region_idx(region_name_list, all_region_names):
    epic_idx = []
    for name in region_name_list:
        if name in all_region_names:
            epic_idx.append(all_region_names.index(name))

    if not epic_idx:
        epic_idx = [len(all_region_names) - 1]
        print("[Model] No region labels found, falling back to last node.")
    epic_name_list = [all_region_names[i] for i in epic_idx]

    return epic_idx, epic_name_list
