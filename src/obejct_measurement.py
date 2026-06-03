from skimage.measure import regionprops_table
import pandas as pd
import numpy as np

# =====================================================================
# ROI Properties Calculation
# =====================================================================

def measure_objects(
    masks: np.ndarray,
    spot_labels: np.ndarray,
    dx: float,
    dz: float,
    mode: str,
    condition: str,
    source_file: str,
    experiment: str,
    scene: int,
) -> pd.DataFrame:
    """Measure morphological and spatial properties of segmented objects.

    Args:
        masks (np.ndarray): 2D (Y, X) or 3D (Z, Y, X) integer mask array from segmentation.
        spot_labels (np.ndarray): Per-spot mask label assignments from assign_spots_to_masks.
        dx (float): Pixel size in XY in micrometers.
        dz (float): Pixel size in Z in micrometers. Ignored in 2D mode.
        mode (str): Dimensionality of the mask, either '2d' or '3d'.
        condition (str): Experimental condition label, added as metadata column.
        source_file (str): Source file name, added as metadata column.
        scene (int): Scene index within the source file, added as metadata column.

    Returns:
        pd.DataFrame: One row per segmented object with morphological measurements,
            spot counts, and metadata columns.
    """
    assert mode in ("2d", "3d"), "mode must be '2d' or '3d'"
    assert masks.ndim == (3 if mode == "3d" else 2), \
        f"Expected {'3D' if mode == '3d' else '2D'} mask for mode='{mode}'"

    # --- spot counts ---
    spot_counts = (
        np.bincount(spot_labels[spot_labels > 0], minlength=masks.max() + 1)
        if len(spot_labels) > 0
        else np.zeros(masks.max() + 1, dtype=int)
    )

    # --- regionprops_table ---
    props_3d = [
        "label", "area", "bbox", "centroid",
        "equivalent_diameter_area",
    ]
    props_2d = props_3d + ["eccentricity"]  # only available in 2D

    raw = pd.DataFrame(
        regionprops_table(masks, properties=props_3d if mode == "3d" else props_2d)
    )

    # --- build output ---
    df = pd.DataFrame()

    # metadata
    df["Experiment"] = experiment
    df["Source File"] = source_file
    df["Condition"] = condition
    df["Scene"] = scene
    df["Object_Label"] = raw["label"]
    df["Spot_Count"] = spot_counts[raw["label"].to_numpy()]

    if mode == "3d":
        df["Volume_um3"] = (raw["area"] * dz * dx * dx).round(4)
        df["Area_um2"] = np.nan
        df["Spot_Density_per_um3"]  = df["Spot_Count"] / df["Volume_um3"]
        df["Spot_Density_per_um2"]  = np.nan
        df["Z_Span_um"] = ((raw["bbox-3"] - raw["bbox-0"]) * dz).round(4)
        df["Centroid_Z_um"] = (raw["centroid-0"] * dz).round(4)
        df["Centroid_Y_um"] = (raw["centroid-1"] * dx).round(4)  
        df["Centroid_X_um"] = (raw["centroid-2"] * dx).round(4)  
        df["Equivalent_Diameter_um"] = (raw["equivalent_diameter_area"] * dx).round(4)

    else:
        df["Volume_um3"] = np.nan
        df["Area_um2"] = (raw["area"] * dx * dx).round(4)
        df["Spot_Density_per_um3"]  = np.nan
        df["Spot_Density_per_um2"]  = df["Spot_Count"] / df["Area_um2"]          
        df["Z_Span_um"] = np.nan
        df["Centroid_Z_um"] = np.nan
        df["Centroid_Y_um"] = (raw["centroid-0"] * dx).round(4)  
        df["Centroid_X_um"] = (raw["centroid-1"] * dx).round(4)  
        df["Equivalent_Diameter_um"] = (raw["equivalent_diameter_area"] * dx).round(4)
        
    return df