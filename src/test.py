# 
# from pathlib import Path
# import re
# from skimage.measure import block_reduce
# import pandas as pd
# import numpy as np
# from cellpose import models, utils
# from spotiflow.model import Spotiflow


##seg_det
import numpy as np
from skimage.measure import block_reduce
from cellpose import models, utils
from spotiflow.model import Spotiflow
from typing import Tuple
from types import SimpleNamespace

##object mewasurement
from skimage.measure import regionprops_table
import pandas as pd
#import numpy as np

##io_utils
import re

##qc figures
# from pathlib import Path
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde

from typing import Optional

##config
from pathlib import Path

## main loop
from bioio import BioImage


# =====================================================================
# 1. Functions
# =====================================================================
# %% Segementation
def segment_2d(
    bf_stack: np.ndarray,
    model_cellpose: models.CellposeModel,
    factor: int = 4,
) -> np.ndarray:
    """Run BAG-pretrained Cellpose-SAM in 2D on stdev projection of stack, using image downscaling for faster processing.

    Args:
        bf_stack (np.ndarray): Input 3D BAG image for segmentation.
        model_cellpose (models.CellposeModel): Initiated Cellpose model.
        factor (int): Downscaling factor. Defaults to 4.

    Returns:
        np.ndarray: Segmentation masks, not touching the edges.
    """
    if bf_stack.ndim == 3 and bf_stack.shape[0] > 1:
        std_proj = bf_stack.std(axis=0).astype(np.float32)
    else:
        std_proj = np.squeeze(bf_stack).astype(np.float32)

    img_binned = block_reduce(std_proj, block_size=(factor, factor), func=np.mean) # type: ignore[arg-type]

    masks, _, _ = model_cellpose.eval(img_binned)

    masks_resized = masks.repeat(factor, axis=-2).repeat(factor, axis=-1)
    masks_cleaned = utils.remove_edge_masks(masks_resized, change_index=True)

    return masks_cleaned

def segment_3d(
    bf_stack: np.ndarray,
    model_cellpose: models.CellposeModel,
    factor: int = 4,
    stitch_threshold: float = 0.4,
) -> np.ndarray:
    """Run BAG-pretrained Cellpose-SAM in pseudo-3D (segementing individual planes and stiching them together) on minimal projection substraced stack, using image downscaling for faster processing.

    Args:
        bf_stack (np.ndarray): Input 3D BAG image for segmentation.
        model_cellpose (models.CellposeModel): Initiated Cellpose model.
        factor (int): Downscaling factor. Defaults to 4.
        stitch_threshold (float): Treshold for stiching segmented planes. Defaults to 0.4.

    Returns:
        np.ndarray: Segmentation masks, not touching the edges.
    """
    min_substracted = bf_stack.astype(np.float32) - np.min(bf_stack, axis=0).astype(np.float32)
    img_binned = block_reduce(min_substracted, block_size=(1,factor, factor), func=np.mean) # type: ignore[arg-type]
    
    masks, _, _ = model_cellpose.eval(
        img_binned,
        do_3D=False,
        z_axis=0,
        stitch_threshold=stitch_threshold
        )
    
    masks_resized = masks.repeat(factor, axis=-2).repeat(factor, axis=-1)
    masks_cleaned = np.zeros_like(masks_resized)
    for z in range(masks_resized.shape[0]):
        masks_cleaned[z] = utils.remove_edge_masks(masks_resized[z], change_index=True)
    
    return masks_cleaned

# %% Spot detection and assigment to masks
def detect_spots_spotiflow(
    spot_stack: np.ndarray,
    model_spotiflow: Spotiflow,
    prob_thresh: float,
    min_distance: int,
    ) -> Tuple[np.ndarray, SimpleNamespace]:
    """Detect spot-like features in an image using the default Spotiflow model.

    Args:
        image (np.ndarray): Input spot image as a 2D.
        min_distance (int): Minimum distance between detected spots. Defaults to 10.

    Returns:
        Tuple[np.ndarray, SimpleNamespace]: 
            - points (np.ndarray): Array of detected spot coordinates.
            - details (SimpleNamespace): List of metadata dictionaries for each spot, including confidence scores and other attributes.
    """
    points, details = model_spotiflow.predict(
        img=spot_stack,
        verbose=False,
        prob_thresh=prob_thresh,
        min_distance=min_distance
    )
    
    return points, details

def assign_spots_to_mask(
    coordinates: np.ndarray | list, 
    masks: np.ndarray
    ) -> np.ndarray:
    """Assigns coordinates (spots) to masks (objects), for 2D and 3D outputs.

    Args:
        coordinates (np.ndarray | list): Coordinates of detected spots.
        masks (np.ndarray): Numpy array of object masks from segmentation.
        
    Raises:
        ValueError: If coordinates and masks dimentions are mismatched.

    Returns:
        np.ndarray: A list of coordinates that falls within the non-zero masks, with object IDs.
    """
    if len(coordinates) == 0:
        return np.array([], dtype=int)
    
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    
    ndim = coordinates.shape[1]
    
    if ndim == 2 and masks.ndim == 2:
        yi = np.clip(np.round(coordinates[:, 0]).astype(int), 0, masks.shape[0] - 1)
        xi = np.clip(np.round(coordinates[:, 1]).astype(int), 0, masks.shape[1] - 1)
        return masks[yi, xi]
    
    elif ndim == 3 and masks.ndim == 3:
        zi = np.clip(np.round(coordinates[:, 0]).astype(int), 0, masks.shape[0] - 1)
        yi = np.clip(np.round(coordinates[:, 1]).astype(int), 0, masks.shape[1] - 1)
        xi = np.clip(np.round(coordinates[:, 2]).astype(int), 0, masks.shape[2] - 1)
        return masks[zi, yi, xi]
    
    else:
        raise ValueError(
            f"Mismatch: coords have {ndim} dims but masks have {masks.ndim} dims."
        )

# %% object measurement
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
    props_2d = props_3d + ["eccentricity"]

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


# %% io utils
def parse_condition_from_name(filename_stem: str) -> str:
    """Extracts the base condition name from a filename stem.
    
    Examples:
        'Control_01'         -> 'Control'
        'Treated-DrugA_FOV3' -> 'Treated-DrugA'
        'WT_high_res'        -> 'WT_high_res'
    """
    # Regex splits at the last occurrence of an underscore or dash followed by digits
    # Adjust this regex if your naming convention uses a different delimiter
    match = re.split(r'[-_]\d+$', filename_stem)
    return match[0]

# %% qc figures
def make_qc_figure(
    coordinates: np.ndarray,
    condition: str,
    scene: int,
    mode: str,
    out_path: Path,
    segmentation_image: np.ndarray,
    spots_image: np.ndarray,
    masks: np.ndarray,
    flow_details,
    spot_labels: np.ndarray,
    dx: float,
    dz: float,
    config: dict
) -> None:
    """
    6-panel QC figure per scene (2 rows × 3 cols):

    Row 1: BF stdev proj + mask overlays | SPot stdev-proj + detections | Spotiflow flows
    Row 2: Spots per z-slice OR spot nearest neighbourg distance       | SNR        | XY spot map (z-colored)
    
    Args:
        df (pd.DataFrame): Combined scene dataframe of 2d or 3d analysis.
        condition (str): Experimental condition name.
        mode (str): 2d or 3d mode.
        out_path (Path): Output path.
    """
    is_3d = mode == "3d"
    has_spots = len(coordinates) > 0
    n_obj = len(np.unique(masks)) - 1
    
    # getting extra variables
    
    ## bf processing
    if segmentation_image.ndim == 3 and segmentation_image.shape[0] > 1:
        seg_stdev = np.std(segmentation_image, axis=0).astype(np.float32)
    else:
        seg_stdev = np.squeeze(segmentation_image).astype(np.float32)    
    
    seg_stdev_inv = seg_stdev.max() - seg_stdev
    
    seg_inv_norm = (seg_stdev_inv - seg_stdev_inv.min()) / (np.ptp(seg_stdev_inv) + 1e-8)
    
    ## spot processing
    if spots_image.ndim == 3 and spots_image.shape[0] > 1:
        spots_stdev = np.std(spots_image, axis=0).astype(np.float32)
    else:
        spots_stdev = np.squeeze(spots_image).astype(np.float32)      
    
    spots_stdev_norm = (spots_stdev - spots_stdev.min()) / (np.ptp(spots_stdev) + 1e-8)
    
    ## masks
    masks_2d = np.max(masks, axis=0) if (masks is not None and masks.ndim == 3) else masks
    
    ## spots
    if has_spots:
        if is_3d:
            spots_z =  np.round(coordinates[:,0]).astype(int)
            spots_y =  np.round(coordinates[:,1]).astype(int)
            spots_x =  np.round(coordinates[:,2]).astype(int)
            
            spots_z_um = spots_z * dz
            spots_y_um = spots_y * dx
            spots_x_um = spots_x * dx            
            
        else:
            spots_z = np.zeros(len(coordinates))
            spots_y = np.round(coordinates[:,0]).astype(int)
            spots_x = np.round(coordinates[:,1]).astype(int)
            
            spots_z_um = spots_z
            spots_y_um = spots_y * dx
            spots_x_um = spots_x * dx            
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()
    fig.suptitle(
        f"{condition} - Scene {scene:02d} - [{mode.upper()} mode] - "
        f"{n_obj} object(s) - {len(coordinates)} spot(s)",
        fontsize=12, fontweight="bold")    

# %% Panel A 
    # Panel A - Inverted stack stdev + outlines
    ax_a = axes_flat[0]
    ax_a.imshow(seg_inv_norm, cmap="gray")
    if masks_2d is not None:
        mask_overlay = np.ma.masked_where(masks_2d == 0, masks_2d)
        ax_a.imshow(mask_overlay, alpha = 0.3, cmap='tab10', vmin=1)
    ax_a.set_title("StDev Projection + Masks")
    ax_a.axis("off")

# %% Panel B
    # Panel B - Spots stdev + spots (coloured by object)
    ax_b = axes_flat[1]
    ax_b.imshow(spots_stdev_norm, cmap="magma")
    if has_spots and spots_x is not None and spots_y is not None: # type: ignore
        inside_objects = spot_labels > 0
        # background elements
        ax_b.scatter(
            spots_x[~inside_objects], spots_y[~inside_objects],  # type: ignore
            color="white", alpha=0.3, s=6, marker="x"
        )
        # assigned spots
        ax_b.scatter(
            spots_x[inside_objects], spots_y[inside_objects],  # type: ignore
            c=spot_labels[inside_objects], cmap="tab10",
            s=15, edgecolors='white', linewidths=0.3, alpha=0.9
        )
    ax_b.set_title("Spot Detections (StDev Proj)")
    ax_b.axis("off")

# %% Panel C
    # Panel C - Spotiflow flows
    ax_c = axes_flat[2]
    if flow_details is not None:
        try:
            flow_data = getattr(flow_details, 'flow', flow_details)
            
            if isinstance(flow_data, np.ndarray):
                if flow_data.ndim == 4 and flow_data.shape[0] == 4:
                    flow_proj = np.max(flow_data, axis=1)
                    rgb_flow = flow_proj[:3, :, :]
                    rgb_flow = np.moveaxis(rgb_flow, 0, -1)
                    
                    # Scale the vector weights uniformly between [0, 1]
                    flow_viz = 0.5 * (1 + rgb_flow)
                    ax_c.imshow(np.clip(flow_viz, 0, 1))
                
                elif flow_data.ndim == 3 and flow_data.shape[0] == 3:
                    rgb_flow = np.moveaxis(flow_data, 0, -1)
                    flow_viz = 0.5 * (1 + rgb_flow)
                    ax_c.imshow(np.clip(flow_viz, 0, 1))
                
                else:
                    flat_view = np.max(flow_data, axis=0) if flow_data.ndim >= 3 else flow_data
                    ax_c.imshow(flat_view, cmap="viridis")
                    
            else:
                ax_c.text(0.5, 0.5, "Non-Array Flow Format", 
                        ha = "center", va = "center",
                        transform=ax_c.transAxes, 
                        fontsize=11, color='gray')
        
        except Exception as e:
            ax_c.text(0.5, 0.5, f"Flow Render Error\n{str(e)}", 
                        ha = "center", va = "center",
                        transform=ax_c.transAxes, 
                        fontsize=11, color='gray')
    
    else:
        ax_c.text(0.5, 0.5, "No Flow Data Provided", 
                        ha = "center", va = "center",
                        transform=ax_c.transAxes, 
                        fontsize=11, color='gray')

    ax_c.set_title('Stereographic Flow')
    ax_c.axis("off")

# %% Panel D
    # Panel D - Spots per z slice | Spot nearest neighbour distance
    ax_d = axes_flat[3]
    if not has_spots:
        ax_d.text(0.5, 0.5, "No Spots Detected",
                ha="center", va="center", color="gray")
        ax_d.axis("off")
    elif is_3d:
        # Spots per z slice stacked histogram
        unique_labels = np.unique(spot_labels)
        hist_data = []
        colors = []
        labels = []
        cmap_colors = plt.cm.tab10.colors
        
        for lbl in unique_labels:
            mask_label = (spot_labels == lbl)
            if lbl == 0:
                labels.append("Background")
                colors.append("lightgray")
            else:
                labels.append(f"Obj {lbl}")
                colors.append(cmap_colors[(lbl - 1) % len(cmap_colors)])               
            hist_data.append(spots_z_um[mask_label]) # type: ignore
        
        total_z_planes = spots_image.shape[0] if segmentation_image.ndim == 3 else 1
        bin_edges = np.arange(total_z_planes + 1) * dz
        
        ax_d.hist(hist_data, bins=bin_edges, stacked=True,
                color=colors, label=labels, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax_d.set_title("Z-Distribution Profile (µm)")
        ax_d.set_xlabel("Z-Depth Position (µm)")
        ax_d.set_ylabel("Spot Count")
        ax_d.grid(True, linestyle=":", alpha=0.5)
        if n_obj < 10:
            ax_d.legend(fontsize=8, loc="upper right")
    else:
        # Spot nearest neighbour distance
        if len(coordinates) < 2:
            ax_d.text(0.5, 0.5, "Insufficient Spots for NND",
                    ha="center", va="center", coor="gray")
            ax_d.grid("off")
        else:
            spatial_xy_um = np.column_stack((spots_x_um, spots_y_um)) # type: ignore
            tree = KDTree(spatial_xy_um)
            distances, _ = tree.query(spatial_xy_um, k=2)
            nnd_um = distances[:,1]
            
            ax_d.hist(nnd_um, bins="auto", density=True,
                    color="#FF66CC", alpha=0.4, edgecolor="#FF66CC")
            try:
                kde = gaussian_kde(nnd_um)
                x_vals = np.linspace(nnd_um.min(), nnd_um.max(), 200)
                ax_d.plot(x_vals, kde(x_vals), color="#FF66CC", linewidth=1.5)
                
            except Exception:
                pass
        ax_d.set_title("Spot Proximity Distribution")
        ax_d.set_xlabel("Nearest Neighbor Distance (µm)")
        ax_d.set_ylabel("Density")
        ax_d.grid(True, linestyle=":", alpha=0.5)

# %% Panel E
    # Panel E - Spotiflow probability score ECDF (inside vs background)
    ax_e = axes_flat[4]
    if not has_spots:
        ax_e.text(0.5, 0.5, "No Spots Detected",
                ha="center", va="center", coor="gray")
        ax_e.axis("off")
    else:
        prob_arr = np.array(flow_details.prob)
        inside = spot_labels > 0
        
        for mask, label, color, ls in [
            (inside,  "Inside object", "#D4537E", "-"),
            (~inside, "Background",    "#888780", "--"),
        ]:
            subset = np.sort(prob_arr[mask])
            if len(subset) == 0:
                continue
            ecdf_y = np.arange(1, len(subset) + 1) / len(subset)
            ax_e.step(subset, ecdf_y, where="pre",
                    color=color, linestyle=ls, linewidth=1.5,
                    label=f"{label} (n={len(subset)})")
        ax_e.axvline(config["prob_thresh"], color="gray",
                linewidth=0.8, linestyle=":", alpha=0.6,
                label=f"thresh={config['prob_thresh']}")
        ax_e.set_xlim(0,1)
        ax_e.set_ylim(0,1)
        ax_e.set_xlabel("Spotiflow probability score")
        ax_e.set_ylabel("Cumulative fraction of spots")
        ax_e.set_title("Detection Confidence (ECDF)")
        ax_e.legend(fontsize=8, loc="upper left")
        ax_e.grid(True, linestyle=":", alpha=0.4)
    
# %% Panel F
    # Panel F - Outlines + xy spotmap (coloured by z depth)
    ax_f = axes_flat[5]
    if masks_2d is not None:
        ax_f.contour(masks_2d, levels=np.unique(masks_2d)[1:] - 0.5,
                    colors="white", linewidths=0.5, alpha=0.6)
    if has_spots and spots_x is not None and spots_y is not None: # type: ignore
        if is_3d:
            sc = ax_f.scatter(spots_x, spots_y, c=spots_z_um, cmap="turbo", # type: ignore
                            s=12, edgecolors="black", linewidths=0.15, alpha=0.85)
            cbar = fig.colorbar(sc, ax=ax_f, orientation="vertical",
                                pad=0.02, shrink=0.7)
            cbar.ax.tick_params(labelsize=8)
        
        else:
            ax_f.scatter(spots_x, spots_y, color="#00FFCC", s=10, alpha=0.7) # type: ignore
    
    ax_f.set_xlim(0, segmentation_image.shape[-1])
    ax_f.set_ylim(segmentation_image.shape[-2], 0)
    ax_f.set_title("XY Spatial Spotmap (µm)")
    ax_f.axis("off")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [QC] Saved summary: {out_path.name}")

# %% Summary Figures
def make_scene_summary_figure(
    df: pd.DataFrame,
    condition: str,
    mode: str,
    out_path: Path
) -> None:
    """Generates a 4-panel quality control summary figure for a scene.

    Args:
        df (pd.DataFrame): Combined scene dataframe of 2d or 3d analysis.
        condition (str): Experimental condition name.
        mode (str): 2d or 3d mode.
        out_path (Path): Output path.
    """
    is_3d = mode == "3d"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    fig.suptitle(f"{condition}  —  Summary [{mode.upper()} mode]", fontsize=13, fontweight="bold")
    
    # Panel A - Spot per object per scene (Swarm plot per scene)
    ax_a = axes_flat[0]
    sns.swarmplot(
        data=df, x="Spot_Count", y="Scene",
        size=4, color=".3", ax=ax_a
    )
    ax_a.set_title("Spot Count per Object per Scene")
    ax_a.set_xlabel("Spot Count")
    ax_a.set_ylabel("Scene")
    
    # Panel B - Object size distribution per scene violin
    ax_b = axes_flat[1]
    size_metric = "Volume_um3" if is_3d else "Area_um2"
    size_label = "Volume (µm³)" if is_3d else "Area (µm²)"
    sns.boxplot(
        data=df, x="Scene", y=size_metric,
        whis=(0, 100), width=.6, ax=ax_b, palette="vlag"
    )
    sns.stripplot(data=df, x="Scene", y=size_metric, 
                size=4, color=".3", ax=ax_b)
    ax_b.set_title(f"Object Size Distribution ({size_label})")
    ax_b.set_xlabel(f"{size_label}")
    ax_b.set_ylabel("Scene") 
        
    # Panel C - Spots per object histogram (pooled across scene)
    ax_c = axes_flat[2]
    sns.histplot(
        data=df, x="Spot_Count", ax=ax_c
    )
    ax_c.set_title("Pooled Spots per Object Distribution")
    ax_c.set_xlabel("Spots per Object")
    ax_c.set_ylabel("Count")    
    
    # Panel D - Object size vs spot count (coloured by scene)
    ax_d = axes_flat[3]
    size_metric = "Volume_um3" if is_3d else "Area_um2"
    size_label = "Volume (µm³)" if is_3d else "Area (µm²)"
    sns.scatterplot(
        data=df, x=size_metric, y="Spot_Count", 
        hue="Scene", alpha=0.7, ax=ax_d
    )
    ax_d.set_title("Object Size vs Spot Count per Scene")
    ax_d.set_xlabel(f"{size_label}")
    ax_d.set_ylabel("Spot Count")
    
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [QC] Saved scene summary: {out_path.name}")    

def make_run_summary_figure(
    df: pd.DataFrame,
    experiment: str,
    mode: str,
    out_path: Path
) -> None:
    """Generates a 4-panel quality control summary figure for an experiment.

    Args:
        df (pd.DataFrame): Combined run dataframe of 2d or 3d analysis.
        experiment: Run/Experiment name.
        mode (str): 2d or 3d mode.
        out_path (Path): Output path.
    """
    is_3d = mode == "3d"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    fig.suptitle(f"{experiment}  —  Summary [{mode.upper()} mode]", fontsize=13, fontweight="bold")
    
    # Panel A - Spot per condition
    ax_a = axes_flat[0]
    sns.boxplot(
        data=df, x="Spot_Count" , y="Condition",
        whis=(0, 100), width=.6, ax=ax_a, palette="vlag"
    )
    sns.stripplot(data=df, x="Spot_Count" , y="Condition", 
                size=4, color=".3", ax=ax_a)
    ax_a.set_title("Spot Count per Condition")
    ax_a.set_xlabel("Spot Count")
    
    # Panel B - Spot per area/volume per condition
    ax_b = axes_flat[1]
    norm_metric = "Spot_Density_per_um3" if is_3d else "Spot_Density_per_um2"
    norm_label = "Spot Density per µm3" if is_3d else "Spot Density per µm2"
    sns.boxplot(
        data=df, x=norm_metric , y="Condition",
        whis=(0, 100), width=.6, ax=ax_b, palette="vlag"
    )
    sns.stripplot(data=df, x=norm_metric , y="Condition", 
                size=4, color=".3", ax=ax_b)
    ax_b.set_title(f"{norm_label} per Condition")
    ax_b.set_xlabel("Density")
    
    # Panel C - Spot coeficient of variation per condition (dot plot)
    ax_c = axes_flat[2]
    cv_df = df.groupby("Condition")["Spot_Count"].agg(lambda x: x.std() / x.mean()).reset_index()
    cv_df.rename(columns={"Spot_Count": "CV"}, inplace=True)
    
    sns.scatterplot(
        data=cv_df, x="CV", y="Condition", 
        s=100, color="crimson", marker="D", ax=ax_c
    )
    ax_c.set_title("Coefficient of Variation (Spot Count)")
    ax_c.set_xlabel("CV (Standard Deviation / Mean)")    
    
    # Panel D - Object size vs spot count (coloured by condition)
    ax_d = axes_flat[3]
    size_metric = "Volume_um3" if is_3d else "Area_um2"
    size_label = "Volume (µm³)" if is_3d else "Area (µm²)"
    sns.scatterplot(
        data=df, x=size_metric, y="Spot_Count", 
        hue="Condition", alpha=0.7, ax=ax_d
    )
    ax_d.set_title("Object Size vs Spot Count per Condition")
    ax_d.set_xlabel(f"{size_label}")
    ax_d.set_ylabel("Spot Count")
    
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [QC] Saved run summary: {out_path.name}")   


# %% config
# =====================================================================
# 2. Config
# =====================================================================

CONFIG = {
    # mode
    "do_3d": True,
    
    # paths
    "data_folder": Path(__file__).parent.parent / 'data',
    "cellpose_models_path": Path(__file__).parent.parent.parent / '_pipeline_assets/cellpose_models/cpsam_pseudo3d_4x_20260506',
    "spotiflow_models_path": Path(__file__).parent.parent.parent / '_pipeline_assets/bag_spot_model',
    "output_dir": Path(__file__).parent.parent / "output",
    
    # images order
    "segmentation_image": 0,
    "spot_image": 1,
    
    # cellpose
    "bin_factor": 4,
    "stitch_threshold": 0.4,    
    
    # spotiflow
    "prob_thresh": 0.5,
    "min_distance": 1,
    
}
# %% Main Loop
# =====================================================================
# 3. Main Loop
# =====================================================================

def run_pipeline(config: dict = CONFIG) -> None:
    """_summary_

    Args:
        config (dict, optional): _description_. Defaults to CONFIG.
    """
    # define dim mode
    do_3d = config["do_3d"]
    mode = "3d" if do_3d else "2d"
    
    # establish folder structure
    data_folder = Path(config["data_folder"])
    out_dir = Path(config["output_dir"])
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Pipeline starting | mode={mode.upper()} ===")
    print(f"Data folder: {data_folder}")
    
    # load models correct to the mode
    print("Loading Cellpose model...")
    model_cellpose = models.CellposeModel(gpu=True, pretrained_model=str(config["cellpose_models_path"]))
    
    print("Loading Spotiflow model...")
    try:
        model_spotiflow = Spotiflow.from_folder(str(config["spotiflow_models_path"]))
    except Exception as e:
        print(f"Something went wrong! Error details: {e}")
        print(f"Defaulting to the standard Spotiflow pretrained model...")
        if do_3d:
            model_spotiflow = Spotiflow.from_pretrained("smfish_3d")
        else:
            model_spotiflow = Spotiflow.from_pretrained("synth_complex")

    if model_spotiflow.config.is_3d and not do_3d:
        print(f"Conflict: Loaded Spotiflow model is 3D, but pipeline is set to {mode.upper()}.")
        print("Overriding with default 2D Spotiflow model...")
        model_spotiflow = Spotiflow.from_pretrained("synth_complex")
        
    elif not model_spotiflow.config.is_3d and do_3d:
        print(f"Conflict: Loaded Spotiflow model is 2D, but pipeline is set to {mode.upper()}.")
        print("Overriding with default 3D Spotiflow model...")
        model_spotiflow = Spotiflow.from_pretrained("smfish_3d")

    experiment = Path(__file__).parent.parent.name
    all_run_records = []

    file_list = [p for p in data_folder.iterdir() if p.is_file()]
    print(f"Found {len(file_list)} file(s): {[p.name for p in file_list]}")
    
    for filepath in file_list:
        condition = parse_condition_from_name(filepath.stem)
        source_file = filepath.name
        print(f"\n--- Processing: {filepath.name} ---")
        
        img = BioImage(filepath)
        all_scene_records = []
        
        num_scenes = len(img.scenes)
        print(f"  Scenes: {num_scenes}")
        
        
        for scene in range(num_scenes):
            img.set_scene(scene)
            print(f"  Scene {scene:02d} / {num_scenes - 1}")
            
            objects_stack = img.get_image_data("YX" if "Z" not in img.dims.order else "ZYX", C=config["segmentation_image"]).astype(np.float32)
            spots_stack = img.get_image_data("YX" if "Z" not in img.dims.order else "ZYX", C=config["spot_image"]).astype(np.float32)

            dx = img.physical_pixel_sizes.X or 1.0
            dz = img.physical_pixel_sizes.Z or 1.0
            
            print(f"    Segmenting ({mode})...")
            if do_3d:
                masks = segment_3d(
                    bf_stack=objects_stack, 
                    model_cellpose=model_cellpose, 
                    factor=config["bin_factor"], 
                    stitch_threshold=config["stitch_threshold"])
            else:
                masks = segment_2d(
                    bf_stack=objects_stack, 
                    model_cellpose=model_cellpose, 
                    factor=config["bin_factor"])

            n_obj = len(np.unique(masks)) -1
            print(f"    Found {n_obj} object(s) after border clearing")
            
            print(f"    Detecting spots ({mode})...")
            points, details = detect_spots_spotiflow(
                spot_stack=spots_stack,
                model_spotiflow=model_spotiflow,
                prob_thresh=config["prob_thresh"],
                min_distance=config["min_distance"],
            )

            spot_labels = assign_spots_to_mask(coordinates=points, masks=masks)
            print(f"    Detected {len(points)} spot(s), {(spot_labels > 0).sum()} assigned")
            
            scene_df = measure_objects(
                masks=masks,
                spot_labels=spot_labels,
                dx=dx,
                dz=dz,
                mode=mode,
                condition=condition,
                source_file=source_file,
                experiment=experiment,
                scene=scene,
            )
            all_scene_records.append(scene_df)
            
            print(f"    Generating QC figure...")
            qc_path = fig_dir / f"{condition}_S{scene:02d}_{mode}_qc.png"
            # make_qc_fifure()
            
        if all_scene_records:
            combined_df = pd.concat(all_scene_records, ignore_index=True)
            csv_path = tab_dir / f"{condition}_objects_{mode}.csv"
            combined_df.to_csv(csv_path, index=False)
            print(f"  Saved CSV: {csv_path.name}  ({len(combined_df)} rows)")
            all_run_records.append(combined_df)
            
            print(f"  Generating summary figure...")
            summary_path = fig_dir / f"{condition}_summary_{mode}.png"
            make_scene_summary_figure(
                df = combined_df, 
                condition = condition, 
                mode = mode, 
                out_path = summary_path)

    if all_run_records:
        run_df = pd.concat(all_run_records, ignore_index=True)
        csv_path = tab_dir / f"_run_objects_{mode}.csv"
        run_df.to_csv(csv_path, index=False)
        print(f"\nSaved run CSV: {run_csv_path.name}  ({len(run_df)} rows, {run_df['condition'].nunique()} condition(s))")

        print("Generating run summary figure...")
        run_summary_path = fig_dir / f"_run_summary_{mode}.png"
        make_run_summary_figure(
            df = run_df,
            experiment = experiment, 
            mode = mode, 
            out_path = run_summary_path)


    

# =====================================================================
# 4. Name = main
# =====================================================================

if __name__ == "__main__":
    run_pipeline(CONFIG)