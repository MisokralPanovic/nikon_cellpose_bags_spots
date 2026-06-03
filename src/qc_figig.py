import re
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Industry standard image processing tools
import skimage.exposure
import skimage.measure
import skimage.segmentation
from bioio import BioImage
from cellpose import models, plot, utils
from skimage.measure import block_reduce, regionprops_table
from spotiflow.model import Spotiflow

# =====================================================================
# 1. Functions
# =====================================================================
# %% Segmentation
def segment_2d(
    bf_stack: np.ndarray,
    model_cellpose: models.CellposeModel,
    factor: int = 4,
) -> np.ndarray:
    """Run BAG-pretrained Cellpose-SAM in 2D on stdev projection of stack, using image downscaling for faster processing."""
    std_proj = bf_stack.std(axis=0).astype(np.float32)
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
    """Run BAG-pretrained Cellpose-SAM in pseudo-3D on minimal projection subtracted stack."""
    min_substracted = bf_stack.astype(np.float32) - np.min(bf_stack, axis=0).astype(np.float32)
    img_binned = block_reduce(min_substracted, block_size=(1, factor, factor), func=np.mean) # type: ignore[arg-type]
    
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

# %% Spot detection and assignment to masks
def detect_spots_spotiflow(
    spot_stack: np.ndarray,
    model_spotiflow: Spotiflow,
    prob_thresh: float,
    min_distance: int,
) -> Tuple[np.ndarray, SimpleNamespace]:
    """Detect spot-like features in an image using the default Spotiflow model."""
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
    """Assigns coordinates (spots) to masks (objects), for 2D and 3D outputs."""
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
    """Measure morphological and spatial properties of segmented objects."""
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

    if raw.empty:
        return df

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
    """Extracts the base condition name from a filename stem."""
    match = re.split(r'[-_]\d+$', filename_stem)
    return match[0]


# %% config
# =====================================================================
# 2. Config
# =====================================================================

CONFIG = {
    "do_3d": True,
    "data_folder": Path(__file__).parent.parent / 'data',
    "cellpose_models_path": Path(__file__).parent.parent.parent / '_pipeline_assets/cellpose_models/cpsam_pseudo3d_4x_20260506',
    "spotiflow_models_path": Path(__file__).parent.parent.parent / '_pipeline_assets/bag_spot_model',
    "output_dir": Path(__file__).parent.parent / "output",
    "segmentation_image": 0,
    "spot_image": 1,
    "bin_factor": 4,
    "stitch_threshold": 0.4,    
    "prob_thresh": 0.5,
    "min_distance": 1,
}

# =====================================================================
# 3. Main Loop
# =====================================================================

def run_pipeline(config: dict = CONFIG) -> None:
    """Main execution block processing datasets loop over multi-scene frameworks."""
    do_3d = config["do_3d"]
    mode = "3d" if do_3d else "2d"
    
    data_folder = Path(config["data_folder"])
    out_dir = Path(config["output_dir"])
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Pipeline starting | mode={mode.upper()} ===")
    print(f"Data folder: {data_folder}")
    
    print("Loading Cellpose model...")
    model_cellpose = models.CellposeModel(gpu=True, pretrained_model=str(config["cellpose_models_path"]))
    
    print("Loading Spotiflow model...")
    try:
        model_spotiflow = Spotiflow.from_folder(str(config["spotiflow_models_path"]))
    except Exception as e:
        print(f"Something went wrong! Error details: {e}")
        print("Defaulting to the standard Spotiflow pretrained model...")
        model_spotiflow = Spotiflow.from_pretrained("smfish_3d" if do_3d else "synth_complex")

    if model_spotiflow.config.is_3d and not do_3d:
        print(f"Conflict: Loaded Spotiflow model is 3D, but pipeline is set to {mode.upper()}. Overriding...")
        model_spotiflow = Spotiflow.from_pretrained("synth_complex")
    elif not model_spotiflow.config.is_3d and do_3d:
        print(f"Conflict: Loaded Spotiflow model is 2D, but pipeline is set to {mode.upper()}. Overriding...")
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
            
            objects_stack = img.get_image_data("ZYX", C=config["segmentation_image"]).astype(np.float32)
            spots_stack = img.get_image_data("ZYX", C=config["spot_image"]).astype(np.float32)

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

            n_obj = len(np.unique(masks)) - 1
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
            if not scene_df.empty:
                all_scene_records.append(scene_df)
            
            print("    Generating QC figure...")
            qc_path = fig_dir / f"{condition}_S{scene:02d}_{mode}_qc.png"
            make_qc_figure(
                df=scene_df,
                coordinates=points,
                condition=condition,
                scene=scene,
                mode=mode,
                out_path=qc_path,
                segmentation_image=objects_stack,
                spots_image=spots_stack,
                masks=masks,
                flow_details=details,
                spot_labels=spot_labels
            )
            
        if all_scene_records:
            combined_df = pd.concat(all_scene_records, ignore_index=True)
            csv_path = tab_dir / f"{condition}_objects_{mode}.csv"
            combined_df.to_csv(csv_path, index=False)
            print(f"  Saved CSV: {csv_path.name}  ({len(combined_df)} rows)")
            all_run_records.append(combined_df)
            
            print("  Generating summary figure...")
            summary_path = fig_dir / f"{condition}_summary_{mode}.png"
            make_scene_summary_figure(
                df=combined_df, 
                condition=condition, 
                mode=mode, 
                out_path=summary_path)

    if all_run_records:
        run_df = pd.concat(all_run_records, ignore_index=True)
        run_csv_path = tab_dir / f"_run_objects_{mode}.csv"
        run_df.to_csv(run_csv_path, index=False)
        print(f"\nSaved run CSV: {run_csv_path.name}  ({len(run_df)} rows, {run_df['Condition'].nunique()} condition(s))")

        print("Generating run summary figure...")
        run_summary_path = fig_dir / f"_run_summary_{mode}.png"
        make_run_summary_figure(
            df=run_df,
            experiment=experiment, 
            mode=mode, 
            out_path=run_summary_path)


def make_qc_figure(
    df: pd.DataFrame,
    coordinates: np.ndarray,
    condition: str,
    scene: int,
    mode: str,
    out_path: Path,
    segmentation_image: np.ndarray,
    spots_image: np.ndarray,
    masks: np.ndarray,
    flow_details,
    spot_labels: np.ndarray
) -> None:
    """Generates a 6-panel QC matrix containing spatial overlays combined with distribution statistics."""
    is_3d = mode == "3d"
    has_spots = len(coordinates) > 0
    n_obj = len(np.unique(masks)) - 1
    
    # -------------------------------------------------------------------------
    # Projections Handling
    # -------------------------------------------------------------------------
    if segmentation_image.ndim == 3:
        seg_stdev = np.std(segmentation_image, axis=0)
    else:
        seg_stdev = segmentation_image
    seg_stdev_inv = seg_stdev.max() - seg_stdev
    seg_inv_norm = (seg_stdev_inv - seg_stdev_inv.min()) / (seg_stdev_inv.ptp() + 1e-8)
    
    if spots_image.ndim == 3:
        spots_stdev = np.std(spots_image, axis=0)
    else:
        spots_stdev = spots_image
    spots_stdev_norm = (spots_stdev - spots_stdev.min()) / (spots_stdev.ptp() + 1e-8)
    
    masks_2d = np.max(masks, axis=0) if (masks is not None and masks.ndim == 3) else masks

    # Parse spatial coordinate columns matching matrix positions [Z, Y, X]
    if has_spots:
        if is_3d:
            spot_z = coordinates[:, 0]
            spot_y = coordinates[:, 1]
            spot_x = coordinates[:, 2]
        else:
            spot_z = np.zeros(len(coordinates))
            spot_y = coordinates[:, 0]
            spot_x = coordinates[:, 1]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()
    fig.suptitle(
        f"{condition} - Scene {scene:02d} - [{mode.upper()} mode] - "
        f"{n_obj} object(s) - {len(coordinates)} spot(s)",
        fontsize=14, fontweight="bold")
    
    # Panel A - Inverted stack stdev + outlines
    ax_a = axes_flat[0]
    ax_a.imshow(seg_inv_norm, cmap="gray")
    if masks_2d is not None:
        mask_overlay = np.ma.masked_where(masks_2d == 0, masks_2d)
        ax_a.imshow(mask_overlay, alpha=0.3, cmap='tab10', vmin=1)
    ax_a.set_title("A: StDev Projection + Masks", loc='left', fontweight='bold')
    ax_a.axis("off")
    
    # Panel B - Spots stdev + spots (coloured by object)
    ax_b = axes_flat[1]
    ax_b.imshow(spots_stdev_norm, cmap="magma")
    if has_spots:
        inside_object = spot_labels > 0
        # Background elements
        ax_b.scatter(
            spot_x[~inside_object], spot_y[~inside_object],
            color="white", alpha=0.3, s=6, marker="x"
        )
        # Assigned spots
        ax_b.scatter(
            spot_x[inside_object], spot_y[inside_object],
            c=spot_labels[inside_object], cmap="tab10",
            s=15, edgecolors='white', linewidths=0.3, alpha=0.9
        )
    ax_b.set_title("B: Spot Detections (StDev Proj)", loc='left', fontweight='bold')
    ax_b.axis("off")
    
    # Panel C - Spotiflow flows
    ax_c = axes_flat[2]
    if flow_details is not None:
        try:
            flow_data = getattr(flow_details, 'flow', flow_details)
            if isinstance(flow_data, np.ndarray):
                if flow_data.ndim == 3 and flow_data.shape[0] == 2:  # 2D Vector Field
                    flow_viz = np.sqrt(flow_data[0]**2 + flow_data[1]**2)
                    ax_c.imshow(flow_viz, cmap="viridis")
                elif flow_data.ndim == 3:  # Heatmap output projection
                    ax_c.imshow(np.max(flow_data, axis=0), cmap="inferno")
                else:
                    ax_c.imshow(flow_data, cmap="inferno")
            else:
                ax_c.text(0.5, 0.5, 'Non-array Flow Format', ha='center', va='center', transform=ax_c.transAxes)
        except Exception:
            ax_c.text(0.5, 0.5, 'No Flow Data Found', ha='center', va='center', transform=ax_c.transAxes, color='gray')
    ax_c.set_title('C: Stereographic Flow / Centers', loc='left', fontweight='bold')
    ax_c.axis("off")
    
# Panel D - Spots per z slice | Spot nearest neighbour distance
    ax_d = axes_flat[3]
    if not has_spots:
        ax_d.text(0.5, 0.5, "No Spots Detected", ha="center", va="center", color="gray")
        ax_d.set_title("Distribution Profile")
        ax_d.axis("off")
    
    elif is_3d:
        # Volumetric Mode -> Histogram step count per Z slice channel
        total_z_planes = segmentation_image.shape[0] if segmentation_image.ndim == 3 else 1
        counts_per_slice = np.zeros(total_z_planes, dtype=int)
        
        slices, counts = np.unique(spots_z.astype(int), return_counts=True) # type: ignore
        for slc, cnt in zip(slices, counts):
            if 0 <= slc < total_z_planes:
                counts_per_slice[slc] = cnt
                
        z_indices = np.arange(total_z_planes)
        ax_d.step(z_indices, counts_per_slice, where="mid", color="#00FFCC", linewidth=1.5)
        ax_d.fill_between(z_indices, counts_per_slice, step="mid", color="#00FFCC", alpha=0.15)
        ax_d.set_xlabel("Z-Slice Index")
        ax_d.set_ylabel("Spot Count")
        ax_d.set_title("Spots per Z-Slice")
        ax_d.grid(True, linestyle=":", alpha=0.5)
        
    else:
        # Flat 2D Mode -> Nearest Neighbor Distance distribution
        if len(coordinates) < 2:
            ax_d.text(0.5, 0.5, "Insufficient Spots for NND", ha="center", va="center", color="gray")
            ax_d.axis("off")
        else:
            # Drop Z column if passing 3D-shaped coordinate arrays into 2D mode tracking loops
            spatial_xy = np.column_stack((spots_x, spots_y)) # type: ignore
            tree = KDTree(spatial_xy)
            distances, _ = tree.query(spatial_xy, k=2)
            nnd = distances[:, 1]
            
            ax_d.hist(nnd, bins="auto", density=True, color="#FF66CC", alpha=0.4, edgecolor="#FF66CC")
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(nnd)
                x_vals = np.linspace(nnd.min(), nnd.max(), 200)
                ax_d.plot(x_vals, kde(x_vals), color="#FF66CC", linewidth=1.5)
            except Exception:
                pass
            ax_d.set_xlabel("Nearest Neighbor Distance (px)")
            ax_d.set_ylabel("Density")
            ax_d.set_title("Spot Nearest Neighbor Distance")
            ax_d.grid(True, linestyle=":", alpha=0.5)
    
    # Panel E - Probability/Confidence Metric Score Tracking
    ax_e = axes_flat[4]
    if flow_details is not None and hasattr(flow_details, 'probs') and len(flow_details.probs) > 0:
        sns.histplot(x=flow_details.probs, kde=True, color='crimson', ax=ax_e, edgecolor='black')
        ax_e.set_xlabel("Spotiflow Probability Scores")
        ax_e.set_ylabel("Frequency")
    else:
        ax_e.text(0.5, 0.5, 'No Probability Metric Logs Available', ha='center', va='center', transform=ax_e.transAxes, color='gray')
    ax_e.set_title("E: Detection Confidence Profile", loc='left', fontweight='bold')
    
    # Panel F - Outlines + xy spotmap (coloured by z depth)
    ax_f = axes_flat[5]      
    if masks_2d is not None:
        ax_f.imshow(masks_2d > 0, cmap="bone", alpha=0.15)
    if has_spots:
        color_metric = spot_z if is_3d else (flow_details.probs if hasattr(flow_details, 'probs') else np.ones(len(spot_x)))
        label_metric = "Z-Slice" if is_3d else "Confidence Score"
        
        scatter_f = ax_f.scatter(
            spot_x, spot_y, c=color_metric, 
            cmap="coolwarm" if is_3d else "viridis", 
            s=20, edgecolors='black', linewidths=0.3
        )
        cbar = fig.colorbar(scatter_f, ax=ax_f, orientation='vertical', shrink=0.7)
        cbar.set_label(label_metric)
        
    ax_f.set_title(f"F: Spatial Mapping ({label_metric if has_spots else ''})", loc='left', fontweight='bold')
    ax_f.set_xlim(0, spots_stdev.shape[1])
    ax_f.set_ylim(spots_stdev.shape[0], 0)  # Maintain spatial orientation layout consistency
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [QC] Saved scene-level QC chart matrix: {out_path.name}") 


def make_scene_summary_figure(
    df: pd.DataFrame,
    condition: str,
    mode: str,
    out_path: Path
) -> None:
    """Generates a 4-panel quality control summary figure for a scene."""
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
    
    # Panel B - Object size distribution per scene boxplot
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
    ax_b.set_xlabel("Scene")
    ax_b.set_ylabel(size_label) 
        
    # Panel C - Spots per object histogram (pooled across scene)
    ax_c = axes_flat[2]
    sns.histplot(
        data=df, x="Spot_Count", ax=ax_c, discrete=True
    )
    ax_c.set_title("Pooled Spots per Object Distribution")
    ax_c.set_xlabel("Spots per Object")
    ax_c.set_ylabel("Count")    
    
    # Panel D - Object size vs spot count (coloured by scene)
    ax_d = axes_flat[3]
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
    """Generates a 4-panel quality control summary figure for an experiment."""
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
    norm_label = "Spot Density per µm³" if is_3d else "Spot Density per µm²"
    sns.boxplot(
        data=df, x=norm_metric , y="Condition",
        whis=(0, 100), width=.6, ax=ax_b, palette="vlag"
    )
    sns.stripplot(data=df, x=norm_metric , y="Condition", 
                size=4, color=".3", ax=ax_b)
    ax_b.set_title(f"{norm_label} per Condition")
    ax_b.set_xlabel("Density")
    
    # Panel C - Spot coefficient of variation per condition (dot plot)
    ax_c = axes_flat[2]
    cv_df = df.groupby("Condition")["Spot_Count"].agg(lambda x: x.std() / (x.mean() + 1e-8)).reset_index()
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
    

# =====================================================================
# 4. Name = main
# =====================================================================

if __name__ == "__main__":
    run_pipeline(CONFIG)