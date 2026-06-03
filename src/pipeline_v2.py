"""
Gel Ball Segmentation & Spot Counting Pipeline
===============================================
Segments gel balls from BF z-stacks with Cellpose, detects TAMRA spots
with Spotiflow, assigns spots to objects, estimates ball thickness, and
produces QC figures + CSV outputs.

Set `do_3d = True/False` in CONFIG to switch modes.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import zoom, laplace
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Industry standard image processing packages
import skimage.exposure
import skimage.segmentation
import skimage.measure

from bioio import BioImage
from cellpose import models, utils
import cellpose.plot
from spotiflow.model import Spotiflow
import torch
torch.backends.cudnn.benchmark = True  # optimises kernels for your input size

# =====================================================================
# CONFIG — edit these before running
# =====================================================================
CONFIG = {
    # --- mode ---
    "do_3d": True,

    # --- paths ---
    "data_folder": Path(__file__).parent.parent / "data",
    "cellpose_model_path": Path(__file__).parent.parent.parent
        / "_pipeline_assets/cellpose_models/cpsam_pseudo3d_4x_20260506",
    "spotiflow_model_path": Path(__file__).parent.parent.parent
        / "_pipeline_assets/bag_spot_model",
    "output_dir": Path(__file__).parent / "results",

    # --- Cellpose ---
    "bin_factor": 4,
    "stitch_threshold": 0.4,
    "clear_border_buffer": 2,   # pixels in binned space

    # --- Spotiflow ---
    "prob_thresh": 0.5,
    "min_distance": 1,
}

# =====================================================================
# 1. HELPER FUNCTIONS
# =====================================================================

def bin_xy(img: np.ndarray, factor: int = 4) -> np.ndarray:
    """Block-mean XY binning. img: (Z,Y,X) or (Y,X), uint16 or float."""
    dtype = img.dtype
    if img.ndim == 2:
        h, w = img.shape
        h2, w2 = h // factor, w // factor
        out = (img[: h2 * factor, : w2 * factor]
               .reshape(h2, factor, w2, factor)
               .mean(axis=(1, 3)))
    else:
        z, h, w = img.shape
        h2, w2 = h // factor, w // factor
        out = (img[:, : h2 * factor, : w2 * factor]
               .reshape(z, h2, factor, w2, factor)
               .mean(axis=(2, 4)))
    return out.astype(dtype)


def upscale_xy(mask: np.ndarray, factor: int = 4, order: int = 1) -> np.ndarray:
    """Upscale XY only. order=0 nearest, order=1 bilinear."""
    zoom_factors = (factor, factor) if mask.ndim == 2 else (1, factor, factor)
    return zoom(mask.astype(np.float32), zoom_factors, order=order).astype(mask.dtype)


def _gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


def thickness_from_mask(mask_3d: np.ndarray, label: int, dz: float) -> float:
    """
    Mask-derived thickness: median number of z-planes occupied per XY pixel × dz.
    Returns NaN if label not found.
    """
    obj_vox = (mask_3d == label)
    if not obj_vox.any():
        return np.nan
    z_count_per_pixel = obj_vox.sum(axis=0)  # (Y, X)
    nonzero = z_count_per_pixel[z_count_per_pixel > 0]
    return float(np.median(nonzero) * dz)


def thickness_from_bf(
    bf_stack: np.ndarray,
    mask_3d: np.ndarray,
    label: int,
    dz: float,
) -> float:
    """
    BF edge-contrast thickness: Laplacian variance within the object's 2D
    bounding box per z-plane. Fit Gaussian to focus curve → FWHM × dz.
    Fallback: z-range where focus > 50% of peak.
    Returns NaN if label not found or fit fails badly.
    """
    obj_vox = (mask_3d == label)
    if not obj_vox.any():
        return np.nan

    # 2D bounding box of this object
    footprint = obj_vox.any(axis=0)
    rows = np.where(footprint.any(axis=1))[0]
    cols = np.where(footprint.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return np.nan
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1

    # Per-z Laplacian variance inside bounding box
    focus = np.array([
        laplace(bf_stack[z, r0:r1, c0:c1].astype(float)).var()
        for z in range(bf_stack.shape[0])
    ])

    if focus.max() == 0:
        return np.nan

    z_idx = np.arange(len(focus), dtype=float)
    focus_norm = focus / focus.max()

    # Try Gaussian fit
    try:
        mu0 = float(np.argmax(focus_norm))
        p0 = [1.0, mu0, len(focus) / 4.0, 0.0]
        bounds = ([0, 0, 0.5, -0.5], [2, len(focus), len(focus), 0.5])
        popt, _ = curve_fit(_gaussian, z_idx, focus_norm, p0=p0, bounds=bounds, maxfev=2000)
        fwhm_planes = 2.355 * abs(popt[2])
        return float(fwhm_planes * dz)
    except Exception:
        # Fallback: half-max z-range
        half_max = 0.5
        above = z_idx[focus_norm >= half_max]
        if len(above) < 2:
            return float(np.sum(focus_norm >= half_max) * dz)
        return float((above[-1] - above[0]) * dz)


# =====================================================================
# 2. SEGMENTATION
# =====================================================================

def segment_3d(
    bf_stack: np.ndarray,
    model_cp,
    dx: float,
    dz: float,
    cfg: dict,
) -> np.ndarray:
    """Returns filtered 3D mask (Z,Y,X)."""
    min_proj = np.min(bf_stack, axis=0)
    min_sub = bf_stack - min_proj  # (Z,Y,X)

    binned = bin_xy(min_sub, cfg["bin_factor"])
    masks, _, _ = model_cp.eval(
        binned,
        do_3D=False,
        z_axis=0,
        stitch_threshold=cfg["stitch_threshold"],
    )

    upscaled = upscale_xy(masks, cfg["bin_factor"], order=1)
    return upscaled


def segment_2d(
    bf_stack: np.ndarray,
    model_cp,
    dx: float,
    cfg: dict,
) -> np.ndarray:
    """Returns filtered 2D mask (Y,X) from std-projection."""
    std_proj = bf_stack.std(axis=0).astype(np.float32)
    binned = bin_xy(std_proj, cfg["bin_factor"])

    masks, _, _ = model_cp.eval(binned, do_3D=False)

    upscaled = upscale_xy(masks, cfg["bin_factor"], order=1)
    upscaled = upscaled[: bf_stack.shape[1], : bf_stack.shape[2]]

    filtered = skimage.segmentation.clear_border(
        upscaled, buffer_size=cfg["clear_border_buffer"]
    )
    return filtered


# =====================================================================
# 3. SPOT DETECTION
# =====================================================================

def detect_spots_3d(
    fl_stack: np.ndarray,
    model_sp,
    cfg: dict,
) -> np.ndarray:
    """Returns (N,3) array of (z,y,x) spot coordinates in pixels."""
    coords, _ = model_sp.predict(
        fl_stack,
        prob_thresh=cfg["prob_thresh"],
        min_distance=cfg["min_distance"],
    )
    return np.array(coords) if len(coords) > 0 else np.empty((0, 3))


def detect_spots_2d(
    fl_stack: np.ndarray,
    model_sp,
    cfg: dict,
) -> np.ndarray:
    """Max-projects TAMRA, returns (N,2) array of (y,x) spot coordinates."""
    max_proj = fl_stack.max(axis=0)
    coords, _ = model_sp.predict(
        max_proj,
        prob_thresh=cfg["prob_thresh"],
        min_distance=cfg["min_distance"],
    )
    return np.array(coords) if len(coords) > 0 else np.empty((0, 2))


# =====================================================================
# 4. SPOT ASSIGNMENT
# =====================================================================

def assign_spots_3d(masks_3d: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Look up mask label at each (z,y,x) voxel.
    coords: (N,3) float array of (z,y,x).
    Returns (N,) int array of labels (0 = unassigned).
    """
    if len(coords) == 0:
        return np.array([], dtype=int)
    zi = np.clip(np.round(coords[:, 0]).astype(int), 0, masks_3d.shape[0] - 1)
    yi = np.clip(np.round(coords[:, 1]).astype(int), 0, masks_3d.shape[1] - 1)
    xi = np.clip(np.round(coords[:, 2]).astype(int), 0, masks_3d.shape[2] - 1)
    return masks_3d[zi, yi, xi]


def assign_spots_2d(masks_2d: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Look up mask label at each (y,x) pixel.
    coords: (N,2) float array of (y,x).
    Returns (N,) int array of labels (0 = unassigned).
    """
    if len(coords) == 0:
        return np.array([], dtype=int)
    yi = np.clip(np.round(coords[:, 0]).astype(int), 0, masks_2d.shape[0] - 1)
    xi = np.clip(np.round(coords[:, 1]).astype(int), 0, masks_2d.shape[1] - 1)
    return masks_2d[yi, xi]


# =====================================================================
# 5. OBJECT MEASUREMENTS
# =====================================================================

def measure_objects(
    masks: np.ndarray,
    bf_stack: np.ndarray,
    coords: np.ndarray,
    spot_labels: np.ndarray,
    dx: float,
    dz: float,
    mode: str,  # "3d" or "2d"
    condition: str,
    source_file: str,
    scene: int,
    fov_id: str,
) -> pd.DataFrame:
    """Returns a DataFrame with one row per segmented object."""
    records = []
    unique_labels = np.unique(masks)
    unique_labels = unique_labels[unique_labels > 0]

    for lbl in unique_labels:
        obj_mask = masks == lbl
        spot_count = int((spot_labels == lbl).sum())

        row: dict = {
            "Condition": condition,
            "Source_File": source_file,
            "Scene": scene,
            "FOV_ID": fov_id,
            "Object_Label": int(lbl),
            "Spot_Count": spot_count,
        }

        if mode == "3d":
            voxel_count = int(obj_mask.sum())
            row["Volume_um3"] = round(voxel_count * dx * dx * dz, 4)
            row["Area_um2"] = np.nan

            z_coords = np.where(obj_mask.any(axis=(1, 2)))[0]
            z_span = float((z_coords[-1] - z_coords[0] + 1) * dz) if len(z_coords) > 0 else np.nan
            row["Z_Span_um"] = z_span

            row["Thickness_mask_um"] = thickness_from_mask(masks, lbl, dz)
            row["Thickness_BF_um"] = thickness_from_bf(bf_stack, masks, lbl, dz)

        else:  # 2d
            pixel_count = int(obj_mask.sum())
            row["Volume_um3"] = np.nan
            row["Area_um2"] = round(pixel_count * dx * dx, 4)
            row["Z_Span_um"] = np.nan
            row["Thickness_mask_um"] = np.nan
            row["Thickness_BF_um"] = np.nan

        records.append(row)

    return pd.DataFrame(records)


# =====================================================================
# 6. QC FIGURES (CLEANED UP & COLOR-SYNCHRONIZED VIA SKIMAGE + CELLPOSE)
# =====================================================================

def make_qc_figure(
    bf_stack: np.ndarray,
    fl_stack: np.ndarray,
    masks: np.ndarray,
    coords: np.ndarray,
    spot_labels: np.ndarray,
    scene: int,
    stem: str,
    mode: str,
    dx: float,
    dz: float,
    out_path: Path,
    objects_df: pd.DataFrame,
    spot_probs: np.ndarray | None = None,
) -> None:
    """6-panel QC figure per scene using offloaded skimage and cellpose plotting engines."""
    is_3d = mode == "3d"
    _DS = 4   # display downsample for image panels

    # ── Downsampled Projections & Footprints ───────────────────────────────
    bf_proj = np.min(bf_stack, axis=0) if is_3d else bf_stack.std(axis=0)
    fl_max  = fl_stack.max(axis=0)
    mask_2d = masks.max(axis=0) if is_3d else masks
    Z       = bf_stack.shape[0]
    z_um    = np.arange(Z) * dz

    # Native optimization pass to grab object centers and metrics via skimage
    props = skimage.measure.regionprops(mask_2d)
    n_obj = len(props)

    # Offload robust contrast stretching completely to scikit-image
    bf_scaled = skimage.exposure.rescale_intensity(bf_proj, in_range=tuple(np.percentile(bf_proj, [2, 98])))
    fl_max_scaled = skimage.exposure.rescale_intensity(fl_max, in_range=tuple(np.percentile(fl_max, [1, 99.9])))
    fl_std_proj = fl_stack.std(axis=0)
    fl_std_scaled = skimage.exposure.rescale_intensity(fl_std_proj, in_range=tuple(np.percentile(fl_std_proj, [1, 99])))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="white")
    fig.suptitle(
        f"{stem}  ·  Scene {scene:02d}  ·  [{mode.upper()} mode]  ·  "
        f"{n_obj} object(s)  ·  {len(coords)} spot(s)",
        fontsize=12, fontweight="bold",
    )
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axes

    # --- Panel 1: Native Cellpose Mask Overlay (Fixed for ax) ---
    # Generate the exact background overlay canvas that Cellpose generates internally
    cp_canvas = cellpose.plot.mask_overlay(bf_scaled, mask_2d)
    
    # Render it to your subplot grid axis safely with the matching downsampling slice
    ax1.imshow(cp_canvas[::_DS, ::_DS])
    
    # Overlay high-contrast object boundaries using standard skimage routines
    if n_obj > 0:
        outlines = skimage.segmentation.mark_boundaries(np.zeros_like(mask_2d), mask_2d, color=(1, 1, 1), mode="inner")
        ax1.imshow(outlines[::_DS, ::_DS], alpha=0.25)

    # Label each object exactly at its center of mass
    for p in props:
        y_c, x_c = p.centroid
        obj_color = cp_canvas[int(y_c), int(x_c)] / 255.0
        ax1.text(x_c / _DS, y_c / _DS, str(p.label), color="white", fontsize=8, 
                 ha="center", va="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", fc=obj_color, alpha=0.85, ec="none"))
    proj_name = "Min proj" if is_3d else "Std proj"
    ax1.set_title(f"BF {proj_name} + Cellpose Overlay")
    ax1.axis("off")

    # --- Panel 2: TAMRA max-proj + Sync-Colored Detections ---
    ax2.imshow(fl_max_scaled[::_DS, ::_DS], cmap="hot", interpolation="nearest")
    if len(coords) > 0:
        ys = coords[:, -2] / _DS
        xs = coords[:, -1] / _DS
        for p in props:
            sel = spot_labels == p.label
            if sel.any():
                obj_color = cp_canvas[int(p.centroid[0]), int(p.centroid[1])] / 255.0
                ax2.scatter(xs[sel], ys[sel], s=14, c=[obj_color[:3]], alpha=0.4, linewidths=0.4, edgecolors="white")
        # Handle unassigned spots (gray)
        sel0 = spot_labels == 0
        if sel0.any():
            ax2.scatter(xs[sel0], ys[sel0], s=14, c=[(0.5, 0.5, 0.5)], alpha=0.3, linewidths=0.4, edgecolors="white")
    ax2.set_title(f"TAMRA max proj + detections")
    ax2.axis("off")

    # --- Panel 3: Spotiflow Probability Map ---
    ax3.imshow(fl_std_scaled[::_DS, ::_DS], cmap="gray", interpolation="nearest")
    if spot_probs is not None and len(spot_probs) > 0 and len(coords) > 0:
        sc = ax3.scatter(
            coords[:, -1] / _DS, coords[:, -2] / _DS,
            c=spot_probs, cmap="plasma", vmin=0, vmax=1,
            s=18, linewidths=0.3, edgecolors="k", alpha=0.9,
        )
        plt.colorbar(sc, ax=ax3, fraction=0.03, pad=0.02, label="Detection prob.")
        ax3.set_title("Spot detection probabilities")
    else:
        if len(coords) > 0:
            ax3.scatter(coords[:, -1] / _DS, coords[:, -2] / _DS, s=14, c="cyan", alpha=0.7, linewidths=0.3, edgecolors="k")
        ax3.set_title("TAMRA std proj + detections")
    ax3.axis("off")

    # --- Panel 4: Spots per z-slice ---
    if is_3d and len(coords) > 0:
        z_coords_int = np.round(coords[:, 0]).astype(int)
        z_counts = np.bincount(np.clip(z_coords_int, 0, Z - 1), minlength=Z)
        ax4.bar(z_um, z_counts, width=dz * 0.85, color="#0279EE", edgecolor="k", linewidth=0.4, alpha=0.15)
        
        if n_obj > 0:
            bottoms = np.zeros(Z)
            for p in props:
                sel = spot_labels == p.label
                if sel.any():
                    z_lbl = np.round(coords[sel, 0]).astype(int)
                    lbl_counts = np.bincount(np.clip(z_lbl, 0, Z - 1), minlength=Z)
                    obj_color = cp_canvas[int(p.centroid[0]), int(p.centroid[1])] / 255.0
                    ax4.bar(z_um, lbl_counts, width=dz * 0.85, bottom=bottoms,
                            color=obj_color, edgecolor="none", alpha=0.75, label=f"Obj {p.label}")
                    bottoms += lbl_counts
            sel0 = spot_labels == 0
            if sel0.any():
                z_un = np.round(coords[sel0, 0]).astype(int)
                un_counts = np.bincount(np.clip(z_un, 0, Z - 1), minlength=Z)
                ax4.bar(z_um, un_counts, width=dz * 0.85, bottom=bottoms, color="gray", edgecolor="none", alpha=0.5, label="Unassigned")
            ax4.legend(fontsize=7, loc="upper right")
        ax4.set_xlabel("Z (µm)")
        ax4.set_ylabel("Spot count")
        ax4.set_title("Spots per z-slice")
        ax4.grid(axis="y", alpha=0.3)
    else:
        if len(coords) > 0:
            spot_intensities = [float(fl_max[int(np.clip(round(c[0]), 0, fl_stack.shape[-2]-1)), int(np.clip(round(c[1]), 0, fl_stack.shape[-1]-1))]) for c in coords]
            ax4.hist(spot_intensities, bins=30, color="#0279EE", edgecolor="k", linewidth=0.4)
            ax4.set_xlabel("Spot intensity (max-proj)")
            ax4.set_ylabel("Count")
            ax4.set_title("Spot intensity distribution (2D)")
        else:
            ax4.text(0.5, 0.5, "No spots", ha="center", va="center", transform=ax4.transAxes)
        ax4.grid(axis="y", alpha=0.3)

    # --- Panel 5: Radial spot density per object ---
    if n_obj > 0 and len(coords) > 0:
        plotted = False
        for p in props:
            cy, cx = p.centroid
            r_max = np.sqrt(p.area / np.pi)
            sel = spot_labels == p.label
            if not sel.any():
                continue
            r_abs = np.sqrt((coords[sel, -2] - cy) ** 2 + (coords[sel, -1] - cx) ** 2)
            r_norm = r_abs / r_max
            obj_color = cp_canvas[int(cy), int(cx)] / 255.0
            ax5.hist(r_norm, bins=15, range=(0, 1.2), density=True, alpha=0.55, 
                     color=obj_color, edgecolor="k", linewidth=0.4, label=f"Obj {p.label} (n={sel.sum()})")
            plotted = True

        if plotted:
            ax5.axvline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.6, label="Ball edge")
            ax5.set_xlabel("Normalised radial distance")
            ax5.set_ylabel("Density")
            ax5.set_title("Radial spot density")
            ax5.legend(fontsize=7)
        else:
            ax5.text(0.5, 0.5, "No assigned spots", ha="center", va="center", transform=ax5.transAxes)
        ax5.grid(axis="y", alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "No objects / spots", ha="center", va="center", transform=ax5.transAxes)

    # --- Panel 6: XY spot map + Multi-Object Boundary Outlines ---
    if n_obj > 0:
        boundary_canvas = skimage.segmentation.mark_boundaries(np.zeros_like(mask_2d), mask_2d, color=(0.4, 0.4, 0.4), mode="thick")
        ax6.imshow(boundary_canvas, alpha=0.4)
        
    if len(coords) > 0:
        if is_3d:
            sc6 = ax6.scatter(coords[:, 2], coords[:, 1], c=coords[:, 0] * dz, cmap="viridis", s=10, alpha=0.8, zorder=3)
            plt.colorbar(sc6, ax=ax6, fraction=0.03, pad=0.02, label="Z depth (µm)")
        else:
            for p in props:
                sel = spot_labels == p.label
                if sel.any():
                    obj_color = cp_canvas[int(p.centroid[0]), int(p.centroid[1])] / 255.0
                    ax6.scatter(coords[sel, 1], coords[sel, 0], s=10, c=[obj_color[:3]], alpha=0.8, zorder=3)
        ax6.set_xlim(0, bf_stack.shape[2])
        ax6.set_ylim(bf_stack.shape[1], 0)
        ax6.set_title("XY Spot Map" + (" (Z-Colored)" if is_3d else " (Object-Colored)"))
        ax6.set_aspect("equal")
    else:
        ax6.text(0.5, 0.5, "No spots", ha="center", va="center", transform=ax6.transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [QC] Saved {out_path.name}")


# =====================================================================
# 7. METRIC COMPARISON & SUMMARY PLOTS (DELEGATED TO SEABORN)
# =====================================================================

def make_summary_figure(
    all_records: pd.DataFrame,
    stem: str,
    mode: str,
    out_path: Path,
) -> None:
    """Cross-scene summary figure. Uses Seaborn structures instead of explicit loop layouts."""
    is_3d = mode == "3d"
    size_col = "Volume_um3" if is_3d else "Area_um2"
    size_label = "Volume (µm³)" if is_3d else "Area (µm²)"

    all_records = all_records.sort_values("Scene")
    all_records["Scene_Label"] = all_records["Scene"].apply(lambda s: f"S{s:02d}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    fig.suptitle(f"{stem}  —  Summary [{mode.upper()} mode]", fontsize=13, fontweight="bold")
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # Panel A: Automated bar aggregation using Seaborn
    sns.barplot(data=all_records, x="Scene_Label", y="Spot_Count", estimator=np.sum, errorbar=None, color="#0279EE", edgecolor="k", ax=ax_a)
    ax_a.set_ylabel("Total spot count")
    ax_a.set_xlabel("")
    ax_a.set_title("Total spots per scene")

    # Panel B: Clean object morphometrics via violin plots
    sns.violinplot(data=all_records, x="Scene_Label", y=size_col, inner="median", color="#75A025", ax=ax_b)
    ax_b.set_ylabel(size_label)
    ax_b.set_xlabel("")
    ax_b.set_title(f"Object size distribution ({size_label})")

    # Panel C: Thickness histograms or spots-per-object profiles
    if is_3d:
        if "Thickness_mask_um" in all_records.columns and all_records["Thickness_mask_um"].notna().any():
            sns.histplot(data=all_records, x="Thickness_mask_um", color="#FF9400", label="Mask-derived", kde=True, ax=ax_c, alpha=0.6)
        if "Thickness_BF_um" in all_records.columns and all_records["Thickness_BF_um"].notna().any():
            sns.histplot(data=all_records, x="Thickness_BF_um", color="#E9ED4C", label="BF edge-contrast", kde=True, ax=ax_c, alpha=0.6)
        ax_c.set_xlabel("Thickness (µm)")
        ax_c.set_title("Ball thickness distribution (3D)")
        ax_c.legend()
    else:
        sns.histplot(data=all_records, x="Spot_Count", discrete=True, color="#FF9400", edgecolor="k", ax=ax_c)
        ax_c.set_xlabel("Spots per object")
        ax_c.set_title("Spots per object distribution (2D)")

    # Panel D: Scatter comparisons with automated grouping legends
    sns.scatterplot(data=all_records, x=size_col, y="Spot_Count", hue="Scene_Label", palette="tab10", alpha=0.8, s=40, edgecolors="k", ax=ax_d)
    ax_d.set_xlabel(size_label)
    ax_d.set_ylabel("Spot count")
    ax_d.set_title("Object size vs spot count")
    ax_d.legend(fontsize=7, title="Scenes", ncol=2)

    for ax in axes.flat: ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [QC] Saved summary: {out_path.name}")


# =====================================================================
# 8. MAIN LOOP
# =====================================================================

def run_pipeline(cfg: dict = CONFIG) -> None:
    do_3d = cfg["do_3d"]
    mode = "3d" if do_3d else "2d"
    data_folder = Path(cfg["data_folder"])
    out_dir = Path(cfg["output_dir"])
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Pipeline starting | mode={mode.upper()} ===")
    print(f"Data folder: {data_folder}")

    # Load models
    print("Loading Cellpose model...")
    model_cellpose = models.CellposeModel(
        gpu=True,
        pretrained_model=str(cfg["cellpose_model_path"]),
    )
    print("Loading Spotiflow model...")
    model_spotiflow = Spotiflow.from_folder(str(cfg["spotiflow_model_path"]))
    
    file_list = sorted([p for p in data_folder.iterdir() if p.is_file()])
    print(f"Found {len(file_list)} file(s): {[p.name for p in file_list]}")

    for filepath in file_list:
        stem = filepath.stem
        condition = stem  # full stem = condition
        print(f"\n--- Processing: {filepath.name} ---")

        img = BioImage(filepath)
        num_scenes = len(img.scenes)
        print(f"  Scenes: {num_scenes}")

        all_scene_records: list[pd.DataFrame] = []

        for scene in range(num_scenes):
            img.set_scene(scene)
            print(f"  Scene {scene:02d} / {num_scenes - 1}")

            bf_stack = img.get_image_data("ZYX", C=0)
            fl_stack = img.get_image_data("ZYX", C=1)

            dx = float(img.physical_pixel_sizes.X or 1.0)
            dz = float(img.physical_pixel_sizes.Z or 1.0)
            fov_id = f"{stem}_S{scene:02d}"

            # --- Segmentation ---
            print(f"    Segmenting ({mode})...")
            if do_3d:
                masks = segment_3d(bf_stack, model_cellpose, dx, dz, cfg)
            else:
                masks = segment_2d(bf_stack, model_cellpose, dx, cfg)

            n_obj = len(np.unique(masks)) - 1
            print(f"    Found {n_obj} object(s) after border clearing")

            # --- Spot detection ---
            print(f"    Detecting spots ({mode})...")
            if do_3d:
                coords = detect_spots_3d(fl_stack, model_spotiflow, cfg)
                spot_labels = assign_spots_3d(masks, coords)
            else:
                coords = detect_spots_2d(fl_stack, model_spotiflow, cfg)
                mask_2d_footprint = masks.max(axis=0) if masks.ndim == 3 else masks
                spot_labels = assign_spots_2d(mask_2d_footprint, coords)

            print(f"    Detected {len(coords)} spot(s), {(spot_labels > 0).sum()} assigned")

            # --- Measurements ---
            print(f"    Measuring objects...")
            scene_df = measure_objects(
                masks=masks,
                bf_stack=bf_stack,
                coords=coords,
                spot_labels=spot_labels,
                dx=dx,
                dz=dz,
                mode=mode,
                condition=condition,
                source_file=filepath.name,
                scene=scene,
                fov_id=fov_id,
            )
            all_scene_records.append(scene_df)

            # --- QC figure ---
            print(f"    Generating QC figure...")
            qc_path = fig_dir / f"{stem}_S{scene:02d}_{mode}_qc.png"
            make_qc_figure(
                bf_stack=bf_stack,
                fl_stack=fl_stack,
                masks=masks,
                coords=coords,
                spot_labels=spot_labels,
                scene=scene,
                stem=stem,
                mode=mode,
                dx=dx,
                dz=dz,
                out_path=qc_path,
                objects_df=scene_df,
            )

        # --- Save combined CSV ---
        if all_scene_records:
            combined_df = pd.concat(all_scene_records, ignore_index=True)
            csv_path = tbl_dir / f"{stem}_objects_{mode}.csv"
            combined_df.to_csv(csv_path, index=False)
            print(f"  Saved CSV: {csv_path.name}  ({len(combined_df)} rows)")

            # --- Summary figure ---
            print(f"  Generating summary figure...")
            summary_path = fig_dir / f"{stem}_summary_{mode}.png"
            make_summary_figure(combined_df, stem, mode, summary_path)

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    run_pipeline(CONFIG)