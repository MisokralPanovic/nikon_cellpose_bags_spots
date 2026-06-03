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
import skimage.segmentation
import skimage.measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bioio import BioImage
from cellpose import models, utils
from spotiflow.model import Spotiflow
import torch
torch.backends.cudnn.benchmark = True  # optimises kernels for your input size
# =====================================================================
# CONFIG — edit these before running
# =====================================================================
CONFIG = {
    # --- mode ---
    "do_3d": False,

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
    # Min-subtract
    min_proj = np.min(bf_stack, axis=0)
    min_sub = bf_stack - min_proj  # (Z,Y,X)

    binned = bin_xy(min_sub, cfg["bin_factor"])
    masks, _, _ = model_cp.eval(
        binned,
        do_3D=False,
        z_axis=0,
        #anisotropy=dz / dx,
        stitch_threshold=cfg["stitch_threshold"],
    )

    upscaled = upscale_xy(masks, cfg["bin_factor"], order=1)

    # Trim to original size (bin_xy may have cropped 1-3 px)
    # upscaled = upscaled[:, : bf_stack.shape[1], : bf_stack.shape[2]]

    # filtered = utils.remove_edge_masks(upscaled)
    
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
    """
    Returns a DataFrame with one row per segmented object.
    mode: "3d" or "2d"
    """
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

            # Z bounding box
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
# 6. QC FIGURES
# =====================================================================

# Colormap for up to 20 objects (label 0 = transparent)
_LABEL_COLORS = plt.cm.tab20.colors  # 20 distinct colors


def _label_to_color(label: int) -> tuple:
    if label == 0:
        return (0.5, 0.5, 0.5, 0.6)  # grey for unassigned
    return _LABEL_COLORS[(label - 1) % len(_LABEL_COLORS)] + (0.9,)


def _contours_from_mask(mask_2d: np.ndarray):
    """Return list of (label, contour_array) pairs."""
    result = []
    for lbl in np.unique(mask_2d):
        if lbl == 0:
            continue
        binary = (mask_2d == lbl).astype(np.uint8)
        contours = skimage.measure.find_contours(binary, 0.5)
        for c in contours:
            result.append((lbl, c))
    return result


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
    """
    6-panel QC figure per scene (2 rows × 3 cols):

    Row 1: BF proj + mask contours | TAMRA max-proj + detections | Spotiflow prob map
    Row 2: Spots per z-slice       | Radial spot density          | XY spot map (z-colored)

    spot_probs: (N,) array of Spotiflow detection probabilities (optional).
    """
    is_3d = mode == "3d"
    _DS = 4   # display downsample for image panels

    # ── projections & 2D footprint ─────────────────────────────────────────
    bf_proj = np.min(bf_stack, axis=0) if is_3d else bf_stack.std(axis=0)
    fl_max  = fl_stack.max(axis=0)
    mask_2d = masks.max(axis=0) if is_3d else masks
    Z       = bf_stack.shape[0]
    z_um    = np.arange(Z) * dz
    unique_labels = np.unique(mask_2d)
    unique_labels = unique_labels[unique_labels > 0]
    n_obj   = len(unique_labels)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"{stem}  ·  Scene {scene:02d}  ·  [{mode.upper()} mode]  ·  "
        f"{n_obj} object(s)  ·  {len(coords)} spot(s)",
        fontsize=12, fontweight="bold",
    )
    ax1, ax2, ax3 = axes[0]
    ax4, ax5, ax6 = axes[1]

    # ── Panel 1: BF projection + mask contours ─────────────────────────────
    p_lo, p_hi = np.percentile(bf_proj, [2, 98])
    ax1.imshow(bf_proj[::_DS, ::_DS], cmap="gray", vmin=p_lo, vmax=p_hi,
               interpolation="nearest")
    contours = _contours_from_mask(mask_2d)
    for lbl, c in contours:
        color = _LABEL_COLORS[(lbl - 1) % len(_LABEL_COLORS)]
        ax1.plot(c[:, 1] / _DS, c[:, 0] / _DS, linewidth=1.5, color=color)
        ax1.text(c[:, 1].mean() / _DS, c[:, 0].mean() / _DS, str(lbl),
                 color="white", fontsize=8, ha="center", va="center", fontweight="bold")
    proj_name = "Min proj" if is_3d else "Std proj"
    ax1.set_title(f"BF {proj_name} + mask contours")
    ax1.axis("off")

    # ── Panel 2: TAMRA max-proj + spot detections ──────────────────────────
    p_lo2, p_hi2 = np.percentile(fl_max, [1, 99.9])
    ax2.imshow(fl_max[::_DS, ::_DS], cmap="hot", vmin=p_lo2, vmax=p_hi2,
               interpolation="nearest")
    if len(coords) > 0:
        ys = coords[:, -2] / _DS
        xs = coords[:, -1] / _DS
        for lbl_val in np.unique(spot_labels):
            sel = spot_labels == lbl_val
            color = _label_to_color(int(lbl_val))
            ax2.scatter(xs[sel], ys[sel], s=14, c=[color[:3]], alpha=0.3,
                        linewidths=0.4, edgecolors="white")
    ax2.set_title(f"TAMRA max proj + detections")
    ax2.axis("off")

    # ── Panel 3: Spotiflow probability map (max-proj) ──────────────────────
    # If probs available: scatter spots sized/colored by probability
    # Otherwise: show TAMRA std-projection as a proxy for signal quality
    if spot_probs is not None and len(spot_probs) > 0 and len(coords) > 0:
        fl_std_proj = fl_stack.std(axis=0)
        p_lo3, p_hi3 = np.percentile(fl_std_proj, [1, 99])
        ax3.imshow(fl_std_proj[::_DS, ::_DS], cmap="gray", vmin=p_lo3, vmax=p_hi3,
                   interpolation="nearest")
        sc = ax3.scatter(
            coords[:, -1] / _DS, coords[:, -2] / _DS,
            c=spot_probs, cmap="plasma", vmin=0, vmax=1,
            s=18, linewidths=0.3, edgecolors="k", alpha=0.9,
        )
        plt.colorbar(sc, ax=ax3, fraction=0.03, pad=0.02, label="Detection prob.")
        ax3.set_title("Spot detection probabilities")
    else:
        fl_std_proj = fl_stack.std(axis=0)
        p_lo3, p_hi3 = np.percentile(fl_std_proj, [1, 99])
        ax3.imshow(fl_std_proj[::_DS, ::_DS], cmap="gray", vmin=p_lo3, vmax=p_hi3,
                   interpolation="nearest")
        if len(coords) > 0:
            ax3.scatter(coords[:, -1] / _DS, coords[:, -2] / _DS,
                        s=14, c="cyan", alpha=0.7, linewidths=0.3, edgecolors="k")
        ax3.set_title("TAMRA std proj + detections")
    ax3.axis("off")

    # ── Panel 4: Spots per z-slice ─────────────────────────────────────────
    if is_3d and len(coords) > 0:
        z_coords_int = np.round(coords[:, 0]).astype(int)
        z_counts = np.bincount(np.clip(z_coords_int, 0, Z - 1), minlength=Z)
        bars = ax4.bar(z_um, z_counts, width=dz * 0.85, color="#0279EE",
                       edgecolor="k", linewidth=0.4, alpha=0.85)
        # Colour bars by object membership: stacked if multiple objects
        if n_obj > 0:
            bottoms = np.zeros(Z)
            for lbl in unique_labels:
                sel = spot_labels == lbl
                if sel.any():
                    z_lbl = np.round(coords[sel, 0]).astype(int)
                    lbl_counts = np.bincount(np.clip(z_lbl, 0, Z - 1), minlength=Z)
                    color = _LABEL_COLORS[(lbl - 1) % len(_LABEL_COLORS)]
                    ax4.bar(z_um, lbl_counts, width=dz * 0.85, bottom=bottoms,
                            color=color, edgecolor="none", alpha=0.75, label=f"Obj {lbl}")
                    bottoms += lbl_counts
            # unassigned on top
            sel0 = spot_labels == 0
            if sel0.any():
                z_un = np.round(coords[sel0, 0]).astype(int)
                un_counts = np.bincount(np.clip(z_un, 0, Z - 1), minlength=Z)
                ax4.bar(z_um, un_counts, width=dz * 0.85, bottom=bottoms,
                        color="gray", edgecolor="none", alpha=0.5, label="Unassigned")
            ax4.legend(fontsize=7, loc="upper right")
        ax4.set_xlabel("Z (µm)")
        ax4.set_ylabel("Spot count")
        ax4.set_title("Spots per z-slice")
        ax4.grid(axis="y", alpha=0.3)
    else:
        # 2D mode: spot intensity histogram
        if len(coords) > 0:
            spot_intensities = []
            for c in coords:
                yi = int(np.clip(round(c[0]), 0, fl_stack.shape[-2] - 1))
                xi = int(np.clip(round(c[1]), 0, fl_stack.shape[-1] - 1))
                spot_intensities.append(float(fl_max[yi, xi]))
            ax4.hist(spot_intensities, bins=30, color="#0279EE", edgecolor="k", linewidth=0.4)
            ax4.set_xlabel("Spot intensity (max-proj)")
            ax4.set_ylabel("Count")
            ax4.set_title("Spot intensity distribution (2D)")
        else:
            ax4.text(0.5, 0.5, "No spots", ha="center", va="center", transform=ax4.transAxes)
            ax4.set_title("Spot intensity distribution")
        ax4.grid(axis="y", alpha=0.3)

    # ── Panel 5: Radial spot density per object ────────────────────────────
    if n_obj > 0 and len(coords) > 0:
        # Compute centroid of each object's 2D footprint
        from skimage.measure import regionprops
        props = regionprops(mask_2d)
        centroids = {p.label: np.array(p.centroid) for p in props}  # (row, col) = (y, x)
        # Compute max radius per object (approx from area)
        radii = {p.label: np.sqrt(p.area / np.pi) for p in props}

        plotted = False
        for lbl in unique_labels:
            if lbl not in centroids:
                continue
            cy, cx = centroids[lbl]
            r_max = radii[lbl]
            sel = spot_labels == lbl
            if not sel.any():
                continue
            spot_y = coords[sel, -2]
            spot_x = coords[sel, -1]
            r_abs = np.sqrt((spot_y - cy) ** 2 + (spot_x - cx) ** 2)
            r_norm = r_abs / r_max  # 0 = centre, 1 = edge
            color = _LABEL_COLORS[(lbl - 1) % len(_LABEL_COLORS)]
            ax5.hist(r_norm, bins=15, range=(0, 1.2), density=True,
                     alpha=0.55, color=color, edgecolor="k", linewidth=0.4,
                     label=f"Obj {lbl} (n={sel.sum()})")
            plotted = True

        if plotted:
            ax5.axvline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.6, label="Ball edge")
            ax5.set_xlabel("Normalised radial distance (0=centre, 1=edge)")
            ax5.set_ylabel("Density")
            ax5.set_title("Radial spot density")
            ax5.legend(fontsize=7)
        else:
            ax5.text(0.5, 0.5, "No assigned spots", ha="center", va="center",
                     transform=ax5.transAxes)
            ax5.set_title("Radial spot density")
        ax5.grid(axis="y", alpha=0.3)
    else:
        ax5.text(0.5, 0.5, "No objects / spots", ha="center", va="center",
                 transform=ax5.transAxes)
        ax5.set_title("Radial spot density")

    # ── Panel 6: XY spot map coloured by z-depth ──────────────────────────
    if len(coords) > 0 and is_3d:
        sc6 = ax6.scatter(
            coords[:, 2], coords[:, 1],   # x, y in pixels
            c=coords[:, 0] * dz,           # z in µm
            cmap="viridis", s=10, alpha=0.8, linewidths=0,
            vmin=0, vmax=(Z - 1) * dz,
        )
        plt.colorbar(sc6, ax=ax6, fraction=0.03, pad=0.02, label="Z depth (µm)")
        # Overlay mask contours for context
        for lbl, c in contours:
            color = _LABEL_COLORS[(lbl - 1) % len(_LABEL_COLORS)]
            ax6.plot(c[:, 1], c[:, 0], linewidth=1, color=color, alpha=0.6)
        ax6.set_xlim(0, bf_stack.shape[2])
        ax6.set_ylim(bf_stack.shape[1], 0)
        ax6.set_xlabel("X (px)")
        ax6.set_ylabel("Y (px)")
        ax6.set_title("XY spot map coloured by Z depth")
        ax6.set_aspect("equal")
    elif len(coords) > 0:
        # 2D: colour by assigned object
        for lbl_val in np.unique(spot_labels):
            sel = spot_labels == lbl_val
            color = _label_to_color(int(lbl_val))
            ax6.scatter(coords[sel, 1], coords[sel, 0], s=10,
                        c=[color[:3]], alpha=0.8, label=f"Obj {lbl_val}")
        ax6.set_xlim(0, bf_stack.shape[2])
        ax6.set_ylim(bf_stack.shape[1], 0)
        ax6.set_xlabel("X (px)")
        ax6.set_ylabel("Y (px)")
        ax6.set_title("XY spot map (2D)")
        ax6.set_aspect("equal")
    else:
        ax6.text(0.5, 0.5, "No spots", ha="center", va="center", transform=ax6.transAxes)
        ax6.set_title("XY spot map")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [QC] Saved {out_path.name}")


def make_summary_figure(
    all_records: pd.DataFrame,
    stem: str,
    mode: str,
    out_path: Path,
) -> None:
    """
    Cross-scene summary figure: spot counts, size distribution,
    thickness distribution (3D), spots vs size scatter.
    """
    is_3d = mode == "3d"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{stem}  —  Summary [{mode.upper()} mode]", fontsize=13, fontweight="bold")

    scenes = sorted(all_records["Scene"].unique())
    scene_labels = [f"S{s:02d}" for s in scenes]

    # Panel A: total spot count per scene
    ax = axes[0, 0]
    spot_totals = [all_records[all_records["Scene"] == s]["Spot_Count"].sum() for s in scenes]
    ax.bar(scene_labels, spot_totals, color="#0279EE", edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Scene")
    ax.set_ylabel("Total spot count")
    ax.set_title("Total spots per scene")
    ax.grid(axis="y", alpha=0.3)

    # Panel B: object size distribution (violin)
    ax = axes[0, 1]
    size_col = "Volume_um3" if is_3d else "Area_um2"
    size_label = "Volume (µm³)" if is_3d else "Area (µm²)"
    data_by_scene = [
        all_records[all_records["Scene"] == s][size_col].dropna().values
        for s in scenes
    ]
    data_by_scene_nonempty = [(d if len(d) > 0 else np.array([0])) for d in data_by_scene]
    parts = ax.violinplot(data_by_scene_nonempty, positions=range(len(scenes)),
                          showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#75A025")
        pc.set_alpha(0.6)
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels(scene_labels)
    ax.set_xlabel("Scene")
    ax.set_ylabel(size_label)
    ax.set_title(f"Object size distribution ({size_label})")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: thickness distribution (3D only) or spots per object (2D)
    ax = axes[1, 0]
    if is_3d:
        t_mask = all_records["Thickness_mask_um"].dropna().values
        t_bf = all_records["Thickness_BF_um"].dropna().values
        if len(t_mask) > 0:
            ax.hist(t_mask, bins=20, alpha=0.6, color="#FF9400", label="Mask-derived", edgecolor="k", linewidth=0.5)
        if len(t_bf) > 0:
            ax.hist(t_bf, bins=20, alpha=0.6, color="#E9ED4C", label="BF edge-contrast", edgecolor="k", linewidth=0.5)
        ax.set_xlabel("Thickness (µm)")
        ax.set_ylabel("Count")
        ax.set_title("Ball thickness distribution (3D)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    else:
        spots_per_obj = all_records["Spot_Count"].values
        ax.hist(spots_per_obj, bins=max(1, int(spots_per_obj.max()) + 1),
                color="#FF9400", edgecolor="k", linewidth=0.5)
        ax.set_xlabel("Spots per object")
        ax.set_ylabel("Count")
        ax.set_title("Spots per object distribution (2D)")
        ax.grid(axis="y", alpha=0.3)

    # Panel D: size vs spot count scatter
    ax = axes[1, 1]
    size_vals = all_records[size_col].values
    spot_vals = all_records["Spot_Count"].values
    scene_vals = all_records["Scene"].values
    cmap = plt.cm.tab10
    for i, s in enumerate(scenes):
        sel = scene_vals == s
        ax.scatter(size_vals[sel], spot_vals[sel],
                   color=cmap(i % 10), label=f"S{s:02d}", alpha=0.8, s=40, edgecolors="k", linewidths=0.4)
    ax.set_xlabel(size_label)
    ax.set_ylabel("Spot count")
    ax.set_title("Object size vs spot count")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [QC] Saved summary: {out_path.name}")


# =====================================================================
# 7. MAIN LOOP
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
    #model_spotiflow = Spotiflow.from_folder(str(cfg["spotiflow_model_path"]))
    model_spotiflow = Spotiflow.from_pretrained("general")
    
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
