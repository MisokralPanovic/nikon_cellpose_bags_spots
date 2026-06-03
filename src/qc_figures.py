"""
QC figures for segmentation and spot detection pipeline.

make_qc_figure   — 6-panel per-scene QC (2×3 grid)
make_summary_figure — cross-scene/condition summary (2×2 grid)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import skimage.measure
import seaborn as sns
from typing import List

# ── colour helpers ─────────────────────────────────────────────────────────────
_LABEL_COLORS = plt.cm.tab20.colors  # 20 distinct RGBA tuples


def _obj_color(label: int) -> tuple:
    if label == 0:
        return (0.55, 0.55, 0.55, 0.6)
    return _LABEL_COLORS[(label - 1) % len(_LABEL_COLORS)] + (0.9,)


def _contours_from_mask(mask_2d: np.ndarray) -> list:
    """Return list of (label, contour_xy) pairs for all objects."""
    out = []
    for lbl in np.unique(mask_2d):
        if lbl == 0:
            continue
        binary = (mask_2d == lbl).astype(np.uint8)
        for c in skimage.measure.find_contours(binary, 0.5):
            out.append((lbl, c))
    return out


def _pct_clim(img: np.ndarray, lo: float = 1.0, hi: float = 99.5) -> tuple:
    v0, v1 = np.percentile(img[::4, ::4], [lo, hi])
    return (v0, v1 + 1e-6) if v1 <= v0 else (v0, v1)


def _compute_snr(fluor_2d: np.ndarray, masks_2d: np.ndarray) -> dict:
    """Mean intensity inside each mask / mean intensity of background."""
    bg = fluor_2d[masks_2d == 0]
    bg_mean = bg.mean() if len(bg) > 0 else 1.0
    snr = {}
    for lbl in np.unique(masks_2d):
        if lbl == 0:
            continue
        inside = fluor_2d[masks_2d == lbl]
        snr[int(lbl)] = round(float(inside.mean()) / bg_mean, 3) if bg_mean > 0 else np.nan
    return snr


# ── per-scene QC figure ────────────────────────────────────────────────────────
def make_qc_figure(
    bf_stack: np.ndarray,           # (Z,Y,X) or (Y,X) — segmentation channel
    fl_stack: np.ndarray,           # (Z,Y,X) or (Y,X) — fluorescence channel
    masks: np.ndarray,              # (Z,Y,X) or (Y,X) — integer labels
    coords: np.ndarray,             # (N,3) z,y,x  or  (N,2) y,x
    spot_labels: np.ndarray,        # (N,) mask label per spot; 0=unassigned
    dx: float,                      # XY pixel size µm
    dz: float,                      # Z step µm (ignored in 2D)
    mode: str,                      # "3d" or "2d"
    stem: str,                      # file stem for title
    scene: int,
    out_path: Path,
    spot_probs: np.ndarray | None = None,  # (N,) Spotiflow probabilities
    dpi: int = 130,
) -> None:
    """
    6-panel QC figure (2 rows × 3 cols):

    Row 1: BF std proj + mask contours  |  TAMRA std proj + detections  |  Spotiflow prob map
    Row 2: Spots per Z slice (3D) /         SNR per object               XY spot map + contours
           Spot intensity dist (2D)                                       (Z-coloured in 3D)
    """
    is_3d = mode == "3d"

    # ── projections ────────────────────────────────────────────────────────
    if is_3d:
        bf_proj  = bf_stack.std(axis=0).astype(np.float32)
        fl_proj  = fl_stack.max(axis=0).astype(np.float32)
        fl_std   = fl_stack.std(axis=0).astype(np.float32)
        mask_2d  = masks.max(axis=0)
        Z        = bf_stack.shape[0]
        z_um     = np.arange(Z) * dz
    else:
        bf_proj  = bf_stack.astype(np.float32)
        fl_proj  = fl_stack.astype(np.float32)
        fl_std   = fl_stack.astype(np.float32)
        mask_2d  = masks
        Z        = 1

    unique_labels = np.unique(mask_2d)
    unique_labels = unique_labels[unique_labels > 0]
    n_obj    = len(unique_labels)
    n_spots  = len(coords)
    contours = _contours_from_mask(mask_2d)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"{stem}  ·  Scene {scene:02d}  ·  [{mode.upper()} mode]  ·  "
        f"{n_obj} object(s)  ·  {n_spots} spot(s)",
        fontsize=12, fontweight="bold",
    )
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes

    # ── P1: BF std proj + mask contours ────────────────────────────────────
    ax1.imshow(bf_proj, cmap="gray", clim=_pct_clim(bf_proj), interpolation="nearest")
    for lbl, c in contours:
        color = _LABEL_COLORS[(lbl - 1) % len(_LABEL_COLORS)]
        ax1.plot(c[:, 1], c[:, 0], lw=1.5, color=color)
        ax1.text(c[:, 1].mean(), c[:, 0].mean(), str(lbl),
                 color="white", fontsize=8, ha="center", va="center", fontweight="bold")
    ax1.set_title("BF std proj + mask contours")
    ax1.axis("off")

    # ── P2: TAMRA std proj + faint detections ──────────────────────────────
    ax2.imshow(fl_std, cmap="magma", clim=_pct_clim(fl_std, 1, 99.9), interpolation="nearest")
    if n_spots > 0:
        ys = coords[:, -2]
        xs = coords[:, -1]
        for lbl_val in np.unique(spot_labels):
            sel = spot_labels == lbl_val
            c   = _obj_color(int(lbl_val))
            ax2.scatter(xs[sel], ys[sel], s=8, color=c[:3], alpha=0.35,
                        linewidths=0, zorder=3)
    ax2.set_title("TAMRA std proj + detections")
    ax2.axis("off")

    # ── P3: Spotiflow prob map or TAMRA max + detections ───────────────────
    if spot_probs is not None and n_spots > 0:
        ax3.imshow(fl_std, cmap="gray", clim=_pct_clim(fl_std), interpolation="nearest")
        sc = ax3.scatter(
            coords[:, -1], coords[:, -2],
            c=spot_probs, cmap="plasma", vmin=0, vmax=1,
            s=12, linewidths=0.3, edgecolors="k", alpha=0.9,
        )
        plt.colorbar(sc, ax=ax3, fraction=0.03, pad=0.02, label="Detection prob.")
        ax3.set_title("Spotiflow detection probabilities")
    else:
        ax3.imshow(fl_proj, cmap="hot", clim=_pct_clim(fl_proj, 1, 99.9), interpolation="nearest")
        if n_spots > 0:
            ax3.scatter(coords[:, -1], coords[:, -2], s=8, c="cyan",
                        alpha=0.5, linewidths=0)
        ax3.set_title("TAMRA max proj + detections")
    ax3.axis("off")

    # ── P4: Spots per Z slice (3D) or spot intensity distribution (2D) ─────
    if is_3d and n_spots > 0:
        z_idx = np.clip(np.round(coords[:, 0]).astype(int), 0, Z - 1)
        bottoms = np.zeros(Z)
        for lbl in unique_labels:
            sel = spot_labels == lbl
            if not sel.any():
                continue
            counts = np.bincount(z_idx[sel], minlength=Z)
            color  = _LABEL_COLORS[(lbl - 1) % len(_LABEL_COLORS)]
            ax4.bar(z_um, counts, width=dz * 0.85, bottom=bottoms,
                    color=color, edgecolor="none", alpha=0.8, label=f"Obj {lbl}")
            bottoms += counts
        sel0 = spot_labels == 0
        if sel0.any():
            counts0 = np.bincount(z_idx[sel0], minlength=Z)
            ax4.bar(z_um, counts0, width=dz * 0.85, bottom=bottoms,
                    color="gray", edgecolor="none", alpha=0.5, label="Unassigned")
        ax4.legend(fontsize=7, loc="upper right")
        ax4.set_xlabel("Z (µm)")
        ax4.set_ylabel("Spot count")
        ax4.set_title("Spots per Z slice")
    elif not is_3d and n_spots > 0:
        intensities = [
            float(fl_proj[
                int(np.clip(round(c[-2]), 0, fl_proj.shape[0] - 1)),
                int(np.clip(round(c[-1]), 0, fl_proj.shape[1] - 1)),
            ])
            for c in coords
        ]
        ax4.hist(intensities, bins=30, color="#0279EE", edgecolor="k", linewidth=0.4)
        ax4.set_xlabel("Spot intensity")
        ax4.set_ylabel("Count")
        ax4.set_title("Spot intensity distribution")
    else:
        ax4.text(0.5, 0.5, "No spots detected", ha="center", va="center",
                 transform=ax4.transAxes, color="gray")
        ax4.set_title("Spots per Z slice" if is_3d else "Spot intensity distribution")
    ax4.grid(axis="y", alpha=0.3)

    # ── P5: SNR per object ─────────────────────────────────────────────────
    snr = _compute_snr(fl_proj, mask_2d)
    if snr:
        lbls = list(snr.keys())
        vals = list(snr.values())
        colors = [_LABEL_COLORS[(l - 1) % len(_LABEL_COLORS)] for l in lbls]
        ax5.bar([f"Obj {l}" for l in lbls], vals, color=colors, edgecolor="k", linewidth=0.5)
        ax5.axhline(1.0, color="red", linestyle="--", linewidth=0.9, label="SNR = 1")
        ax5.set_ylabel("Mean intensity inside / outside")
        ax5.legend(fontsize=7)
    else:
        ax5.text(0.5, 0.5, "No objects", ha="center", va="center",
                 transform=ax5.transAxes, color="gray")
    ax5.set_title("Signal-to-noise per object")
    ax5.grid(axis="y", alpha=0.3)

    # ── P6: XY spot map + contours (Z-coloured in 3D) ─────────────────────
    if n_spots > 0:
        if is_3d:
            sc6 = ax6.scatter(
                coords[:, 2], coords[:, 1],
                c=coords[:, 0] * dz,
                cmap="viridis", s=8, alpha=0.75, linewidths=0,
                vmin=0, vmax=(Z - 1) * dz,
            )
            plt.colorbar(sc6, ax=ax6, fraction=0.03, pad=0.02, label="Z depth (µm)")
        else:
            for lbl_val in np.unique(spot_labels):
                sel = spot_labels == lbl_val
                c   = _obj_color(int(lbl_val))
                ax6.scatter(coords[sel, -1], coords[sel, -2], s=8,
                            color=c[:3], alpha=0.75, linewidths=0,
                            label=f"Obj {int(lbl_val)}" if lbl_val > 0 else "Unassigned")
            ax6.legend(fontsize=7, markerscale=2)

        for lbl, c in contours:
            color = _LABEL_COLORS[(lbl - 1) % len(_LABEL_COLORS)]
            ax6.plot(c[:, 1], c[:, 0], lw=1.2, color=color, alpha=0.7)

        h, w = mask_2d.shape
        ax6.set_xlim(0, w)
        ax6.set_ylim(h, 0)
        ax6.set_xlabel("X (px)")
        ax6.set_ylabel("Y (px)")
        ax6.set_aspect("equal")
    else:
        ax6.text(0.5, 0.5, "No spots", ha="center", va="center",
                 transform=ax6.transAxes, color="gray")
    ax6.set_title("XY spot map" + (" — Z depth" if is_3d else ""))

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [QC] {out_path.name}")


# ── cross-scene / cross-condition summary ──────────────────────────────────────
def make_summary_figure(
    df: pd.DataFrame,           # concatenated output of measure_objects() across scenes
    stem: str,                  # condition / file stem for title
    mode: str,                  # "3d" or "2d"
    out_path: Path,
    dpi: int = 120,
) -> None:
    """
    4-panel cross-scene summary figure (2 rows × 2 cols):

    Row 1: Total spots per scene (bar)  |  Object size distribution (violin)
    Row 2: Spots per object (hist)      |  Object size vs spot count (scatter)
    """
    is_3d    = mode == "3d"
    size_col = "Volume_um3" if is_3d else "Area_um2"
    size_lbl = "Volume (µm³)" if is_3d else "Area (µm²)"

    scenes       = sorted(df["Scene"].unique())
    scene_labels = [f"S{s:02d}" for s in scenes]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{stem}  —  Summary [{mode.upper()} mode]",
                 fontsize=13, fontweight="bold")
    ax_a, ax_b, ax_c, ax_d = axes.flat

    # ── A: total spots per scene ───────────────────────────────────────────
    spot_totals = [df[df["Scene"] == s]["Spot_Count"].sum() for s in scenes]
    ax_a.bar(scene_labels, spot_totals, color="#0279EE", edgecolor="k", linewidth=0.5)
    ax_a.set_xlabel("Scene")
    ax_a.set_ylabel("Total spot count")
    ax_a.set_title("Total spots per scene")
    ax_a.grid(axis="y", alpha=0.3)

    # ── B: object size distribution per scene (violin) ─────────────────────
    data_by_scene = [
        df[df["Scene"] == s][size_col].dropna().values
        for s in scenes
    ]
    data_by_scene = [(d if len(d) > 0 else np.array([0])) for d in data_by_scene]
    parts = ax_b.violinplot(data_by_scene, positions=range(len(scenes)),
                            showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#75A025")
        pc.set_alpha(0.6)
    ax_b.set_xticks(range(len(scenes)))
    ax_b.set_xticklabels(scene_labels)
    ax_b.set_xlabel("Scene")
    ax_b.set_ylabel(size_lbl)
    ax_b.set_title(f"Object size distribution ({size_lbl})")
    ax_b.grid(axis="y", alpha=0.3)

    # ── C: spots per object distribution ──────────────────────────────────
    spot_vals = df["Spot_Count"].dropna().values
    if len(spot_vals) > 0:
        ax_c.hist(spot_vals, bins=max(1, int(spot_vals.max()) + 1),
                  color="#FF9400", edgecolor="k", linewidth=0.4)
    ax_c.set_xlabel("Spots per object")
    ax_c.set_ylabel("Count")
    ax_c.set_title("Spots per object distribution")
    ax_c.grid(axis="y", alpha=0.3)

    # ── D: size vs spot count scatter, coloured by scene ──────────────────
    cmap = plt.cm.tab10
    for i, s in enumerate(scenes):
        sel = df["Scene"] == s
        ax_d.scatter(
            df.loc[sel, size_col], df.loc[sel, "Spot_Count"],
            color=cmap(i % 10), label=f"S{s:02d}",
            alpha=0.8, s=40, edgecolors="k", linewidths=0.4,
        )
    ax_d.set_xlabel(size_lbl)
    ax_d.set_ylabel("Spot count")
    ax_d.set_title("Object size vs spot count")
    ax_d.legend(fontsize=7, ncol=2)
    ax_d.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [QC] Summary: {out_path.name}")


# ── cross-condition biological summary ─────────────────────────────────────────
def make_condition_summary(
    df: pd.DataFrame,           # concatenated measure_objects() across all conditions
    mode: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """
    4-panel biological summary comparing conditions (2 rows × 2 cols):

    Row 1: Spot count by condition (boxplot)  |  Object size by condition (boxplot)
    Row 2: Spots per area/volume (boxplot)    |  Size vs spot count (scatter by condition)
    """
    is_3d    = mode == "3d"
    size_col = "Volume_um3" if is_3d else "Area_um2"
    size_lbl = "Volume (µm³)" if is_3d else "Area (µm²)"
    norm_col = "Spots_per_um3" if is_3d else "Spots_per_um2"
    norm_lbl = "Spots per µm³" if is_3d else "Spots per µm²"

    # derived column: spots normalised by size
    df = df.copy()
    df[norm_col] = df["Spot_Count"] / df[size_col].replace(0, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"Condition summary  [{mode.upper()} mode]",
                 fontsize=13, fontweight="bold")
    ax_a, ax_b, ax_c, ax_d = axes.flat

    palette = sns.color_palette("husl", n_colors=df["Condition"].nunique())

    def _add_n(ax, df, x_col, y_col="Condition"):
        """Annotate number of objects per condition."""
        for i, cond in enumerate(df[y_col].unique()):
            n = df[df[y_col] == cond][x_col].dropna().shape[0]
            xlim = ax.get_xlim()
            ax.text(xlim[1] * 0.97, i, f"n={n}",
                    va="center", ha="right", fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # ── A: spot count by condition ─────────────────────────────────────────
    sns.boxplot(data=df, y="Condition", x="Spot_Count",
                palette=palette, ax=ax_a, linewidth=0.8)
    sns.stripplot(data=df, y="Condition", x="Spot_Count",
                  color="k", size=3, alpha=0.4, ax=ax_a, jitter=True)
    ax_a.set_title("Spot count by condition")
    ax_a.set_xlabel("Spot count")
    ax_a.set_ylabel("")
    _add_n(ax_a, df, "Spot_Count")
    ax_a.grid(axis="x", alpha=0.3)

    # ── B: object size by condition ────────────────────────────────────────
    sns.boxplot(data=df, y="Condition", x=size_col,
                palette=palette, ax=ax_b, linewidth=0.8)
    sns.stripplot(data=df, y="Condition", x=size_col,
                  color="k", size=3, alpha=0.4, ax=ax_b, jitter=True)
    ax_b.set_title(f"Object size by condition")
    ax_b.set_xlabel(size_lbl)
    ax_b.set_ylabel("")
    _add_n(ax_b, df, size_col)
    ax_b.grid(axis="x", alpha=0.3)

    # ── C: normalised spot density by condition ────────────────────────────
    sns.boxplot(data=df, y="Condition", x=norm_col,
                palette=palette, ax=ax_c, linewidth=0.8)
    sns.stripplot(data=df, y="Condition", x=norm_col,
                  color="k", size=3, alpha=0.4, ax=ax_c, jitter=True)
    ax_c.set_title(f"Spot density by condition")
    ax_c.set_xlabel(norm_lbl)
    ax_c.set_ylabel("")
    _add_n(ax_c, df, norm_col)
    ax_c.grid(axis="x", alpha=0.3)

    # ── D: size vs spot count scatter, coloured by condition ───────────────
    sns.scatterplot(data=df, x=size_col, y="Spot_Count",
                    hue="Condition", palette=palette,
                    s=50, edgecolor="k", linewidth=0.4,
                    alpha=0.8, ax=ax_d)
    ax_d.set_xlabel(size_lbl)
    ax_d.set_ylabel("Spot count")
    ax_d.set_title("Object size vs spot count")
    ax_d.legend(fontsize=8, title="Condition", title_fontsize=8)
    ax_d.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [QC] Condition summary: {out_path.name}")