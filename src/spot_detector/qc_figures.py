import logging
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from spot_detector.config import PipelineConfig
from spot_detector.qc_panels import (
    ImageData,
    SpotData,
    _panel_ecdf,
    _panel_flow,
    _panel_segemntation,
    _panel_spot_detection,
    _panel_spotmap,
    _panel_z_distribution,
)

logger = logging.getLogger(__name__)


# qc figures
def make_qc_figure(
    condition: str,
    scene: int,
    mode: str,
    out_path: Path,
    segmentation_image: np.ndarray,
    spots_image: np.ndarray,
    masks: np.ndarray,
    coordinates: np.ndarray,
    spot_labels: np.ndarray,
    flow_details: SimpleNamespace,
    dx: float,
    dz: float,
    config: PipelineConfig,
) -> None:
    """Generate a 2x3 panel QC figure for one scene.

    Panels:
        [0,0] StDev projection of segmentation channel + mask overlays
        [0,1] StDev projection of spot channel (gray_r, contrast-stretched) + detections
        [0,2] Spotiflow flow field (HSV hue-wheel)
        [1,0] Spots per z-slice stacked histogram (3D) | NND distribution (2D)
        [1,1] Detection confidence ECDF (inside object vs background)
        [1,2] XY spotmap in µm, coloured by z-depth (3D) or uniform (2D)

    Args:
        condition (str): Experimental condition name.
        scene (int): Scene index.
        mode (str): '2d' or '3d'.
        out_path (Path): Output file path.
        segmentation_image (np.ndarray): Raw segmentation channel (2D or 3D).
        spots_image (np.ndarray): Raw spot channel (2D or 3D).
        masks (np.ndarray): Segmentation masks (2D or 3D).
        coordinates (np.ndarray): Detected spot coordinates from Spotiflow.
        spot_labels (np.ndarray): Per-spot object label assignments.
        flow_details (SimpleNamespace): Spotiflow details object (flow, prob attributes).
        dx (float): XY pixel size in µm.
        dz (float): Z pixel size in µm.
        config (PipelineConfig): Pipeline config PipelineConfig object (must contain 'prob_thresh').

    Raises:
        FatalPipelineError: Propagated uncaught if raised by any panel-rendering
            call, so a fatal error still terminates the pipeline instead of being
            caught here and treated as an ordinary per-panel rendering failure.
    """
    is_3d = mode == "3d"
    n_obj = len(np.unique(masks)) - 1

    images = ImageData(
        segmentation_image=segmentation_image, spot_image=spots_image, masks=masks
    )
    spots = SpotData(coordinates=coordinates, dz=dz, dx=dx, is_3d=is_3d)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()
    fig.suptitle(
        f"{condition} - Scene {scene:02d} - [{mode.upper()} mode] - "
        f"{n_obj} object(s) - {len(coordinates)} spot(s)",
        fontsize=12,
        fontweight="bold",
    )

    _panel_segemntation(ax=axes_flat[0], images=images, dx=dx)
    _panel_spot_detection(
        ax=axes_flat[1], images=images, spots=spots, spot_labels=spot_labels
    )
    _panel_flow(ax=axes_flat[2], flow_details=flow_details)
    _panel_z_distribution(
        ax=axes_flat[3], images=images, spots=spots, spot_labels=spot_labels
    )
    _panel_ecdf(
        ax=axes_flat[4],
        spots=spots,
        spot_labels=spot_labels,
        flow_details=flow_details,
        config=config,
    )
    _panel_spotmap(ax=axes_flat[5], images=images, spots=spots)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.debug(f"[QC] Saved summary: {out_path.name}")


# %% Summary Figures
def make_scene_summary_figure(
    df: pd.DataFrame,
    condition: str,
    mode: str,
    out_path: Path,
) -> None:
    """Generate a 4-panel summary figure for one condition (all scenes).

    Panels:
        A  Spot count per object per scene (swarmplot, coloured by scene)
        B  Object size distribution per scene (boxplot + stripplot, coloured by scene)
        C  Pooled spots-per-object histogram
        D  Object size vs spot count scatter (coloured by scene)

    Args:
        df (pd.DataFrame): Combined scene dataframe.
        condition (str): Experimental condition name.
        mode (str): '2d' or '3d'.
        out_path (Path): Output file path.
    """
    is_3d = mode == "3d"
    size_metric = "Volume_um3" if is_3d else "Area_um2"
    size_label = "Volume (µm³)" if is_3d else "Area (µm²)"

    # Ensure Scene is treated as a categorical string for consistent palette mapping
    df = df.copy()
    df["Scene"] = df["Scene"].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    fig.suptitle(
        f"{condition}  —  Summary [{mode.upper()} mode]", fontsize=13, fontweight="bold"
    )

    # Panel A — spot count per object per scene (swarmplot, same palette as Panel B)
    ax_a = axes_flat[0]
    sns.swarmplot(
        data=df,
        x="Scene",
        y="Spot_Count",
        hue="Scene",
        palette="tab10",
        legend=False,
        size=4,
        ax=ax_a,
    )
    ax_a.set_title("Spot Count per Object per Scene")
    ax_a.set_xlabel("Scene")
    ax_a.set_ylabel("Spot Count")

    # Panel B — object size distribution per scene (boxplot + stripplot)
    ax_b = axes_flat[1]
    sns.boxplot(
        data=df,
        x="Scene",
        y=size_metric,
        hue="Scene",
        palette="tab10",
        legend=False,
        whis=(0, 100),
        width=0.6,
        ax=ax_b,
    )
    sns.stripplot(
        data=df,
        x="Scene",
        y=size_metric,
        size=4,
        color=".3",
        ax=ax_b,
    )
    ax_b.set_title(f"Object Size Distribution ({size_label})")
    ax_b.set_xlabel("Scene")
    ax_b.set_ylabel(size_label)

    # Panel C — pooled spots-per-object histogram
    ax_c = axes_flat[2]
    sns.histplot(data=df, x="Spot_Count", ax=ax_c)
    ax_c.set_title("Pooled Spots per Object Distribution")
    ax_c.set_xlabel("Spots per Object")
    ax_c.set_ylabel("Count")

    # Panel D — object size vs spot count scatter (coloured by scene)
    ax_d = axes_flat[3]
    sns.scatterplot(
        data=df,
        x=size_metric,
        y="Spot_Count",
        hue="Scene",
        palette="tab10",
        alpha=0.7,
        ax=ax_d,
    )
    ax_d.set_title("Object Size vs Spot Count per Scene")
    ax_d.set_xlabel(size_label)
    ax_d.set_ylabel("Spot Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[QC] Saved scene summary: {out_path.name}")


# ── Run summary figure ─────────────────────────────────────────────────────────


def make_run_summary_figure(
    df: pd.DataFrame,
    experiment: str,
    mode: str,
    out_path: Path,
) -> None:
    """Generate a 4-panel summary figure for a full experiment run (all conditions).

    Panels:
        A  Spot count per condition (boxplot + stripplot)
        B  Spot density per condition (boxplot + stripplot)
        C  Coefficient of variation of spot count per condition (dot plot)
        D  Object size vs spot count scatter (coloured by condition)

    Args:
        df (pd.DataFrame): Combined run dataframe.
        experiment (str): Experiment/run name.
        mode (str): '2d' or '3d'.
        out_path (Path): Output file path.
    """
    is_3d = mode == "3d"
    norm_metric = "Spot_Density_per_um3" if is_3d else "Spot_Density_per_um2"
    norm_label = "Spot Density per µm³" if is_3d else "Spot Density per µm²"
    size_metric = "Volume_um3" if is_3d else "Area_um2"
    size_label = "Volume (µm³)" if is_3d else "Area (µm²)"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()
    fig.suptitle(
        f"{experiment}  —  Run Summary [{mode.upper()} mode]",
        fontsize=13,
        fontweight="bold",
    )

    # Panel A — spot count per condition
    ax_a = axes_flat[0]
    sns.boxplot(
        data=df,
        x="Spot_Count",
        y="Condition",
        hue="Condition",
        palette="tab10",
        legend=False,
        whis=(0, 100),
        width=0.6,
        ax=ax_a,
    )
    sns.stripplot(
        data=df,
        x="Spot_Count",
        y="Condition",
        size=4,
        color=".3",
        ax=ax_a,
    )
    ax_a.set_title("Spot Count per Condition")
    ax_a.set_xlabel("Spot Count")
    ax_a.set_ylabel("Condition")

    # Panel B — spot density per condition
    ax_b = axes_flat[1]
    sns.boxplot(
        data=df,
        x=norm_metric,
        y="Condition",
        hue="Condition",
        palette="tab10",
        legend=False,
        whis=(0, 100),
        width=0.6,
        ax=ax_b,
    )
    sns.stripplot(
        data=df,
        x=norm_metric,
        y="Condition",
        size=4,
        color=".3",
        ax=ax_b,
    )
    ax_b.set_title(f"{norm_label} per Condition")
    ax_b.set_xlabel("Density")
    ax_b.set_ylabel("Condition")

    # Panel C — coefficient of variation per condition (dot plot)
    ax_c = axes_flat[2]
    cv_df = (
        df.groupby("Condition")["Spot_Count"]
        .agg(lambda x: x.std() / x.mean())
        .reset_index()
        .rename(columns={"Spot_Count": "CV"})
    )
    sns.scatterplot(
        data=cv_df,
        x="CV",
        y="Condition",
        s=100,
        color="crimson",
        marker="D",
        ax=ax_c,
    )
    ax_c.set_title("Coefficient of Variation (Spot Count)")
    ax_c.set_xlabel("CV (SD / Mean)")
    ax_c.set_ylabel("Condition")

    # Panel D — object size vs spot count scatter (coloured by condition)
    ax_d = axes_flat[3]
    sns.scatterplot(
        data=df,
        x=size_metric,
        y="Spot_Count",
        hue="Condition",
        palette="tab10",
        alpha=0.7,
        ax=ax_d,
    )
    ax_d.set_title("Object Size vs Spot Count per Condition")
    ax_d.set_xlabel(size_label)
    ax_d.set_ylabel("Spot Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[QC] Saved run summary: {out_path.name}")
