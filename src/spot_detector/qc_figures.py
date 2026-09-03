import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.colors as mcolors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde

from spot_detector.config import PipelineConfig
from spot_detector.exceptions import FatalPipelineError

logger = logging.getLogger(__name__)


@dataclass
class SpotData:
    """Derived spot coordinate data in pixel and physical units.

    Constructed from raw Spotiflow coordinates and pixel sizes. Handles both
    2D and 3D coordinate unpacking, and converts to micron units.

    Attributes:
        coordinates (np.ndarray): Raw spot coordinates from Spotiflow.
        dz (float): Z pixel size in micrometers.
        dx (float): XY pixel size in micrometers.
        is_3d (bool): Whether the pipeline is running in 3D mode.
        has_spots (bool): True if any spots were detected.
        x (np.ndarray | None): Spot x coordinates in pixels.
        y (np.ndarray | None): Spot y coordinates in pixels.
        z (np.ndarray | None): Spot z coordinates in pixels.
        x_um (np.ndarray | None): Spot x coordinates in micrometers.
        y_um (np.ndarray | None): Spot y coordinates in micrometers.
        z_um (np.ndarray | None): Spot z coordinates in micrometers.
    """

    coordinates: np.ndarray
    dz: float
    dx: float
    is_3d: bool

    has_spots: bool = field(init=False)
    x: np.ndarray | None = field(init=False)
    y: np.ndarray | None = field(init=False)
    z: np.ndarray | None = field(init=False)
    x_um: np.ndarray | None = field(init=False)
    y_um: np.ndarray | None = field(init=False)
    z_um: np.ndarray | None = field(init=False)

    def __post_init__(self):
        self.has_spots = len(self.coordinates) > 0
        if self.has_spots:
            if self.is_3d:
                self.z = np.round(self.coordinates[:, 0]).astype(int)
                self.y = np.round(self.coordinates[:, 1]).astype(int)
                self.x = np.round(self.coordinates[:, 2]).astype(int)
                self.z_um = self.z * self.dz
            else:
                self.z = np.zeros(len(self.coordinates))
                self.y = np.round(self.coordinates[:, 0]).astype(int)
                self.x = np.round(self.coordinates[:, 1]).astype(int)
                self.z_um = None
            self.x_um = self.x * self.dx
            self.y_um = self.y * self.dx
        else:
            self.x = self.y = self.z = None
            self.x_um = self.y_um = self.z_um = None


@dataclass
class ImageData:
    """Preprocessed image data for QC figure generation.

    Handles stdev projection, normalisation, and inversion of segmentation
    and spot images, and max-projects masks to 2D if needed.

    Attributes:
        segmentation_image (np.ndarray): Raw segemntation channel image (2D or 3D).
        spot_image (np.ndarray): Raw spot channel image (2D or 3D).
        masks (np.ndarray | None): Segmentation masks (2D or 3D), or None if unavailable.
        seg_inv_norm (np.ndarray): Inverted, normalised stdev projection of segmentation image.
        spots_stdev_norm (np.ndarray): Normalised stdev projection of spot image.
        masks_2d (np.ndarray | None): Max-projected 2D masks, or original masks if already 2D.
    """

    segmentation_image: np.ndarray
    spot_image: np.ndarray
    masks: np.ndarray | None

    seg_inv_norm: np.ndarray = field(init=False)
    spots_stdev_norm: np.ndarray = field(init=False)
    masks_2d: np.ndarray | None = field(init=False)

    def __post_init__(self):
        if self.segmentation_image.ndim == 3 and self.segmentation_image.shape[0] > 1:
            seg_stdev = np.std(self.segmentation_image, axis=0).astype(np.float32)
        else:
            seg_stdev = np.squeeze(self.segmentation_image).astype(np.float32)
        seg_stdev_inv = seg_stdev.max() - seg_stdev
        self.seg_inv_norm = (seg_stdev_inv - seg_stdev_inv.min()) / (
            np.ptp(seg_stdev_inv) + 1e-8
        )

        if self.spot_image.ndim == 3 and self.spot_image.shape[0] > 1:
            spots_stdev = np.std(self.spot_image, axis=0).astype(np.float32)
        else:
            spots_stdev = np.squeeze(self.spot_image).astype(np.float32)
        p_lo = float(np.percentile(spots_stdev, 0.5))
        p_hi = float(np.percentile(spots_stdev, 99.5))
        spots_clipped = np.clip(spots_stdev, p_lo, p_hi)
        self.spots_stdev_norm = (spots_clipped - p_lo) / (p_hi - p_lo + 1e-8)

        if self.masks is not None and self.masks.ndim == 3:
            self.masks_2d = np.max(self.masks, axis=0)
        else:
            self.masks_2d = self.masks


# panel helpers
def _panel_segemntation(ax: Axes, images: ImageData, dx: float) -> None:
    """Plot of segmentation image with masks overlays.
    Args:
        ax (Axes): Matplotlib axes object.
        images (ImageData): ImageData object with processed segmentation image and 2d masks.

    Raises:
        FatalPipelineError: Propagated uncaught if raised during rendering, so a
            fatal error still terminates the pipeline instead of being logged and
            replaced with a placeholder like an ordinary rendering failure.
    """
    try:
        ax.imshow(images.seg_inv_norm, cmap="gray")
        if images.masks_2d is not None and images.masks_2d.max() > 0:
            mask_overlay = np.ma.masked_where(images.masks_2d == 0, images.masks_2d)
            ax.imshow(
                mask_overlay,
                alpha=0.3,
                cmap="tab10",
                vmin=1,
                vmax=max(images.masks_2d.max(), 1),
            )
    except FatalPipelineError:
        raise
    except Exception as e:
        logger.warning(
            f"Segmentation panel rendering failed ({e}), skipping.", exc_info=True
        )
        ax.text(
            0.5,
            0.5,
            "Failed to render",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    scalebar = ScaleBar(
        dx,
        units="um",
        fixed_value=50,
        fixed_units="um",
        location="lower right",
        color="white",
        box_color="black",
        box_alpha=0.4,
        font_properties={"size": 8},
        sep=3,
        frameon=True,
    )
    ax.add_artist(scalebar)
    ax.set_title("StDev Projection + Masks")
    ax.axis("off")


def _panel_spot_detection(
    ax: Axes, images: ImageData, spots: SpotData, spot_labels: np.ndarray
) -> None:
    """Plot of spot image with detections coloured by object.
    Args:
        ax (Axes): Matplotlib axes object.
        images (ImageData): ImageData object with processed spots image.
        spots (SpotData): SpotData object with x and y spot coordinates.
        spot_labels (np.ndarray): Spot object labels.

    Raises:
        FatalPipelineError: Propagated uncaught if raised during rendering, so a
            fatal error still terminates the pipeline instead of being logged and
            replaced with a placeholder like an ordinary rendering failure.
    """
    try:
        ax.imshow(images.spots_stdev_norm, cmap="gray_r")
        if spots.has_spots and spots.x is not None and spots.y is not None:  # type: ignore
            inside_objects = spot_labels > 0
            # background elements
            ax.scatter(
                spots.x[~inside_objects],
                spots.y[~inside_objects],  # type: ignore
                color="gray",
                alpha=0.5,
                s=5,
                marker="x",
            )
            # assigned spots
            ax.scatter(
                spots.x[inside_objects],
                spots.y[inside_objects],  # type: ignore
                c=spot_labels[inside_objects],
                cmap="tab10",
                s=8,
                edgecolors="black",
                linewidths=0.3,
                alpha=0.2,
            )
    except FatalPipelineError:
        raise
    except Exception as e:
        logger.warning(
            f"Spot detection panel rendering failed ({e}), skipping.", exc_info=True
        )
        ax.text(
            0.5,
            0.5,
            "Failed to render",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_title("Spot Detections (StDev Proj)")
    ax.axis("off")


def _flow_to_rgb(flow_data: np.ndarray) -> np.ndarray:
    """Convert a Spotiflow flow field to a displayable RGB image in [0, 1].

    Spotiflow `predict()` returns `details.flow` in **channels-last** layout:
        2D model → (Y, X, 3)    — last dim = [stereographic_component, fy, fx]
        3D model → (Z, Y, X, 4) — last dim = [stereographic_component, fz, fy, fx]

    Both cases are rendered as an HSV hue-wheel image where flow direction maps
    to hue and magnitude maps to brightness. For 3D, the Z-slice with the
    highest mean XY flow magnitude is selected for display.

    Args:
        flow_data (np.ndarray): Flow array from Spotiflow details.flow.

    Returns:
        np.ndarray: RGB image of shape (Y, X, 3), dtype float32, values in [0, 1].

    Raises:
        ValueError: If flow_data has an unrecognised shape.
    """
    flow_data = flow_data.astype(np.float32)

    # Case 1: 2D channels-last (Y, X, 3) — actual Spotiflow 2D output
    # Last dim: [stereographic, fy, fx]. Use fy/fx for the hue-wheel; ignore
    # the 1st stereographic component.
    if flow_data.ndim == 3 and flow_data.shape[-1] >= 3:
        fy = flow_data[..., 1]  # (Y, X)
        fx = flow_data[..., 2]  # (Y, X)

    # Case 2: 3D channels-last (Z, Y, X, 4) — actual Spotiflow 3D output
    # Last dim: [stereographic, fz, fy, fx]. Pick the Z-slice with the
    # highest mean XY flow magnitude and render the XY components.
    elif flow_data.ndim == 4 and flow_data.shape[-1] >= 3:
        fy_vol = flow_data[..., 2]  # (Z, Y, X)
        fx_vol = flow_data[..., 3]  # (Z, Y, X)
        xy_mag = np.sqrt(fy_vol**2 + fx_vol**2)  # (Z, Y, X)
        best_z = int(np.argmax(xy_mag.mean(axis=(1, 2))))  # scalar
        fy = fy_vol[best_z]  # (Y, X)
        fx = fx_vol[best_z]  # (Y, X)

    else:
        raise ValueError(
            f"Unrecognised flow shape: {flow_data.shape}. "
            "Expected (Y,X,≥2) for 2D or (Z,Y,X,≥3) for 3D."
        )

    # HLS hue-wheel: direction → hue, magnitude → saturation + lightness.
    # Low magnitude  → lightness=1.0, saturation=0  → white background
    # High magnitude → lightness=0.5, saturation=1  → fully saturated colour
    angle = np.arctan2(fy, fx)  # (Y, X)
    hue = (angle + np.pi) / (2.0 * np.pi)  # [0, 1]
    magnitude = np.sqrt(fx**2 + fy**2)  # (Y, X)
    norm_mag = np.clip(magnitude / (magnitude.max() + 1e-8), 0.0, 1.0)
    lightness = 1.0 - 0.5 * norm_mag  # 1.0 (white) → 0.5 (vivid)
    saturation = norm_mag  # 0 (white) → 1 (vivid)

    # Vectorised HLS → RGB via the HLS→HSV identity, avoiding a Python loop.
    # HLS with L<0.5: V = L*(1+S),  S_hsv = 2*(V-L)/V
    # HLS with L≥0.5: V = L+S-L*S, S_hsv = 2*(V-L)/V
    v = np.where(
        lightness < 0.5,
        lightness * (1.0 + saturation),
        lightness + saturation - lightness * saturation,
    )
    s_hsv = np.where(v > 0, 2.0 * (v - lightness) / v, 0.0)
    hsv = np.stack([hue, s_hsv, v], axis=-1).astype(np.float32)
    return mcolors.hsv_to_rgb(hsv)


def _panel_flow(ax: Axes, flow_details: SimpleNamespace) -> None:
    """Plot of Spotiflow stereographic flow as an HSV hue-wheel image.

    Flow direction maps to hue and magnitude maps to brightness, using the
    channels-last layout returned by ``Spotiflow.predict()``:
        2D → ``details.flow`` shape ``(Y, X, 3)``
        3D → ``details.flow`` shape ``(Z, Y, X, 4)``

    Falls back to the probability heatmap when flow is ``None`` (i.e. when the
    model was trained without ``compute_flow=True`` or subpixel localisation is
    disabled). Falls back to a plain error message for any other failure.

    Args:
        ax (Axes): Matplotlib axes object.
        flow_details (SimpleNamespace): Spotiflow details object with ``flow``
            and optional ``heatmap`` attributes.

    Raises:
        FatalPipelineError: Propagated uncaught if raised during rendering, so a
            fatal error still terminates the pipeline instead of being logged and
            replaced with a placeholder like an ordinary rendering failure.
    """
    ax.set_title("Stereographic Flow")
    ax.axis("off")

    if flow_details is None:
        ax.text(
            0.5,
            0.5,
            "No Flow Data Provided",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="gray",
        )
        return

    flow_data = getattr(flow_details, "flow", None)

    # flow is None when model has compute_flow=False or subpix was disabled
    if flow_data is None:
        heatmap = getattr(flow_details, "heatmap", None)
        if heatmap is not None and isinstance(heatmap, np.ndarray):
            # show probability heatmap as a useful substitute
            display = heatmap if heatmap.ndim == 2 else heatmap.max(axis=0)
            ax.imshow(display, cmap="magma")
            ax.set_title("Probability Heatmap\n(flow unavailable — subpix disabled)")
        else:
            ax.text(
                0.5,
                0.5,
                "Flow unavailable\n(model has compute_flow=False)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray",
            )
        return

    if not isinstance(flow_data, np.ndarray):
        ax.text(
            0.5,
            0.5,
            "Flow Render Error\nflow attribute is not a numpy array",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
        )
        return

    try:
        rgb = _flow_to_rgb(flow_data)
        ax.imshow(rgb)
    except FatalPipelineError:
        raise
    except Exception as e:
        logger.warning(
            f"Constructing flow panel failed ({e}), falling back to empty panel with error message",
            exc_info=True,
        )
        ax.text(
            0.5,
            0.5,
            f"Flow Render Error\n{e!s}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
        )


def _panel_z_distribution(
    ax: Axes, images: ImageData, spots: SpotData, spot_labels: np.ndarray
) -> None:
    """Plot of Spots per z slice | Spot nearest neighbour distance.
    Args:
        ax (Axes): Matplotlib axes object.
        images (ImageData): ImageData object with processed spots image and 2d masks.
        spots (SpotData): SpotData object with a bunch of info needed.
        spot_labels (np.ndarray): Spot object labels.

    Raises:
        FatalPipelineError: Propagated uncaught if raised during rendering, so a
            fatal error still terminates the pipeline instead of being logged and
            replaced with a placeholder like an ordinary rendering failure.
    """
    if not spots.has_spots:
        ax.text(0.5, 0.5, "No Spots Detected", ha="center", va="center", color="gray")
        ax.axis("off")
        return

    elif spots.is_3d:
        try:
            # Spots per z slice stacked histogram
            unique_labels = np.unique(spot_labels)
            hist_data = []
            colors = []
            labels = []
            cmap_colors = plt.cm.tab10.colors  # type: ignore

            for lbl in unique_labels:
                mask_label = spot_labels == lbl
                if lbl == 0:
                    labels.append("Background")
                    colors.append("lightgray")
                else:
                    labels.append(f"Obj {lbl}")
                    colors.append(cmap_colors[(lbl - 1) % len(cmap_colors)])
                hist_data.append(spots.z_um[mask_label])  # type: ignore

            total_z_planes = (
                images.spot_image.shape[0] if images.spot_image.ndim == 3 else 1
            )
            bin_edges = (np.arange(total_z_planes + 1) * spots.dz).tolist()

            ax.hist(
                hist_data,
                bins=bin_edges,
                stacked=True,
                color=colors,
                label=labels,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.3,
            )

            n_obj = (
                len(np.unique(images.masks_2d)) - 1
                if images.masks_2d is not None
                else 0
            )
            if n_obj <= 10:
                ax.legend(fontsize=8, loc="upper right")

        except FatalPipelineError:
            raise
        except Exception as e:
            logger.warning(
                f"Constructing z-distribution figure failed ({e}), skipping.",
                exc_info=True,
            )
            ax.text(
                0.5,
                0.5,
                "Failed to render",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title("Z-Distribution Profile (µm)")
        ax.set_xlabel("Z-Depth Position (µm)")
        ax.set_ylabel("Spot Count")
        ax.grid(True, linestyle=":", alpha=0.5)

    else:
        try:
            # Spot nearest neighbour distance
            if len(spots.coordinates) < 2:
                ax.text(
                    0.5,
                    0.5,
                    "Insufficient Spots for NND",
                    ha="center",
                    va="center",
                    color="gray",
                )
                ax.grid(False)
            else:
                y_um_f = spots.coordinates[:, 0] * spots.dx  # float Y in µm
                x_um_f = spots.coordinates[:, 1] * spots.dx  # float X in µm
                spatial_xy_um = np.column_stack((y_um_f, x_um_f))
                tree = KDTree(spatial_xy_um)
                distances, _ = tree.query(spatial_xy_um, k=2)
                nnd_um = distances[:, 1]

                ax.hist(
                    nnd_um,
                    bins="auto",
                    density=True,
                    color="#FF66CC",
                    alpha=0.4,
                    edgecolor="#FF66CC",
                )
                try:
                    kde = gaussian_kde(nnd_um)
                    x_vals = np.linspace(nnd_um.min(), nnd_um.max(), 200)
                    ax.plot(x_vals, kde(x_vals), color="#FF66CC", linewidth=1.5)
                except FatalPipelineError:
                    raise
                except Exception:
                    logger.warning(
                        "Constructing KDE figure failed, skipping.", exc_info=True
                    )
        except FatalPipelineError:
            raise
        except Exception as e:
            logger.warning(
                f"Constructing NND figure failed ({e}), skipping.", exc_info=True
            )
            ax.text(
                0.5,
                0.5,
                "Failed to render",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title("Spot Proximity Distribution")
        ax.set_xlabel("Nearest Neighbor Distance (µm)")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle=":", alpha=0.5)


def _panel_ecdf(
    ax: Axes,
    spots: SpotData,
    spot_labels: np.ndarray,
    flow_details: SimpleNamespace,
    config: PipelineConfig,
) -> None:
    """Spotiflow probability score ECDF (inside vs background)
    Args:
        ax (Axes): Matplotlib axes object.
        spots (SpotData): SpotData object.
        spot_labels (np.ndarray): Spot object labels.
        flow_details (SimpleNamespace): Spotiflow probability score object.
        config (PipelineConfig): Config dictionary containg 'prob_thresh' value used for spotiflow detection.

    Raises:
        FatalPipelineError: Propagated uncaught if raised during rendering, so a
            fatal error still terminates the pipeline instead of being logged and
            replaced with a placeholder like an ordinary rendering failure.
    """
    if not spots.has_spots:
        ax.text(0.5, 0.5, "No Spots Detected", ha="center", va="center", color="gray")
        ax.axis("off")
    else:
        try:
            prob_arr = np.array(flow_details.prob)
            inside = spot_labels > 0

            for mask, label, color, ls in [
                (inside, "Inside object", "#D4537E", "-"),
                (~inside, "Background", "#888780", "--"),
            ]:
                subset = np.sort(prob_arr[mask])
                if len(subset) == 0:
                    continue
                ecdf_y = np.arange(1, len(subset) + 1) / len(subset)
                ax.step(
                    subset,
                    ecdf_y,
                    where="pre",
                    color=color,
                    linestyle=ls,
                    linewidth=1.5,
                    label=f"{label} (n={len(subset)})",
                )
            ax.axvline(
                config.detection.prob_thresh,
                color="gray",
                linewidth=0.8,
                linestyle=":",
                alpha=0.6,
                label=f"thresh={config.detection.prob_thresh}",
            )
        except FatalPipelineError:
            raise
        except Exception as e:
            logger.warning(
                f"ECDF panel rendering failed ({e}), skipping.", exc_info=True
            )
            ax.text(
                0.5,
                0.5,
                "Failed to render",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Spotiflow probability score")
        ax.set_ylabel("Cumulative fraction of spots")
        ax.set_title("Detection Confidence (ECDF)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.4)


def _panel_spotmap(ax: Axes, images: ImageData, spots: SpotData) -> None:
    """Object contours + XY spotmap coloured by z-depth (3D) or object label (2D).

    Coordinates and axis limits are in micrometres. Equal aspect ratio is enforced
    to prevent distortion from non-square FOVs.

    Args:
        ax (Axes): Matplotlib axes object.
        images (ImageData): ImageData object.
        spots (SpotData): SpotData object.

    Raises:
        FatalPipelineError: Propagated uncaught if raised during rendering, so a
            fatal error still terminates the pipeline instead of being logged and
            replaced with a placeholder like an ordinary rendering failure.
    """
    dx = spots.dx
    img_h = images.segmentation_image.shape[-2]
    img_w = images.segmentation_image.shape[-1]

    if images.masks_2d is not None:
        x_coords = np.arange(img_w) * dx
        y_coords = np.arange(img_h) * dx
        unique_labels = np.unique(images.masks_2d)
        levels = unique_labels[unique_labels > 0] - 0.5
        if len(levels) > 0:
            ax.contour(
                x_coords,
                y_coords,
                images.masks_2d,
                levels=levels,
                colors="black",
                linewidths=0.8,
                alpha=0.7,
            )

    if spots.has_spots and spots.x is not None and spots.y is not None:  # type: ignore
        x_plot = spots.x * dx  # type: ignore
        y_plot = spots.y * dx  # type: ignore

        try:
            if spots.is_3d:
                sc = ax.scatter(
                    x_plot,
                    y_plot,
                    c=spots.z_um,
                    cmap="turbo",  # type: ignore
                    s=12,
                    edgecolors="black",
                    linewidths=0.15,
                    alpha=0.85,
                )
                fig = ax.get_figure()
                if fig is not None:
                    cbar = fig.colorbar(
                        sc, ax=ax, orientation="vertical", pad=0.02, shrink=0.7
                    )
                    cbar.set_label("Z depth (µm)", fontsize=8)
                    cbar.ax.tick_params(labelsize=8)
            else:
                ax.scatter(
                    x_plot,
                    y_plot,
                    color="#4FC3F7",
                    s=10,
                    edgecolors="black",
                    linewidths=0.15,
                    alpha=0.85,
                )  # type: ignore
        except FatalPipelineError:
            raise
        except Exception as e:
            logger.warning(
                f"Spotmap panel rendering failed ({e}), skipping.", exc_info=True
            )
            ax.text(
                0.5,
                0.5,
                "Failed to render",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    ax.set_xlim(0, img_w * dx)
    ax.set_ylim(img_h * dx, 0)  # y-axis inverted to match image orientation
    ax.set_aspect("equal")


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
