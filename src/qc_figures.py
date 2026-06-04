from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde
from types import SimpleNamespace

# %% Helper figures

@dataclass
class SpotData:
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
                self.z = np.round(self.coordinates[:,0]).astype(int)
                self.y = np.round(self.coordinates[:,1]).astype(int)
                self.x = np.round(self.coordinates[:,2]).astype(int)
                self.z_um = self.z * self.dz
            else:
                self.z = np.zeros(len(self.coordinates))
                self.y = np.round(self.coordinates[:,0]).astype(int)
                self.x = np.round(self.coordinates[:,1]).astype(int)                
            self.x_um = self.x * self.dx
            self.y_um = self.y * self.dx
@dataclass
class ImageData:
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
        self.seg_inv_norm = (seg_stdev_inv - seg_stdev_inv.min()) / (np.ptp(seg_stdev_inv) + 1e-8)
        
        if self.spot_image.ndim == 3 and self.spot_image.shape[0] > 1:
            spots_stdev = np.std(self.spot_image, axis=0).astype(np.float32)
        else:
            spots_stdev = np.squeeze(self.spot_image).astype(np.float32)
        self.spots_stdev_norm = (spots_stdev - spots_stdev.min()) / (np.ptp(spots_stdev) + 1e-8)  
        
        if self.masks is not None and self.masks.ndim == 3:
            self.masks_2d = np.max(self.masks, axis=0)
        else:
            self.masks_2d = self.masks

# panel helpers
def _panel_segemntation(ax: Axes, images: ImageData) -> None:
    """Plot of segmentation image with masks overlays.
    Args:
        ax (Axes): Matplotlib axes object.
        images (ImageData): ImageData object with processed segmentation image and 2d masks.
    """
    ax.imshow(images.seg_inv_norm, cmap="gray")
    if images.masks_2d is not None:
        mask_overlay = np.ma.masked_where(images.masks_2d == 0, images.masks_2d)
        ax.imshow(mask_overlay, alpha = 0.3, cmap='tab10', vmin=1)
    ax.set_title("StDev Projection + Masks")
    ax.axis("off")

def _panel_spot_detection(ax: Axes, images: ImageData, spots: SpotData, spot_labels: np.ndarray) -> None:
    """Plot of spot image with detections coloured by object.
    Args:
        ax (Axes): Matplotlib axes object.
        images (ImageData): ImageData object with processed spots image.
        spots (SpotData): SpotData object with x and y spot coordinates.
        spot_labels (np.ndarray): Spot object labels.
    """
    ax.imshow(images.spots_stdev_norm, cmap="magma")
    if spots.has_spots and spots.x is not None and spots.y is not None: # type: ignore
        inside_objects = spot_labels > 0
        # background elements
        ax.scatter(
            spots.x[~inside_objects], spots.y[~inside_objects],  # type: ignore
            color="white", alpha=0.3, s=6, marker="x"
        )
        # assigned spots
        ax.scatter(
            spots.x[inside_objects], spots.y[inside_objects],  # type: ignore
            c=spot_labels[inside_objects], cmap="tab10",
            s=15, edgecolors='white', linewidths=0.3, alpha=0.9
        )
    ax.set_title("Spot Detections (StDev Proj)")
    ax.axis("off")

def _panel_flow(ax: Axes, flow_details: SimpleNamespace) -> None:
    """Plot of spotiflow flows.
    Args:
        ax (Axes): Matplotlib axes object.
        flow_details (SimpleNamespace): Spotiflow flow field for visualization.
    """
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
                    ax.imshow(np.clip(flow_viz, 0, 1))
                elif flow_data.ndim == 3 and flow_data.shape[0] == 3:
                    rgb_flow = np.moveaxis(flow_data, 0, -1)
                    flow_viz = 0.5 * (1 + rgb_flow)
                    ax.imshow(np.clip(flow_viz, 0, 1))
                else:
                    flat_view = np.max(flow_data, axis=0) if flow_data.ndim >= 3 else flow_data
                    ax.imshow(flat_view, cmap="viridis")
            else:
                ax.text(
                    0.5, 0.5, "Non-Array Flow Format", 
                    ha = "center", va = "center",
                    transform=ax.transAxes, 
                    fontsize=11, color='gray')
        except Exception as e:
            ax.text(
                0.5, 0.5, f"Flow Render Error\n{str(e)}", 
                ha = "center", va = "center",
                transform=ax.transAxes, 
                fontsize=11, color='gray')
    else:
        ax.text(
            0.5, 0.5, "No Flow Data Provided", 
            ha = "center", va = "center",
            transform=ax.transAxes, 
            fontsize=11, color='gray')
    ax.set_title('Stereographic Flow')
    ax.axis("off")

def _panel_z_distribution(ax: Axes, images: ImageData, spots: SpotData, spot_labels: np.ndarray) -> None:
    """Plot of Spots per z slice | Spot nearest neighbour distance.
    Args:
        ax (Axes): Matplotlib axes object.
        images (ImageData): ImageData object with processed spots image and 2d masks.
        spots (SpotData): SpotData object with a bunch of info needed.
        spot_labels (np.ndarray): Spot object labels.
    """
    if not spots.has_spots:
        ax.text(0.5, 0.5, "No Spots Detected",
                ha="center", va="center", color="gray")
        ax.axis("off")
    elif spots.is_3d:
        # Spots per z slice stacked histogram
        unique_labels = np.unique(spot_labels)
        hist_data = []
        colors = []
        labels = []
        cmap_colors = plt.cm.tab10.colors  # type: ignore
        
        for lbl in unique_labels:
            mask_label = (spot_labels == lbl)
            if lbl == 0:
                labels.append("Background")
                colors.append("lightgray")
            else:
                labels.append(f"Obj {lbl}")
                colors.append(cmap_colors[(lbl - 1) % len(cmap_colors)])               
            hist_data.append(spots.z_um[mask_label]) # type: ignore
        
        total_z_planes = images.spot_image.shape[0] if images.spot_image.ndim == 3 else 1
        bin_edges = (np.arange(total_z_planes + 1) * spots.dz).tolist()
        
        ax.hist(hist_data, bins=bin_edges, stacked=True,
                color=colors, label=labels, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_title("Z-Distribution Profile (µm)")
        ax.set_xlabel("Z-Depth Position (µm)")
        ax.set_ylabel("Spot Count")
        ax.grid(True, linestyle=":", alpha=0.5)
        
        n_obj = len(np.unique(images.masks_2d)) - 1 if images.masks_2d is not None else 0
        if n_obj < 10:
            ax.legend(fontsize=8, loc="upper right")
    else:
        # Spot nearest neighbour distance
        if len(spots.coordinates) < 2:
            ax.text(0.5, 0.5, "Insufficient Spots for NND",
                    ha="center", va="center", coor="gray")
            ax.grid(False)
        else:
            spatial_xy_um = np.column_stack((spots.x_um, spots.y_um)) # type: ignore
            tree = KDTree(spatial_xy_um)
            distances, _ = tree.query(spatial_xy_um, k=2)
            nnd_um = distances[:,1]
            
            ax.hist(nnd_um, bins="auto", density=True,
                    color="#FF66CC", alpha=0.4, edgecolor="#FF66CC")
            try:
                kde = gaussian_kde(nnd_um)
                x_vals = np.linspace(nnd_um.min(), nnd_um.max(), 200)
                ax.plot(x_vals, kde(x_vals), color="#FF66CC", linewidth=1.5)
                
            except Exception:
                pass
        ax.set_title("Spot Proximity Distribution")
        ax.set_xlabel("Nearest Neighbor Distance (µm)")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle=":", alpha=0.5)    

def _panel_ecdf(ax: Axes, spots: SpotData, spot_labels: np.ndarray, flow_details: SimpleNamespace, config: dict) -> None:
    """Spotiflow probability score ECDF (inside vs background)
    Args:
        ax (Axes): Matplotlib axes object.
        spots (SpotData): SpotData object.
        spot_labels (np.ndarray): Spot object labels.
        flow_details (SimpleNamespace): Spotiflow probability score object.
        config (dict): Config dictionary containg 'prob_thresh' value used for spotiflow detection.
    """
    if not spots.has_spots:
        ax.text(0.5, 0.5, "No Spots Detected",
                ha="center", va="center", coor="gray")
        ax.axis("off")
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
            ax.step(subset, ecdf_y, where="pre",
                    color=color, linestyle=ls, linewidth=1.5,
                    label=f"{label} (n={len(subset)})")
        ax.axvline(config["prob_thresh"], color="gray",
                linewidth=0.8, linestyle=":", alpha=0.6,
                label=f"thresh={config['prob_thresh']}")
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xlabel("Spotiflow probability score")
        ax.set_ylabel("Cumulative fraction of spots")
        ax.set_title("Detection Confidence (ECDF)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.4)    

def _panel_spotmap(ax: Axes, images: ImageData, spots: SpotData) -> None:
    """Plot of Outlines + xy spotmap (coloured by z depth)
    Args:
        ax (Axes): Matplotlib axes object.
        images (ImageData): ImageData object.
        spots (SpotData): SpotData object.
    """
    if images.masks_2d is not None:
        ax.contour(images.masks_2d, levels=np.unique(images.masks_2d)[1:] - 0.5,
                    colors="white", linewidths=0.5, alpha=0.6)
    if spots.has_spots and spots.x is not None and spots.y is not None: # type: ignore
        if spots.is_3d:
            sc = ax.scatter(spots.x, spots.y, c=spots.z_um, cmap="turbo", # type: ignore
                            s=12, edgecolors="black", linewidths=0.15, alpha=0.85)
            fig = ax.get_figure()
            if fig is not None:
                cbar = fig.colorbar(sc, ax=ax, orientation="vertical",
                                    pad=0.02, shrink=0.7)
                cbar.ax.tick_params(labelsize=8)
        else:
            ax.scatter(spots.x, spots.y, color="#00FFCC", s=10, alpha=0.7) # type: ignore
    ax.set_xlim(0, images.segmentation_image.shape[-1])
    ax.set_ylim(images.segmentation_image.shape[-2], 0)
    ax.set_title("XY Spatial Spotmap (µm)")
    ax.axis("off")

def make_qc_figure(
    coordinates: np.ndarray,
    condition: str,
    scene: int,
    mode: str,
    out_path: Path,
    segmentation_image: np.ndarray,
    spots_image: np.ndarray,
    masks: np.ndarray,
    flow_details: SimpleNamespace,
    spot_labels: np.ndarray,
    dx: float,
    dz: float,
    config: dict
) -> None:
    """
        Generates a 2x3 panel figure showing:
        1. Segmentation channel with mask overlays
        2. Spots channel with detections coloured by object
        3. Spotiflow flows
        4. Spots per z slice | Spot nearest neighbour distance
        5. Spotiflow probability score ECDF (inside vs background)
        6. Object contours + xy spotmap (coloured by z depth)

    Args:
        df (pd.DataFrame): Combined scene dataframe of 2d or 3d analysis.
        condition (str): Experimental condition name.
        mode (str): 2d or 3d mode.
        out_path (Path): Output path.
    """
    is_3d = mode == "3d"
    n_obj = len(np.unique(masks)) - 1

    images = ImageData(
        segmentation_image=segmentation_image,
        spot_image=spots_image,
        masks=masks
    )
    spots = SpotData(
        coordinates=coordinates,
        dz=dz,
        dx=dx,
        is_3d=is_3d
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()
    fig.suptitle(
        f"{condition} - Scene {scene:02d} - [{mode.upper()} mode] - "
        f"{n_obj} object(s) - {len(coordinates)} spot(s)",
        fontsize=12, fontweight="bold")

    _panel_segemntation(ax=axes_flat[0], images=images)
    _panel_spot_detection(ax=axes_flat[1], images=images, spots=spots, spot_labels=spot_labels)
    _panel_flow(ax=axes_flat[2], flow_details=flow_details)
    _panel_z_distribution(ax=axes_flat[3], images=images, spots=spots, spot_labels=spot_labels)
    _panel_ecdf(ax=axes_flat[4], spots=spots, spot_labels=spot_labels, flow_details=flow_details, config=config)
    _panel_spotmap(ax=axes_flat[5], images=images, spots=spots)

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