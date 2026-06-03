from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde

# %% Helper figures

# def preprocessing if deemed helpful

# panel helpers
def _panel_segemntation(
    ax,
    seg_inv_norm,
    masks_2d
):
    """_summary_

    Args:
        ax (_type_): _description_
        seg_inv_norm (_type_): _description_
        masks_2d (_type_): _description_
    """
    ax.imshow(seg_inv_norm, cmap="gray")
    if masks_2d is not None:
        mask_overlay = np.ma.masked_where(masks_2d == 0, masks_2d)
        ax.imshow(mask_overlay, alpha = 0.3, cmap='tab10', vmin=1)
    ax.set_title("StDev Projection + Masks")
    ax.axis("off")

def _panel_spot_detection(
    ax,
    spots_stdev_norm,
    spot_labels,
    has_spots,
    spots_x,
    spots_y
):
    ax.imshow(spots_stdev_norm, cmap="magma")
    if has_spots and spots_x is not None and spots_y is not None: # type: ignore
        inside_objects = spot_labels > 0
        # background elements
        ax.scatter(
            spots_x[~inside_objects], spots_y[~inside_objects],  # type: ignore
            color="white", alpha=0.3, s=6, marker="x"
        )
        # assigned spots
        ax.scatter(
            spots_x[inside_objects], spots_y[inside_objects],  # type: ignore
            c=spot_labels[inside_objects], cmap="tab10",
            s=15, edgecolors='white', linewidths=0.3, alpha=0.9
        )
    ax.set_title("Spot Detections (StDev Proj)")
    ax.axis("off")

def _panel_flow(...): ...

def _panel_z_distribution(...): ...

def _panel_ecdf(...): ...

def _panel_spotmap(...): ...

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
        ax_d.text(0.5, 0.5, "No Spots Detected",
                ha="center", va="center", coor="gray")
        ax_d.axis("off")
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