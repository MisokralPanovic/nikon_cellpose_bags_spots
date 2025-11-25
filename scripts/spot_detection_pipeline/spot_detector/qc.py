from pathlib import Path
from typing import Tuple
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib_scalebar.scalebar import ScaleBar

class MicroscopyQC:
    """Generates quality control figures for the segmentation and spot detection pipeline. Creates a 2x3 grid of diagnostic panels showing segmentation, spot detection,
    and analysis results.
    
    Args:
        figsize: Figure size in inches (width, height).
        dpi: Resolution in dots per inch for saved figures.

    Examples
    --------
    >>> qc = MicroscopyQC(figsize=(10, 8), dpi=300)
    >>> qc.create_qc_figure(
    ...     segmentation_image=seg_img,
    ...     spots=spots_img,
    ...     coordinates=coords,
    ...     corrected_spots=corrected,
    ...     masks=masks,
    ...     flow_details=flow,
    ...     condition="control",
    ...     image_num=1,
    ...     output_path=Path("qc.png"),
    ...     pixel_size_um=0.65
    ... )
    """
    def __init__(self, figsize: Tuple[int, int] = (15, 10), dpi: int = 300):
        """Initial QC figure generator.

        Args:
            figsize (Tuple[int, int], optional): Figure size in inches (width, height). Defaults to (15, 10).
            dpi (int, optional): Resolution in dots per inch for saved figures. Defaults to 300.
        """
        plt.ioff()
        self.figsize = figsize
        self.dpi = dpi

    def create_qc_figure(
        self,
        segmentation_image: np.ndarray,
        spots: np.ndarray,
        coordinates: np.ndarray | list,
        corrected_spots: np.ndarray,
        masks: np.ndarray,
        flow_details: SimpleNamespace,
        condition: str,
        image_num: int,
        output_path: Path,
        pixel_size_um: float,
        percentile_range: Tuple[float, float] = (1, 99.8),
        downsample_factor: int = 8,
        mask_alpha: float = 0.3
        ) -> None:
        """Create and save comprehensive QC figure.

        Generates a 2x3 panel figure showing:
        1. Segmentation channel (grayscale)
        2. Spots channel (magma colormap)
        3. Spots with detected coordinates overlaid
        4. Segmentation with mask overlays
        5. TopHat filtered spots
        6. Cellpose flow field (stereographic projection)

        Args:
            segmentation_image (np.ndarray): 2D grayscale image used for segmentation.
            spots (np.ndarray): 2D spots channel image.
            coordinates (np.ndarray | list): Detected spot coordinates, shape (N, 2) where each row is (y, x).
            corrected_spots (np.ndarray): TopHat filtered spots image.
            masks (np.ndarray): Segmentation masks with integer labels (0=background).
            flow_details (SimpleNamespace): Spotiflow flow field for visualization.
            condition (str): Experimental condition name.
            image_num (int): Field of view number.
            output_path (Path): Path where figure will be saved.
            pixel_size_um (float): Pixel size in micrometers for scale bar.
            percentile_range (Tuple[float, float], optional): Min and max percentiles for intensity scaling. Defaults to (1, 99.8).
            downsample_factor (int, optional): Factor to downsample images for percentile calculation. Defaults to 8.
            mask_alpha (float, optional): Transparency of mask overlay (0=transparent, 1=opaque). Defaults to 0.3.
            
        Notes
        -----
        The figure is saved and closed automatically to prevent memory leaks.
        Percentile calculation is done on downsampled images for speed.
        """
        fig, axes = plt.subplots(2, 3, figsize=self.figsize, facecolor='white', layout='constrained')

        # Calculate intensity limits
        seg_clim = self._calculate_intensity_limits(segmentation_image, percentile_range, downsample_factor)
        spot_clim = self._calculate_intensity_limits(spots, percentile_range, downsample_factor)
        corrected_clim = self._calculate_intensity_limits(corrected_spots, percentile_range, downsample_factor)

        # Flatten axes array for easier indexing
        axes = axes.flatten()
        
        # Turn off axis ticks/labels for all panels        
        for ax in axes:
            ax.axis('off')

        # Panel 1: Spots Channel
        axes[0].imshow(spots, cmap="magma", clim=spot_clim)
        axes[0].set_title('Spots Channel', fontsize=12)

        # Panel 2: TopHat Filtered
        axes[1].imshow(corrected_spots, cmap="magma", clim=corrected_clim)
        axes[1].set_title('TopHat Filtered', fontsize=12)

        # Panel 3: Stereographic Flow
        self._plot_flow_field(axes[2], flow_details)
        axes[2].set_title('Stereographic Flow', fontsize=12)

        # Panel 4: Segmentation Channel
        axes[3].imshow(segmentation_image, cmap="gray", clim=seg_clim)
        axes[3].set_title('Segmentation Channel', fontsize=12)

        # Panel 5: Segmentation + Masks
        axes[4].imshow(segmentation_image, cmap='gray', clim=seg_clim)
        if masks is not None:
            mask_overlay = np.ma.masked_where(masks == 0, masks)
            axes[4].imshow(mask_overlay, alpha=mask_alpha, cmap='tab10', vmin=1)
        axes[4].set_title(f'Masks ({masks.max() if masks is not None else 0} ROIs)', fontsize=12)

        # Panel 6: Spots + Detections
        axes[5].imshow(spots, cmap="magma", clim=spot_clim)
        coordinates = np.array(coordinates)
        if len(coordinates) > 0:
            axes[5].scatter(
                coordinates[:, 1], coordinates[:, 0],
                facecolors='none', edgecolors='orange',
                s=10, linewidths=1)
        axes[5].set_title(f'Detections ({len(coordinates)} spots)', fontsize=12)

        # Add scale bar
        if pixel_size_um > 0:
            scalebar = ScaleBar(
                pixel_size_um, units='um', 
                location='lower right', box_alpha=0.8, 
                color='white', box_color='black')
            axes[3].add_artist(scalebar)

        # Set main title
        fig.suptitle(f'{condition} - Image {image_num}', fontsize=16)

        # Save figure
        fig.savefig(
            output_path, 
            dpi=self.dpi, 
            bbox_inches='tight',
            facecolor='white', 
            pad_inches=0.1)
        plt.close(fig)

    @staticmethod        
    def _calculate_intensity_limits(
        image: np.ndarray, 
        percentile_range: Tuple[float, float], 
        downsample_factor: int
        ) -> Tuple[float, float]:
        """Calculate intensity limits for display using percentiles.

        Args:
            image (np.ndarray): Input image.
            percentile_range (Tuple[float, float]): Lower and upper percentiles.
            downsample_factor (int): Factor to downsample for speed.

        Returns:
            Tuple[float, float]: Tuple of (min, max) intensity values.
        """
        downsampeled = image[::downsample_factor, ::downsample_factor]
        return tuple(np.percentile(downsampeled, percentile_range))
    
    @staticmethod
    def _plot_flow_field(ax: Axes, flow_details: SimpleNamespace) -> None:
        """ Plot Spotiflow stereographic flow field

        Args:
            ax (Axes): Matplotlib axis to plot on.
            flow_details (SimpleNamespace):  SimpleNamespace from Spotiflow containing 'flow' attribute
        """
        if flow_details is None:
            ax.text(
                0.5, 0.5, 'No Flow Data', 
                ha='center', va='center',
                transform=ax.transAxes, 
                fontsize=12, 
                color='gray'
            )
            return
        
        try:
            # Access flow attribute from SimpleNamespace
            # Normalize flow to [0, 1] range for visualization
            # Spotiflow flows are in [-1, 1], so 0.5 * (1 + flow) maps to [0, 1]
            if hasattr(flow_details, 'flow'):
                flow_viz = 0.5 * (1 + flow_details.flow)
                ax.imshow(flow_viz)
            else:
                # If no flow attribute, show placeholder
                ax.text(
                    0.5, 0.5, 'No Flow Attribute', 
                    ha='center', va='center',
                    transform=ax.transAxes, 
                    fontsize=12, 
                    color='gray'
                )
        except Exception as e:
            # If anything goes wrong, show error
            ax.text(
                0.5, 0.5, f'Flow Visualization Failed\n{type(e).__name__}', 
                ha='center', va='center',
                transform=ax.transAxes, 
                fontsize=10, 
                color='red'
            )   

def create_qc_figure(
    segmentation_image: np.ndarray,
    spots: np.ndarray,
    coordinates: np.ndarray | list,
    corrected_spots: np.ndarray,
    masks: np.ndarray,
    flow_details: SimpleNamespace,
    condition: str,
    image_num: int,
    output_path: Path,
    pixel_size_um: float,
    percentile_range: Tuple[float, float] = (1, 99.8),
    downsample_factor: int = 8,
    mask_alpha: float = 0.3,
    save_figures: bool = True,
    dpi: int = 300,
    figsize: Tuple[float, float] = (10.0, 8.0)
    ) -> None:
    """Convenience function to create QC figure without instantiating class.

    Generates a 2x3 panel figure showing:
    1. Segmentation channel (grayscale)
    2. Spots channel (magma colormap)
    3. Spots with detected coordinates overlaid
    4. Segmentation with mask overlays
    5. TopHat filtered spots
    6. Cellpose flow field (stereographic projection)

    Args:
        segmentation_image (np.ndarray): 2D grayscale image used for segmentation.
        spots (np.ndarray): 2D spots channel image.
        coordinates (np.ndarray | list): Detected spot coordinates, shape (N, 2) where each row is (y, x).
        corrected_spots (np.ndarray): TopHat filtered spots image.
        masks (np.ndarray): Segmentation masks with integer labels (0=background).
        flow_details (SimpleNamespace): Spotiflow flow field for visualization.
        condition (str): Experimental condition name.
        image_num (int): Field of view number.
        output_path (Path): Path where figure will be saved.
        pixel_size_um (float): Pixel size in micrometers for scale bar.
        percentile_range (Tuple[float, float], optional): Min and max percentiles for intensity scaling. Defaults to (1, 99.8).
        downsample_factor (int, optional): Factor to downsample images for percentile calculation. Defaults to 8.
        mask_alpha (float, optional): Transparency of mask overlay (0=transparent, 1=opaque). Defaults to 0.3.
        save_figures (bool): Whether to save figure (if False, function returns immediately). Defaults to True.
        dpi (int): Figure resolution. Defaults to 300.
        figsize (Tuple[float, float]): Figure size (width, height) in inches. Defaults to (10, 8)
        
    Notes
    -----
    The figure is saved and closed automatically to prevent memory leaks.
    Percentile calculation is done on downsampled images for speed.
    """
    if not save_figures:
        return
    
    qc = MicroscopyQC(figsize=figsize, dpi=dpi)
    qc.create_qc_figure(
        segmentation_image=segmentation_image,
        spots=spots,
        coordinates=coordinates,
        corrected_spots=corrected_spots,
        masks=masks,
        flow_details=flow_details,
        condition=condition,
        image_num=image_num,
        output_path=output_path,
        pixel_size_um=pixel_size_um,
        percentile_range=percentile_range,
        downsample_factor=downsample_factor,
        mask_alpha=mask_alpha
    )