class MicroscopyQC:
    def __init__(self, figsize=(15, 10), dpi=300):
        plt.ioff()
        self.figsize = figsize
        self.dpi = dpi

    def create_qc_figure(self,
                         segmentation_image,
                         spot_std,
                         coordinates,
                         corrected_spots,
                         masks,
                         flow_details,
                         condition,
                         image_num,
                         output_path,
                         pixel_size_um,
                         percentile_range=(1, 99.8),
                         downsample_factor=8,
                         mask_alpha=0.3):
        """
        QC figure using the 'constrained' layout engine for robust, automatic spacing.
        """
        # Create figure using the 'constrained' layout engine
        # This one argument replaces plt.tight_layout() and plt.subplots_adjust()
        fig, axes = plt.subplots(2, 3, figsize=self.figsize,
                                 facecolor='white',
                                 layout='constrained')

        # Calculate intensity limits
        seg_clim = tuple(np.percentile(segmentation_image[::downsample_factor, ::downsample_factor], percentile_range))
        spot_clim = tuple(np.percentile(spot_std[::downsample_factor, ::downsample_factor], percentile_range))
        corrected_clim = tuple(np.percentile(corrected_spots[::downsample_factor, ::downsample_factor], percentile_range))

        axes = axes.flatten()
        for ax in axes:
            ax.axis('off')

        # Panel 1: Segmentation Channel
        axes[0].imshow(segmentation_image, cmap="gray", clim=seg_clim)
        axes[0].set_title('Segmentation Channel', fontsize=12)

        # Panel 2: Spots Channel
        axes[1].imshow(spot_std, cmap="magma", clim=spot_clim)
        axes[1].set_title('Spots Channel', fontsize=12)

        # Panel 3: Spots + Detections
        axes[2].imshow(spot_std, cmap="magma", clim=spot_clim)
        coordinates = np.array(coordinates)
        if len(coordinates) > 0:
            axes[2].scatter(coordinates[:, 1], coordinates[:, 0],
                            facecolors='none', edgecolors='orange', s=10, linewidths=1)
        axes[2].set_title(f'Detections ({len(coordinates)} spots)', fontsize=12)

        # Panel 4: Segmentation + Masks
        axes[3].imshow(segmentation_image, cmap='gray', clim=seg_clim)
        if masks is not None:
            mask_overlay = np.ma.masked_where(masks == 0, masks)
            axes[3].imshow(mask_overlay, alpha=mask_alpha, cmap='tab10', vmin=1)
        axes[3].set_title(f'Masks ({masks.max() if masks is not None else 0} ROIs)', fontsize=12)

        # Panel 5: TopHat Filtered
        axes[4].imshow(corrected_spots, cmap="magma", clim=corrected_clim)
        axes[4].set_title('TopHat Filtered', fontsize=12)

        # Panel 6: Stereographic Flow
        if flow_details is not None:
            try:
                flow_viz = 0.5 * (1 + getattr(flow_details, 'flow', flow_details))
                axes[5].imshow(flow_viz)
            except Exception as e:
                axes[5].text(0.5, 0.5, 'No Flow Data', ha='center', va='center',
                             transform=axes[5].transAxes, fontsize=12, color='gray')
        axes[5].set_title('Stereographic Flow', fontsize=12)

        # Add scale bar
        if pixel_size_um > 0:
            scalebar = ScaleBar(pixel_size_um, units='um', location='lower right',
                                box_alpha=0.8, color='white', box_color='black')
            axes[0].add_artist(scalebar)

        # Set main title
        fig.suptitle(f'{condition} - Image {image_num}', fontsize=16)

        # Save figure
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', pad_inches=0.1)
        
        plt.close(fig)