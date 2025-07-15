#!/usr/bin/env python3
"""
Detection Pipeline for Cellpose SAM Segmentation and Spot Analysis
Converts Jupyter notebook to a cluster-ready Python script
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import seaborn as sns
import os
import sys
import argparse
import logging
from pathlib import Path
import warnings
from datetime import datetime
from tqdm import tqdm
import json

# Import required libraries
from cellpose import models
import skimage
import skimage.feature
import skimage.filters
import skimage.segmentation
import nd2
import xarray
import openpyxl

class DetectionPipeline:
    """Main pipeline class for spot detection and analysis"""
    
    def __init__(self, config_file=None, **kwargs):
        """Initialize pipeline with configuration"""
        self.setup_logging()
        self.load_configuration(config_file, **kwargs)
        self.setup_directories()
        self.results = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('detection_pipeline.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_configuration(self, config_file=None, **kwargs):
        """Load configuration from file or kwargs"""
        # Default configuration
        # If running from scripts/ directory, go up one level to experiment folder
        script_dir = Path(__file__).parent if __file__ else Path.cwd()
        if script_dir.name == 'scripts':
            experiment_folder = script_dir.parent
        else:
            experiment_folder = script_dir
            
        default_config = {
            'experiment_folder': experiment_folder,
            'thickness_um': 20,
            'channels_params': {
                'brightfield': 0,
                'spots': 1,
                'bags': 2
            },
            'spot_channel_params': {
                'FAM': False,
                'TAMRA': True
            },
            'segmentation_params': {
                'diameter': 676,
                'model_name': 'cpsam_20x_downsampeled_20250630',
                'gpu': True
            },
            'spot_detection_params': {
                'min_distance': 3,
                'threshold_abs': 800
            },
            'background_sigma_params': {
                'segmentation_sigma': 40,
                'spot_sigma': 100
            }
        }
        
        # Load from config file if provided
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            default_config.update(file_config)
            
        # Override with kwargs
        for key, value in kwargs.items():
            if key in default_config:
                default_config[key] = value
                
        # Set as instance attributes
        for key, value in default_config.items():
            setattr(self, key, value)
            
        self.logger.info(f"Configuration loaded successfully")
        
    def setup_directories(self):
        """Setup input and output directories"""
        self.experiment_folder = Path(self.experiment_folder).resolve()
        self.raw_data_folder = self.experiment_folder / 'raw_data'
        self.figures_folder = self.experiment_folder / 'figures'
        self.processed_data_folder = self.experiment_folder / 'processed_data'
        self.flatfield_map_folder = self.raw_data_folder
        
        # Verify required directories exist
        if not self.raw_data_folder.exists():
            raise FileNotFoundError(f"Raw data folder not found: {self.raw_data_folder}")
        
        # Create output directories
        self.figures_folder.mkdir(exist_ok=True)
        self.processed_data_folder.mkdir(exist_ok=True)
        
        # Set cellpose models path
        cellpose_models_path = self.experiment_folder.parent / '_pipeline_assets/cellpose_models/'
        os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(cellpose_models_path)
        
        self.logger.info(f"Experiment folder: {self.experiment_folder}")
        self.logger.info(f"Raw data folder: {self.raw_data_folder}")
        self.logger.info(f"Figures folder: {self.figures_folder}")
        self.logger.info(f"Processed data folder: {self.processed_data_folder}")
        
        # Log directory structure verification
        self.logger.info("Directory structure verification:")
        self.logger.info(f"  Raw data exists: {self.raw_data_folder.exists()}")
        self.logger.info(f"  Figures exists: {self.figures_folder.exists()}")
        self.logger.info(f"  Processed data exists: {self.processed_data_folder.exists()}")
        
    def gaussian_background_correction(self, image, sigma):
        """Estimate background with heavy gaussian blur"""
        background = skimage.filters.gaussian(image.astype(np.float32), sigma=sigma)
        corrected = image.astype(np.float32) - background
        return corrected
        
    def get_flatfield_files(self):
        """Select appropriate flatfield .nd2 files based on channel parameters"""
        flatfield_files = {
            'BF': self.flatfield_map_folder / 'flatfield_BF.nd2',
            'FAM': self.flatfield_map_folder / 'flatfield_FAM.nd2',
            'TAMRA': self.flatfield_map_folder / 'flatfield_TAMRA.nd2'
        }
        
        # Find which channel is active for spot detection
        active_channels = [channel for channel, is_active in self.spot_channel_params.items() if is_active]
        active_channel = active_channels[0] if active_channels else 'FAM'
        
        return {
            'segmentation': flatfield_files['BF'],
            'spot_detection': flatfield_files[active_channel]
        }
        
    def flatfield_correction(self, image, flatfield_image):
        """Flatfield correction based on previously taken flatfield images"""
        FF_image = nd2.imread(flatfield_image)
        mean_FF_BF = np.mean(FF_image)
        normalised_FF_BF = FF_image / mean_FF_BF
        corrected_image = image / normalised_FF_BF
        return corrected_image
        
    def max_project_xarray(self, array):
        """Maximum projection of single xarray channel"""
        return np.max(array.values, axis=0)
        
    def cellpose_bag(self, image):
        """Run bag pretrained cellpose SAM"""
        model = models.CellposeModel(
            gpu=self.segmentation_params['gpu'],
            pretrained_model=self.segmentation_params['model_name']
        )
        
        masks, flows, styles = model.eval(
            image,
            diameter=self.segmentation_params['diameter']
        )
        return masks
        
    def detect_spots(self, image, mask):
        """Detect spots within a single mask"""
        masked_spots = image * mask
        coords = skimage.feature.peak_local_max(
            masked_spots,
            min_distance=self.spot_detection_params['min_distance'],
            threshold_abs=self.spot_detection_params['threshold_abs']
        )
        return coords
        
    def calculate_roi_properties(self, mask, pixel_size_um):
        """Calculate ROI area and volume"""
        area_pixels = np.sum(mask)
        area_um2 = area_pixels * (pixel_size_um ** 2)
        volume_um3 = area_um2 * self.thickness_um
        return area_pixels, area_um2, volume_um3
        
    def create_qc_figure(self, image_seg, image_spots, masks, all_coords, condition, image_num, save_path, pixel_size_um):
        """Create quality control figure"""
        fig, axes = plt.subplots(2, 2, figsize=(8, 10))
        
        # Original segmentation image
        axes[0, 0].imshow(image_seg, cmap='gray')
        axes[0, 0].set_title('Segmentation Channel')
        axes[0, 0].axis('off')
        
        # Segmentation with masks overlay
        axes[0, 1].imshow(image_seg, cmap='gray')
        axes[0, 1].imshow(masks, alpha=0.3, cmap='tab10')
        axes[0, 1].set_title(f'Segmentation + Masks ({masks.max()} ROIs)')
        axes[0, 1].axis('off')
        
        # Spots channel
        axes[1, 0].imshow(image_spots, cmap='twilight_shifted')
        axes[1, 0].set_title('Spots Channel')
        axes[1, 0].axis('off')
        
        # Spots with detections
        axes[1, 1].imshow(image_spots, cmap='twilight_shifted')
        if len(all_coords) > 0:
            all_coords_array = np.vstack(all_coords)
            axes[1, 1].scatter(all_coords_array[:, 1], all_coords_array[:, 0], 
                             s=10, c='red', marker='x', alpha=0.8)
        axes[1, 1].set_title(f'Spots + Detections ({len(all_coords)} total)')
        axes[1, 1].axis('off')
        
        # Add scale bar to first subplot
        scalebar = ScaleBar(pixel_size_um, units='um', location='lower right')
        axes[0, 0].add_artist(scalebar)
        
        plt.suptitle(f'{condition} - Image {image_num}', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def process_single_file(self, file_path):
        """Process a single .nd2 file"""
        condition = file_path.stem
        self.logger.info(f"Processing {condition}...")
        
        try:
            # Read image
            array = nd2.imread(file_path, xarray=True)
            self.logger.info(f"Image dimensions: {array.sizes}")
            
            with nd2.ND2File(file_path) as nd2_file:
                pixel_size_um = nd2_file.voxel_size().x
                
            # Get flatfield files
            flatfields = self.get_flatfield_files()
            
            # Process each field of view
            for p in tqdm(range(array.sizes['P']), desc=f"Processing {condition}"):
                self.logger.info(f"Field of view {p+1}/{array.sizes['P']}")
                
                # Preprocessing
                image_segmentation = array.isel(P=p, C=self.channels_params['brightfield'])
                image_segmentation_max = self.max_project_xarray(image_segmentation)
                corrected_segmentation = self.flatfield_correction(image_segmentation_max, flatfields['segmentation'])
                
                image_spots = array.isel(P=p, C=self.channels_params['spots'])
                image_spots_max = self.max_project_xarray(image_spots)
                corrected_spots = self.flatfield_correction(image_spots_max, flatfields['spot_detection'])
                
                # Segmentation
                self.logger.info("Running segmentation...")
                masks = self.cellpose_bag(corrected_segmentation)
                filtered_masks = skimage.segmentation.clear_border(masks, buffer_size=25)
                
                num_rois = filtered_masks.max()
                self.logger.info(f"Found {num_rois} gel bags")
                
                # Spot detection and analysis
                all_coords_for_qc = []
                
                for mask_id in range(1, num_rois + 1):
                    single_mask = filtered_masks == mask_id
                    
                    # Detect spots
                    coords = self.detect_spots(corrected_spots, single_mask)
                    spot_count = len(coords)
                    
                    # Calculate ROI properties
                    area_pixels, area_um2, volume_um3 = self.calculate_roi_properties(single_mask, pixel_size_um)
                    
                    # Calculate densities
                    spots_per_area = spot_count / area_um2 if area_um2 > 0 else 0
                    spots_per_volume = spot_count / volume_um3 if volume_um3 > 0 else 0
                    
                    # Store results
                    result = {
                        'Experiment': self.experiment_folder.name,
                        'Condition': condition,
                        'Image_Number': p + 1,
                        'ROI': mask_id,
                        'Spot_Count': spot_count,
                        'ROI_Area_pixels': area_pixels,
                        'ROI_Area_um2': area_um2,
                        'ROI_Volume_um3': volume_um3,
                        'Spots_per_Area': spots_per_area,
                        'Spots_per_Volume': spots_per_volume
                    }
                    
                    self.results.append(result)
                    all_coords_for_qc.extend(coords)
                
                # Create QC figure
                qc_figure_path = self.figures_folder / f"{condition}_image_{p+1:03d}_QC.png"
                self.create_qc_figure(
                    corrected_segmentation, corrected_spots, filtered_masks,
                    all_coords_for_qc, condition, p+1, qc_figure_path, pixel_size_um
                )
                
                total_spots = len(all_coords_for_qc)
                avg_spots = total_spots / num_rois if num_rois > 0 else 0
                self.logger.info(f"Average spots detected: {avg_spots}")
                
        except Exception as e:
            self.logger.error(f"Error processing {condition}: {str(e)}")
            raise
            
    def process_all_files(self):
        """Process all .nd2 files in the raw data folder"""
        # Find all .nd2 files
        nd2_files = list(self.raw_data_folder.glob('*.nd2'))
        
        if not nd2_files:
            raise FileNotFoundError("No .nd2 files found in raw_data folder")
            
        self.logger.info(f"Found {len(nd2_files)} .nd2 files to process")
        for file_path in nd2_files:
            self.logger.info(f"  - {file_path.name}")
            
        # Process each file
        for file_path in nd2_files:
            self.process_single_file(file_path)
            
        self.logger.info(f"Processing complete! Analyzed {len(self.results)} ROIs total.")
        
    def save_results(self):
        """Save results to CSV and Excel files"""
        if not self.results:
            self.logger.warning("No results to save")
            return
            
        df = pd.DataFrame(self.results)
        
        # Save as CSV
        csv_path = self.processed_data_folder / f"{self.experiment_folder.name}_results.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as Excel with multiple sheets
        excel_path = self.processed_data_folder / f"{self.experiment_folder.name}_results.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='All_Data', index=False)
            
            # Summary by condition
            summary = df.groupby('Condition').agg({
                'Spot_Count': ['count', 'mean', 'std', 'sum'],
                'ROI_Area_um2': ['mean', 'std'],
                'Spots_per_Area': ['mean', 'std'],
                'Spots_per_Volume': ['mean', 'std']
            }).round(3)
            
            summary.columns = ['_'.join(col).strip() for col in summary.columns]
            summary.reset_index().to_excel(writer, sheet_name='Summary_by_Condition', index=False)
        
        self.logger.info(f"Results saved to:")
        self.logger.info(f"  CSV: {csv_path}")
        self.logger.info(f"  Excel: {excel_path}")
        
        # Display basic statistics
        self.logger.info(f"Quick Statistics:")
        self.logger.info(f"  Total ROIs analyzed: {len(df)}")
        self.logger.info(f"  Conditions: {df['Condition'].nunique()}")
        self.logger.info(f"  Total spots detected: {df['Spot_Count'].sum()}")
        self.logger.info(f"  Average spots per ROI: {df['Spot_Count'].mean():.1f} ± {df['Spot_Count'].std():.1f}")
        
    def create_summary_figures(self):
        """Create summary analysis figures"""
        if not self.results:
            self.logger.warning("No results to plot")
            return
            
        df = pd.DataFrame(self.results)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Figure 1: Summary plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Function to add sample size annotations
        def add_n_annotations(ax, data, group_col='Condition'):
            conditions = data[group_col].unique()
            for i, condition in enumerate(conditions):
                n = len(data[data[group_col] == condition])
                ax.text(ax.get_xlim()[1] * 0.98, i, f'n={n}', 
                       verticalalignment='center', 
                       horizontalalignment='right',
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Spot count distribution
        sns.boxplot(data=df, y='Condition', x='Spot_Count', ax=axes[0, 0])
        axes[0, 0].set_title('Spot Count Distribution by Condition')
        axes[0, 0].set_ylabel('')
        
        # ROI area distribution
        sns.boxplot(data=df, y='Condition', x='ROI_Area_um2', ax=axes[0, 1])
        axes[0, 1].set_title('ROI Area Distribution by Condition')
        axes[0, 1].set_xlabel('ROI Area (μm²)')
        axes[0, 1].set_ylabel('')
        
        # Spots per area
        sns.boxplot(data=df, y='Condition', x='Spots_per_Area', ax=axes[1, 0])
        axes[1, 0].set_title('Spots per Area by Condition')
        axes[1, 0].set_xlabel('Spots per μm²')
        axes[1, 0].set_ylabel('')
        
        # Spots per volume
        sns.boxplot(data=df, y='Condition', x='Spots_per_Volume', ax=axes[1, 1])
        axes[1, 1].set_title('Spots per Volume by Condition')
        axes[1, 1].set_xlabel('Spots per μm³')
        axes[1, 1].set_ylabel('')
        
        plt.tight_layout()
        summary_path = self.figures_folder / 'Summary_Analysis.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Correlation plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        sns.scatterplot(data=df, x='ROI_Area_um2', y='Spot_Count', hue='Condition', ax=ax)
        ax.set_title('Spot Count vs ROI Area')
        ax.set_xlabel('ROI Area (μm²)')
        ax.set_ylabel('Spot Count')
        
        plt.tight_layout()
        correlation_path = self.figures_folder / 'Correlation_Analysis.png'
        plt.savefig(correlation_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Summary figures saved:")
        self.logger.info(f"  {summary_path}")
        self.logger.info(f"  {correlation_path}")
        
    def run_pipeline(self):
        """Run the complete pipeline"""
        start_time = datetime.now()
        self.logger.info(f"Starting detection pipeline at {start_time}")
        
        try:
            self.process_all_files()
            self.save_results()
            self.create_summary_figures()
            
            end_time = datetime.now()
            duration = end_time - start_time
            self.logger.info(f"Pipeline completed successfully in {duration}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Detection Pipeline for Cellpose SAM Segmentation')
    parser.add_argument('--experiment-folder', type=str, default=None,
                        help='Path to experiment folder (default: auto-detect from script location)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration JSON file')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU usage')
    parser.add_argument('--thickness', type=float, default=20,
                        help='Gel thickness in micrometers (default: 20)')
    parser.add_argument('--diameter', type=int, default=676,
                        help='Segmentation diameter (default: 676)')
    parser.add_argument('--spot-threshold', type=int, default=800,
                        help='Spot detection threshold (default: 800)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show detected directories and files without processing')
    
    args = parser.parse_args()
    
    # Prepare kwargs for pipeline
    kwargs = {}
    if args.experiment_folder:
        kwargs['experiment_folder'] = Path(args.experiment_folder)
    if args.no_gpu:
        kwargs['segmentation_params'] = {'gpu': False}
    if args.thickness:
        kwargs['thickness_um'] = args.thickness
    if args.diameter:
        if 'segmentation_params' not in kwargs:
            kwargs['segmentation_params'] = {}
        kwargs['segmentation_params']['diameter'] = args.diameter
    if args.spot_threshold:
        kwargs['spot_detection_params'] = {'threshold_abs': args.spot_threshold}
    
    # Initialize and run pipeline
    pipeline = DetectionPipeline(config_file=args.config, **kwargs)
    
    # If dry run, just show what would be processed
    if args.dry_run:
        pipeline.logger.info("DRY RUN - Directory structure:")
        pipeline.logger.info(f"  Script location: {Path(__file__).parent}")
        pipeline.logger.info(f"  Experiment folder: {pipeline.experiment_folder}")
        pipeline.logger.info(f"  Raw data folder: {pipeline.raw_data_folder}")
        pipeline.logger.info(f"  Figures folder: {pipeline.figures_folder}")
        pipeline.logger.info(f"  Processed data folder: {pipeline.processed_data_folder}")
        
        # Show files that would be processed
        nd2_files = list(pipeline.raw_data_folder.glob('*.nd2'))
        pipeline.logger.info(f"  Found {len(nd2_files)} .nd2 files:")
        for file_path in nd2_files:
            pipeline.logger.info(f"    - {file_path.name}")
        
        # Check for flatfield files
        flatfields = pipeline.get_flatfield_files()
        pipeline.logger.info("  Flatfield files:")
        for purpose, path in flatfields.items():
            exists = path.exists()
            pipeline.logger.info(f"    {purpose}: {path} {'✓' if exists else '✗'}")
        
        return
    
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()