# spot_detector/process.py
"""
Main pipeline processing orchestration.
"""
import logging
from pathlib import Path
from typing import Dict, List
import gc

import numpy as np
from tqdm import tqdm

from spot_detector.config import (
    get_analysis_params, get_footprint, get_qc_params,
    get_channel_params, get_detection_params, get_segmentation_params
)

from spot_detector.io_utils import (
    find_raw_files, extract_data_nd2, process_field_of_view_nd2,
    czi_group_conditions, extract_pixelSize_czi, process_field_of_view_czi
)

from spot_detector.preprocess import process_field_of_view

from spot_detector.segmentation import memory_efficient_segmentation

from spot_detector.detection import (
    spot_tophat_correction, detect_spots_spotiflow, analyze_rois_memory_efficient
)

from spot_detector.qc import create_qc_figure
from spot_detector.save_data import save_results
from spot_detector.summary_figures import create_summary_figures

logger = logging.getLogger(__name__)


def run_pipeline(config: Dict) -> List[Dict]:
    """
    Main pipeline orchestration.
    
    Processes all ND2 and CZI files found in raw_data folder, runs segmentation
    and spot detection, generates QC figures, saves results, and creates summary figures.
    
    Parameters
    ----------
    config
        Configuration dictionary with resolved paths
        
    Returns
    -------
    results
        List of all ROI result dictionaries
    """
    logger.info("="*60)
    logger.info("Starting pipeline processing")
    logger.info("="*60)
    
    # Extract config parameters once
    footprint = get_footprint(config)
    channels = get_channel_params(config)
    seg_params = get_segmentation_params(config)
    det_params = get_detection_params(config)
    analysis_params = get_analysis_params(config)
    qc_params = get_qc_params(config)
    
    # Find files
    nd2_files, czi_files = find_raw_files(config['paths']['raw_data'])
    
    results = []
    
    # Process ND2 files
    if nd2_files:
        logger.info(f"Found {len(nd2_files)} ND2 files")
        nd2_results = process_nd2_files(
            nd2_files, channels, footprint, seg_params, 
            det_params, analysis_params, qc_params, config
        )
        results.extend(nd2_results)
    
    # Process CZI files
    if czi_files:
        logger.info(f"Found {len(czi_files)} CZI files")
        czi_results = process_czi_files(
            czi_files, channels, footprint, seg_params,
            det_params, analysis_params, qc_params, config
        )
        results.extend(czi_results)
    
    # Save and summarize results
    if results:
        logger.info(f"Processing complete! Analyzed {len(results)} ROIs total")
        save_results(results, config)
        create_summary_figures(results, config)
    else:
        logger.warning("No ROIs detected in any images")
    
    logger.info("="*60)
    logger.info("Pipeline finished")
    logger.info("="*60)
    
    return results


def process_nd2_files(
    files: List[Path],
    channels: Dict,
    footprint: np.ndarray,
    seg_params: Dict,
    det_params: Dict,
    analysis_params: Dict,
    qc_params: Dict,
    config: Dict
) -> List[Dict]:
    """
    Process all ND2 files.
    
    Parameters
    ----------
    files
        List of ND2 file paths
    channels
        Channel configuration
    footprint
        Morphological footprint for tophat filtering
    seg_params, det_params, analysis_params, qc_params
        Processing parameters
    config
        Full configuration dictionary
        
    Returns
    -------
    results
        List of result dictionaries for all ROIs
    """
    all_results = []
    
    for file_path in files:
        logger.info("="*60)
        logger.info(f"Processing {file_path.name}")
        logger.info("="*60)
        
        try:
            # Load ND2 file
            data = extract_data_nd2(file_path)
            
            logger.info(f"Condition: {data.condition}")
            logger.info(f"Fields of view: {data.num_fields}")
            logger.info(f"Pixel size: {data.pixel_size_um:.4f} μm")
            
            # Process each field of view
            for p in tqdm(range(data.num_fields), desc=f"Processing {data.condition}"):
                logger.info(f"  Field of view {p+1}/{data.num_fields}")
                
                # Extract channels for this field
                seg_data, spots_data = process_field_of_view_nd2(
                    array=data.array,
                    field_of_view=p,
                    brightfield_channel=channels['brightfield'],
                    spots_channel=channels['spots']
                )
                
                # Process this field
                roi_results = process_single_field(
                    seg_image=seg_data,
                    spots_image=spots_data,
                    footprint=footprint,
                    condition=data.condition,
                    image_num=p + 1,
                    pixel_size_um=data.pixel_size_um,
                    seg_params=seg_params,
                    det_params=det_params,
                    analysis_params=analysis_params,
                    qc_params=qc_params,
                    config=config
                )
                
                all_results.extend(roi_results)
                
                # Cleanup
                del seg_data, spots_data
                gc.collect()
                
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}", exc_info=True)
            continue
    
    return all_results


def process_czi_files(
    files: List[Path],
    channels: Dict,
    footprint: np.ndarray,
    seg_params: Dict,
    det_params: Dict,
    analysis_params: Dict,
    qc_params: Dict,
    config: Dict
) -> List[Dict]:
    """
    Process all CZI files.
    
    Parameters
    ----------
    files
        List of CZI file paths
    channels
        Channel configuration
    footprint
        Morphological footprint for tophat filtering
    seg_params, det_params, analysis_params, qc_params
        Processing parameters
    config
        Full configuration dictionary
        
    Returns
    -------
    results
        List of result dictionaries for all ROIs
    """
    all_results = []
    
    # Group by condition
    condition_groups = czi_group_conditions(files)
    
    for condition, file_list in condition_groups.items():
        logger.info("="*60)
        logger.info(f"Processing condition: {condition}")
        logger.info(f"Number of fields: {len(file_list)}")
        logger.info("="*60)
        
        for file_path, field_num in tqdm(file_list, desc=f"Processing {condition}"):
            logger.info(f"  Field of view {field_num}")
            
            try:
                # Get pixel size
                pixel_size_um = extract_pixelSize_czi(file_path)
                logger.info(f"    Pixel size: {pixel_size_um:.4f} μm")
                
                # Read channels
                seg_image, spots_image = process_field_of_view_czi(
                    path=file_path,
                    brightfield_channel=channels['brightfield'],
                    spots_channel=channels['spots']
                )
                
                # Process
                roi_results = process_single_field(
                    seg_image=seg_image,
                    spots_image=spots_image,
                    footprint=footprint,
                    condition=condition,
                    image_num=field_num,
                    pixel_size_um=pixel_size_um,
                    seg_params=seg_params,
                    det_params=det_params,
                    analysis_params=analysis_params,
                    qc_params=qc_params,
                    config=config
                )
                
                all_results.extend(roi_results)
                
                # Cleanup
                del seg_image, spots_image
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}", exc_info=True)
                continue
    
    return all_results


def process_single_field(
    seg_image: np.ndarray,
    spots_image: np.ndarray,
    footprint: np.ndarray,
    condition: str,
    image_num: int,
    pixel_size_um: float,
    seg_params: Dict,
    det_params: Dict,
    analysis_params: Dict,
    qc_params: Dict,
    config: Dict
) -> List[Dict]:
    """
    Process a single field of view (common for ND2 and CZI).
    
    This is the core processing function that runs segmentation, spot detection,
    ROI analysis, and generates QC figures.
    
    Parameters
    ----------
    seg_image
        Segmentation channel (raw, not projected)
    spots_image
        Spots channel (raw, not projected)
    footprint
        Morphological footprint for tophat filtering
    condition
        Experimental condition name
    image_num
        Field of view number
    pixel_size_um
        Pixel size in micrometers
    seg_params, det_params, analysis_params, qc_params
        Processing parameters
    config
        Full configuration dictionary
        
    Returns
    -------
    roi_results
        List of result dictionaries, one per ROI
    """
    # Project to 2D
    seg_projected, spots_projected = process_field_of_view(seg_image, spots_image)
    del seg_image, spots_image
    
    # Tophat correction for spots
    corrected_spots = spot_tophat_correction(spots_projected, footprint)
    
    # Segmentation
    logger.info("    Running segmentation...")
    masks = memory_efficient_segmentation(seg_projected, **seg_params)
    num_rois = masks.max()
    logger.info(f"    Found {num_rois} ROIs")
    
    if num_rois == 0:
        logger.warning("    ⚠️ No ROIs detected, skipping field")
        return []
    
    # Spot detection
    logger.info("    Detecting spots...")
    coords, details = detect_spots_spotiflow(corrected_spots, **det_params)
    logger.info(f"    Detected {len(coords)} spots total")
    
    # Analyze ROIs
    logger.info("    Analyzing ROIs...")
    roi_results, assigned_coords = analyze_rois_memory_efficient(
        masks=masks,
        coords_spotiflow=coords,
        pixel_size_um=pixel_size_um,
        thickness_um=analysis_params['thickness_um']
    )
    
    # Add metadata to results
    experiment_name = config['experiment']['name']
    for result in roi_results:
        result.update({
            'Experiment': experiment_name,
            'Condition': condition,
            'Image_Number': image_num
        })
    
    # Filter by minimum area
    num_before_filter = len(roi_results)
    roi_results = [
        r for r in roi_results 
        if r['ROI_Area_um2'] >= analysis_params['min_roi_area_um2']
    ]
    num_after_filter = len(roi_results)
    
    if num_before_filter != num_after_filter:
        logger.info(f"    Filtered out {num_before_filter - num_after_filter} ROIs (< {analysis_params['min_roi_area_um2']} μm²)")
    
    logger.info(f"    {num_after_filter} ROIs passed size filter")
    
    if num_after_filter > 0:
        avg_spots = sum(r['Spot_Count'] for r in roi_results) / num_after_filter
        logger.info(f"    Average spots per ROI: {avg_spots:.1f}")
    
    # Generate QC figure
    if qc_params['save_figures']:
        qc_path = (config['paths']['figures'] / 
                   f"{condition}_image_{image_num:03d}_QC.png")
        
        create_qc_figure(
            segmentation_image=seg_projected,
            spots=spots_projected,
            coordinates=assigned_coords,
            corrected_spots=corrected_spots,
            masks=masks,
            flow_details=details,
            condition=condition,
            image_num=image_num,
            output_path=qc_path,
            pixel_size_um=pixel_size_um,
            **qc_params
        )
        logger.info(f"    QC figure saved: {qc_path.name}")
    
    # Cleanup
    del seg_projected, spots_projected, corrected_spots, masks, coords, details
    gc.collect()
    
    return roi_results