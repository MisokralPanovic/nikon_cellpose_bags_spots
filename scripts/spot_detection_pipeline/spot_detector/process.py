# spot_detector/process.py
"""
Main pipeline processing orchestration.
"""
# %% package import
import logging
from pathlib import Path
from typing import Dict, List, Union
import gc

import numpy as np
import xarray as xr
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

# %% main loop


def run_pipeline(config: Dict) -> List[Dict]:
    """Main pipeline orchestration.
    Processes all ND2 and CZI files found in raw_data folder, runs segmentation
    and spot detection, generates QC figures, saves results, and creates summary figures.
    
    Args:
        config (Dict): Configuration dictionary with resolved paths

    Returns:
        List[Dict]: List of all ROI result dictionaries
    """
    logger.info("="*60)
    logger.info("Starting pipeline processing")
    logger.info("="*60)    
    
    # extract config parameters
    footprint = get_footprint(config)
    channels = get_channel_params(config)
    seg_params = get_segmentation_params(config)
    det_params = get_detection_params(config)
    analysis_params = get_analysis_params(config)
    qc_params = get_qc_params(config)
    
    # get files
    nd2_files, czi_files = find_raw_files(config['paths']['raw_data'])

    results = []
    
    # process .nd2 files
    if nd2_files:
        logger.info(f'Found {len(nd2_files)} .ND2 files.')
        nd2_resuts = process_nd2_files(
            nd2_files, channels, footprint, seg_params, 
            det_params, analysis_params, qc_params, config
        )
        results.extend(nd2_resuts)
    
    # process .czi files
    if czi_files:
        logger.info(f'Found {len(czi_files)} .ND2 files.')
        czi_resuts = process_czi_files(
            czi_files, channels, footprint, seg_params, 
            det_params, analysis_params, qc_params, config
        )
        results.extend(czi_resuts)
        
    if results:
        logger.info(f"Processing finished! Analysed {len(results)} ROIs total.")
        save_results(results, config)
        create_summary_figures(results, config)
    else:
        logger.warning(f"No ROIs detected in any images.")
        
    logger.info("="*60)
    logger.info("Pipeline finished")
    logger.info("="*60)
    
    return results

# %% define helper functions
def process_nd2_files(
    nd2_files: List[Path],
    channels: Dict,
    footprint: np.ndarray,
    seg_params: Dict,
    det_params: Dict,
    analysis_params: Dict,
    qc_params: Dict,
    config: Dict
) -> List[Dict]:
    """Process all .nd2 files from a list.

    Args:
        nd2_files (List[Path]): List of .nd2 files
        channels (Dict): Channel configuration
        footprint (np.ndarray): Morphological footprint for tophat filtering
        seg_params (Dict): Processing parameters
        det_params (Dict): Processing parameters
        analysis_params (Dict): Processing parameters
        qc_params (Dict): Processing parameters
        config (Dict): Full configuration dictionary

    Returns:
        List[Dict]: List of result dictionaries for all ROIs
    """
    all_results = []
    
    for filepath in nd2_files:
        logger.info(f"="*60)
        logger.info(f"Processing {filepath.name}")
        logger.info(f"="*60)
        
        try:
            data = extract_data_nd2(filepath)
            for p in tqdm(range(data.num_fields), desc=f"Processing {data.condition}"):
                seg_image, spots_image = process_field_of_view_nd2(
                    array=data.array,
                    field_of_view=p +1,
                    brightfield_channel=channels['brightfield'],
                    spots_channel=channels['spots']
                )
                
                roi_results = process_single_field(
                    seg_image=seg_image,
                    spot_image=spots_image,
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

                del seg_image, spots_image
                gc.collect()
        
        except Exception as e:
            logger.error(f"Failed to process {filepath.name}: {e}", exc_info=True)
            continue

    return all_results

def process_czi_files(
    czi_files: List[Path],
    channels: Dict,
    footprint: np.ndarray,
    seg_params: Dict,
    det_params: Dict,
    analysis_params: Dict,
    qc_params: Dict,
    config: Dict
) -> List[Dict]:
    """_summary_

    Args:
        czi_files (List[Path]): _description_
        channels (Dict): _description_
        footprint (np.ndarray): _description_
        seg_params (Dict): _description_
        det_params (Dict): _description_
        analysis_params (Dict): _description_
        qc_params (Dict): _description_
        config (Dict): _description_

    Returns:
        List[Dict]: _description_
    """
    all_results = []
    
    condition_groups = czi_group_conditions(czi_files)
    for condition, file_list in condition_groups.items():
        logger.info(f"="*60)
        logger.info(f"Processing {condition}")
        logger.info(f"="*60)        
        
        for path, p in tqdm(file_list, desc=f"Processing {condition}"):
            try:
                pixel_size_um = extract_pixelSize_czi(path)
                seg_image, spots_image = process_field_of_view_czi(
                    path=path,
                    brightfield_channel=channels['brightfield'],
                    spots_channel=channels['spots']
                )
                
                roi_results = process_single_field(
                    seg_image=seg_image,
                    spot_image=spots_image,
                    footprint=footprint,
                    condition=condition,
                    image_num=p,
                    pixel_size_um=pixel_size_um,
                    seg_params=seg_params,
                    det_params=det_params,
                    analysis_params=analysis_params,
                    qc_params=qc_params,
                    config=config
                )
                
                all_results.extend(roi_results)
                
                del seg_image, spots_image
                gc.collect()
            
            except Exception as e:
                logger.error(f"Failed to process {path.name}: {e}", exc_info=True)
                continue
        
    return all_results

def process_single_field(
    seg_image: Union[xr.DataArray, np.ndarray],
    spot_image: Union[xr.DataArray, np.ndarray],
    footprint: np.ndarray,
    condition: str,
    image_num: int,
    pixel_size_um: float | None,
    seg_params: Dict,
    det_params: Dict,
    analysis_params: Dict,
    qc_params: Dict,
    config: Dict,
) -> List[Dict]:
    """Process a single field of view (common for ND2 and CZI).
    This is the core processing function that runs segmentation, spot detection,
    ROI analysis, and generates QC figures.    

    Args:
        seg_image (Union[xr.DataArray, np.ndarray]): Segmentation channel (raw, not projected)
        spot_image (Union[xr.DataArray, np.ndarray]): Spots channel (raw, not projected)
        footprint (np.ndarray): Morphological footprint for tophat filtering
        condition (str): Experimental condition name
        image_num (int): Field of view number
        pixel_size_um (float | None): Pixel size in micrometers
        seg_params (Dict): Processing parameters
        det_params (Dict): Processing parameters
        analysis_params (Dict): Processing parameters
        qc_params (Dict): Processing parameters
        config (Dict): Full configuration dictionary

    Returns:
        List[Dict]: List of result dictionaries, one per ROI
    """
    seg_projected, spots_projected = process_field_of_view(seg_image, spot_image)
    del seg_image, spot_image
    
    corrected_spots = spot_tophat_correction(spots_projected, footprint)    
    
    logger.info(f"   Running segmentation for {condition}_{image_num}")
    filtered_masks = memory_efficient_segmentation(seg_projected, **seg_params)
    num_rois = filtered_masks.max()
    logger.info(f"   Found {num_rois} ROIs")
    
    logger.info(f"   Running spot detection for {condition}_{image_num}")
    points, details = detect_spots_spotiflow(corrected_spots, **det_params)
    if num_rois > 0:
        logger.info(f"   Found {len(points)} spots, {len(points) / num_rois:.1f} spots per ROI")
    else:
        logger.info(f"   Found {len(points)} spots, no ROIs available")
    
    roi_results, roi_coords_spotiflow = analyze_rois_memory_efficient(masks=filtered_masks, coords_spotiflow=points, pixel_size_um=pixel_size_um, thickness_um=analysis_params['thickness_um'])

    # add metadata to results
    experiment_name = config['experiment']['name']
    for result in roi_results:
        result.update({
            'Experiment': experiment_name,
            'Condition': condition,
            'Image_Number': image_num
        })
    
    # filter out small objects
    roi_results = [
        r for r in roi_results 
        if r['ROI_Area_um2'] >= analysis_params['min_roi_area_um2']
    ]
    
    # create qc figure
    if qc_params['save_figures']:
        qc_path = config['paths']['figures'] / f"{condition}_image_{image_num:03d}_QC.png"
        create_qc_figure(
            segmentation_image=seg_projected,
            spots=spots_projected,
            coordinates=roi_coords_spotiflow,
            corrected_spots=corrected_spots,
            masks=filtered_masks,
            flow_details=details,
            condition=condition,
            output_path=qc_path,
            image_num=image_num,
            pixel_size_um=pixel_size_um,
            **qc_params
        )

    del seg_projected, spots_projected, corrected_spots, filtered_masks, points, details
    gc.collect()
    
    return roi_results
