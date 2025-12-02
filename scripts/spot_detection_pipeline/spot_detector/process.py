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

logger = logging.getLogger(__name__)

# %% main loop


def run_pipeline(config: Dict):
    
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
        pass
        


'''czi condition handeling
{
    "sampleA": [
        (Path("sampleA_1.czi"), 1),
        (Path("sampleA_2.czi"), 2),
        (Path("sampleA_3.czi"), 3)
    ],
    "control": [
        (Path("control.czi"), 1)
    ]
}

so dont forget to "for file_path, field_num in tqdm(file_list, desc=f"Processing {condition}"):"
'''

# %% define helper functions
def process_nd2_files():
    pass

def process_czi_files():
    pass

def process_single_field(
    seg_image: Union[xr.DataArray, np.ndarray],
    spot_image: Union[xr.DataArray, np.ndarray],
    footprint: np.ndarray,
    condition: str,
    image_num: int,
    pixel_size_um: float,
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
        pixel_size_um (float): Pixel size in micrometers
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
    
    logger.info("   Running spot detection for {condition}_{image_num}")
    points, details = detect_spots_spotiflow(corrected_spots, **det_params)
    if num_rois > 0:
        logger.info(f"   Found {points} spots, {points / num_rois:.1f} spots per ROI")
    else:
        logger.info(f"   Found {points} spots, no ROIs available")
    
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