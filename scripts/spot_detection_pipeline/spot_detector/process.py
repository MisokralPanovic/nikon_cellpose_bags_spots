# %% package import
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
    find_raw_files, extract_data_nd2, process_field_of_view_nd2
)

from spot_detector.preprocess import (
    process_field_of_view
)

from spot_detector.segmentation import (
    memory_efficient_segmentation
)

from spot_detector.detection import (
    spot_tophat_correction, detect_spots_spotiflow, analyze_rois_memory_efficient
)

logger = logging.getLogger(__name__)

# %% main loop


def run_pipeline(config: Dict):
    
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
    seg_image,
    spot_image,
    model_path,
    scale_factor,
    use_gpu,
    buffer_size,
    footprint
):
    seg_projected, spots_projected = process_field_of_view(seg_image, spot_image)
    
    filtered_masks = memory_efficient_segmentation(
        seg_projected, model_path, scale_factor, use_gpu, buffer_size)
    spots_tophat = spot_tophat_correction(spots_projected, footprint)