# %% import
import logging
from pathlib import Path
from typing import Dict, List
import gc

import numpy as np
from tqdm import tqdm

from spot_detector.io_utils import (
    find_raw_files, 
)

logger = logging.getLogger(__name__)

# %% loop


'''figure out the loop

'''



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