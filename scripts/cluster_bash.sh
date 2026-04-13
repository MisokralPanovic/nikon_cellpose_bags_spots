#!/bin/bash

# git download
GITHUB_REPO="https://github.com/MisokralPanovic/nikon_cellpose_bags_spots.git"


module load EBModules
module load Anaconda3

conda env create -f scripts/spot_detection_pipeline/environment.yml 

conda init bash
source ~/.bashrc

conda activate spot_detector