#!/bin/bash
set -euo pipefail

# setup_analysis.sh
# Usage: bash scripts/bash_scripts/setup_analysis.sh /path/to/source/raw_data
#
# 1. Reads cellpose model location from config.yml and checks it is present
# 2. Symlinks every item in SOURCE_PATH into the experiment's raw_data/ folder
# 3. Writes a file-tree of raw_data/ to raw_file_structure.txt
# 4. Renames the experiment folder to match the source folder name;
#    appends _v2, _v3, … if a folder with that name already exists


# --- Helpers ---


# --- Arguments ---




# check cellpose model
../raw_data/microscopy/segmentation_models/cpsam_20x_downsampeled_20250630

# symlink data and change folder name (or the other way around)

## text file of tree after symlinking
tree > raw_file_strucutre.txt

## version based on if similar exist

