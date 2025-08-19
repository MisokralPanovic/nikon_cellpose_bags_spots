#!/bin/bash

# Microscopy Analysis Setup Script
# Usage: ./setup_analysis.sh /path/to/usb/folder

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if USB path is provided
if [ $# -eq 0 ]; then
    print_error "Please provide USB folder path"
    echo "Usage: $0 /path/to/usb/folder"
    exit 1
fi

USB_PATH="$1"
ANALYSIS_BASE="/home/michalv/Analysis"
GITHUB_REPO="https://github.com/MisokralPanovic/nikon_cellpose_bags_spots.git"
CELLPOSE_MODEL_SOURCE="dropbox_cshl:Microscopy/.cellpose/models/cpsam_20x_downsampeled_20250630"
CELLPOSE_MODEL_DEST="/home/michalv/Analysis/_pipeline_assets/cellpose_models/"

# Validate USB path exists
if [ ! -d "$USB_PATH" ]; then
    print_error "USB path does not exist: $USB_PATH"
    exit 1
fi

# Get folder name from USB path
FOLDER_NAME=$(basename "$USB_PATH")
print_status "Using folder name: $FOLDER_NAME"

# Create analysis base directory if it doesn't exist
mkdir -p "$ANALYSIS_BASE"

# Navigate to analysis directory
cd "$ANALYSIS_BASE"

print_status "Cloning fresh repository..."
if [ -d "$FOLDER_NAME" ]; then
    print_warning "Folder $FOLDER_NAME already exists. Removing..."
    rm -rf "$FOLDER_NAME"
fi

# Clone repo with the target folder name
git clone "$GITHUB_REPO" "$FOLDER_NAME"
print_success "Repository cloned to $ANALYSIS_BASE/$FOLDER_NAME"

# Navigate to the new folder
cd "$FOLDER_NAME"

# Remove Git connection to prevent accidental syncing
print_status "Removing Git connection..."
rm -rf .git
print_success "Git connection removed - this is now a standalone folder"

# Create raw_files directory if it doesn't exist
mkdir -p raw_files

print_status "Copying microscopy files from USB..."
# Copy all files from USB to raw_data
cp -r "$USB_PATH"/* raw_data/
print_success "Files copied to raw_data/"

# Check and copy CellPose model
print_status "Checking CellPose model..."
mkdir -p "$CELLPOSE_MODEL_DEST"

if [ ! -f "$CELLPOSE_MODEL_DEST/cpsam_20x_downsampeled_20250630" ]; then
    print_status "Downloading CellPose model from Dropbox..."
    rclone copy "$CELLPOSE_MODEL_SOURCE" "$CELLPOSE_MODEL_DEST"
    if [ $? -eq 0 ]; then
        print_success "CellPose model downloaded successfully"
    else
        print_error "Failed to download CellPose model"
        exit 1
    fi
else
    print_success "CellPose model already exists locally"
fi

print_success "Setup complete!"
print_status "Analysis folder: $ANALYSIS_BASE/$FOLDER_NAME"
print_status "Raw data location: $ANALYSIS_BASE/$FOLDER_NAME/raw_data/"
print_status "You can now open VS Code and run your analysis"
print_status "When finished, cd to the analysis folder and run: ./upload_results.sh"
