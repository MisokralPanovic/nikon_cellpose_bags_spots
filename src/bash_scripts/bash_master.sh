module load EBModules
module load Python/3.11.5-GCCcore-13.2.0


# load and save module collection
module load EBModules UCX-CUDA/1.14.1-GCCcore-12.3.0-CUDA-12.1.1 OpenMPI/4.1.5-GCC-12.3.0 FLTK/1.3.8-GCCcore-12.3.0

module save cuda-fltk
module restore cuda-fltk

# delete collection
cd ~/.lmod.d
rm cuda-fltk


# load anaconda
module load Anaconda3
conda create --name test_env python=3.11.13

conda env create -f environment.yml 

conda init bash  # Only needs to be run once. If not using "bash", replace with your shell.
source ~/.bashrc # Only needs to be run once
conda activate cellpose-bags

# rclone data from Dropbox
module load rclone

rclone listremotes                  # List configured endpoints
rclone lsd dropbox_wigler:                 # List directories in Dropbox
rclone ls dropbox_wigler:                  # List all files in Dropbox  
rclone copy file.txt dropbox_wigler:test   # Copy file to Dropbox test folder 
rclone copy dropbox_wigler:folder/file.txt # Copy file from some Dropbox folder

# --- setup analyusis
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


# --- upload results

#!/bin/bash

# Microscopy Analysis Upload Script
# Usage: Run from within the analysis folder: ./upload_results.sh

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

# Get current directory name (should be run from within the analysis folder)
CURRENT_DIR=$(pwd)
ANALYSIS_BASE="/home/michalv/Analysis"

# Check if we're in an analysis folder
if [[ "$CURRENT_DIR" != "$ANALYSIS_BASE"/* ]]; then
    print_error "Please run this script from within an analysis folder in $ANALYSIS_BASE"
    echo "Current directory: $CURRENT_DIR"
    exit 1
fi

FOLDER_NAME=$(basename "$CURRENT_DIR")
ANALYSIS_PATH="$CURRENT_DIR"

print_status "Current analysis folder: $FOLDER_NAME"

# Check for file types to determine destination
print_status "Checking file types in raw_files..."

CZI_COUNT=$(find "$ANALYSIS_PATH/raw_data" -name "*.czi" -type f 2>/dev/null | wc -l)
ND2_COUNT=$(find "$ANALYSIS_PATH/raw_data" -name "*.nd2" -type f 2>/dev/null | wc -l)

print_status "Found $CZI_COUNT .czi files and $ND2_COUNT .nd2 files"

# Determine destination based on file types
if [ $CZI_COUNT -gt 0 ] && [ $ND2_COUNT -gt 0 ]; then
    print_warning "Both .czi and .nd2 files found. Please specify destination:"
    echo "1) LSM710 (for .czi files)"
    echo "2) NikonWigler (for .nd2 files)"
    read -p "Enter choice (1 or 2): " choice
    case $choice in
        1)
            DESTINATION="dropbox_cshl:Microscopy/_LSM710/"
            ;;
        2)
            DESTINATION="dropbox_cshl:Microscopy/_NikonWigler/"
            ;;
        *)
            print_error "Invalid choice. Exiting."
            exit 1
            ;;
    esac
elif [ $CZI_COUNT -gt 0 ]; then
    DESTINATION="dropbox_cshl:Microscopy/_LSM710/"
    print_status "Detected .czi files - uploading to LSM710"
elif [ $ND2_COUNT -gt 0 ]; then
    DESTINATION="dropbox_cshl:Microscopy/_NikonWigler/"
    print_status "Detected .nd2 files - uploading to NikonWigler"
else
    print_warning "No .czi or .nd2 files found. Manual destination selection:"
    echo "1) LSM710"
    echo "2) NikonWigler"
    read -p "Enter choice (1 or 2): " choice
    case $choice in
        1)
            DESTINATION="dropbox_cshl:Microscopy/_LSM710/"
            ;;
        2)
            DESTINATION="dropbox_cshl:Microscopy/_NikonWigler/"
            ;;
        *)
            print_error "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

print_status "Uploading analysis folder to: $DESTINATION$FOLDER_NAME/"

# Upload the entire analysis folder
rclone copy "$ANALYSIS_PATH" "$DESTINATION$FOLDER_NAME/" --progress

if [ $? -eq 0 ]; then
    print_success "Upload completed successfully!"
    print_status "Analysis folder uploaded to: $DESTINATION$FOLDER_NAME/"
    
    # Ask if user wants to delete local folder
    echo ""
    read -p "Do you want to delete the local analysis folder? (y/N): " delete_choice
    case $delete_choice in
        [Yy]* )
            rm -rf "$ANALYSIS_PATH"
            print_success "Local analysis folder deleted"
            ;;
        * )
            print_status "Local analysis folder kept at: $ANALYSIS_PATH"
            ;;
    esac
else
    print_error "Upload failed!"
    exit 1
fi