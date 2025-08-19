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

CZI_COUNT=$(find "$ANALYSIS_PATH/raw_files" -name "*.czi" -type f 2>/dev/null | wc -l)
ND2_COUNT=$(find "$ANALYSIS_PATH/raw_files" -name "*.nd2" -type f 2>/dev/null | wc -l)

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