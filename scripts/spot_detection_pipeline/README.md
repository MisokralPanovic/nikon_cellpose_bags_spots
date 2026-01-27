# Spot Detection Pipeline

Automated segmentation and spot detection for microscopy images.

## Installation

### Option A: Conda (Recommended)

1. **Create conda environment from file:**
```bash
   cd spot_detection_pipeline
   conda env create -f environment.yml
```

2. **Activate environment:**
```bash
   conda activate spot_detector
```

3. **Verify installation:**
```bash
   python -c "from spot_detector import run_pipeline; print('✅ Installation successful')"
```

### Option B: Conda + Manual Steps

1. **Create base environment:**
```bash
   conda create -n spot_detector python=3.11
   conda activate spot_detector
```

2. **Install conda packages:**
```bash
   conda install -c conda-forge -c pytorch \
     numpy pandas scipy scikit-image opencv \
     pytorch pytorch-cuda=12.1 matplotlib seaborn \
     xarray dask pyyaml tqdm openpyxl jupyterlab
```

3. **Install pip packages:**
```bash
   pip install nd2 aicspylibczi aicsimageio cellpose spotiflow matplotlib-scalebar
```

4. **Install pipeline:**
```bash
   cd spot_detection_pipeline
   pip install -e .
```

### Option C: Pip Only
```bash
python -m venv spot_detector_env
source spot_detector_env/bin/activate  # On Windows: spot_detector_env\Scripts\activate
cd spot_detection_pipeline
pip install -r requirements.txt
pip install -e .
```

## Quick Start

1. **Navigate to experiment folder:**
```bash
   cd /path/to/your/experiment
```

2. **Ensure folder structure:**
```
   experiment_folder/
   ├── raw_data/          # Put your .nd2 or .czi files here
   ├── figures/           # (created automatically)
   ├── processed_data/    # (created automatically)
   └── scripts/
       └── spot_detection_pipeline/
```

3. **Run pipeline:**
```bash
   # Simple run
   python scripts/spot_detection_pipeline/run_pipeline.py
   
   # Or using CLI
   spot-detect
   
   # With options
   spot-detect --verbose --config custom_config.yml
```

## Usage

### Basic Usage
```bash
cd /path/to/experiment_folder
python scripts/spot_detection_pipeline/run_pipeline.py
```

### CLI Options
```bash
spot-detect --help
spot-detect --verbose                              # Debug logging
spot-detect --config custom_config.yml             # Custom config
spot-detect --experiment-dir /path/to/experiment   # Specify directory
```

### As Python Module
```python
from spot_detector import load_config, setup_paths, run_pipeline
from pathlib import Path

config = load_config('config.yml')
config = setup_paths(config, Path.cwd())
results = run_pipeline(config)
```

## Configuration

Edit `config.yml` to customize:
- Channel assignments
- Segmentation parameters (Cellpose model, downsampling)
- Detection parameters (spot distance threshold)
- Analysis parameters (ROI size filter, gel thickness)
- QC figure settings

## Output

- **CSV**: `processed_data/{experiment_name}_results.csv`
- **Excel**: `processed_data/{experiment_name}_results.xlsx` (with summary sheet)
- **QC Figures**: `figures/{condition}_image_{N}_QC.png`
- **Summary Figures**: `figures/Summary_Analysis.png`, `figures/Correlation_Analysis.png`
- **Log**: `processed_data/pipeline.log`

## Troubleshooting

### GPU Issues
If you don't have CUDA/GPU:
```yaml
# In environment.yml, replace pytorch-cuda line with:
- cpuonly
```

Or in config.yml:
```yaml
segmentation:
  use_gpu: false
```

### Import Errors
```bash
# Verify environment
conda activate spot_detector
conda list | grep cellpose
pip list | grep spotiflow
```

### Path Issues
Make sure you run from the experiment folder, not from the pipeline folder.
```bash
# Wrong: cd spot_detection_pipeline && python run_pipeline.py
# Right: cd experiment_folder && python scripts/spot_detection_pipeline/run_pipeline.py
```