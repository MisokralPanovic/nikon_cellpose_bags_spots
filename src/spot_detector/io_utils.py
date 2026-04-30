# %% load packages
from typing import NamedTuple, Tuple
import numpy as np
from pathlib import Path
import nd2
import xarray as xr
from aicspylibczi import CziFile
from aicsimageio.aics_image import AICSImage

# todo error handling

# %% getting files - common
def find_raw_files(
    raw_data_folder: Path
    ) -> tuple[list[Path], list[Path]]:
    '''Finds .nd2 and .czi files in a given folder and outputs a list of the files for each file type.
    
    Args:
        raw_data_folder (Path): Path for .nd2 and .czi files.
        
    Raises:
        FileNotFoundError: If no .nd2 or .czi files are found.
    
    Returns:
        tuple[list[Path], list[Path]]: Tuple of lists of .nd2 image file paths, and .czi file paths.
    '''
    nd2_files = list(raw_data_folder.glob("*.nd2"))
    czi_files = list(raw_data_folder.glob("*.czi"))

    total_files = len(nd2_files) + len(czi_files)
    if total_files == 0:
        raise FileNotFoundError("No .nd2 or .czi files found in raw_data folder")

    return nd2_files, czi_files

# %% nd2

class ND2Data(NamedTuple):
    """Complete data from ND2 file."""
    condition: str
    array: xr.DataArray
    pixel_size_um: float
    num_fields: int

def extract_data_nd2(
    nd2_file: Path
    ) -> ND2Data:
    '''Extracts condition, array, pixel_size_um, and num_fields metadata from .nd2 file.
    Args:
        nd2_file (Path): Path for single .nd2 file.
        
    Raises:
    
    Returns:
        ND2Data: Named tuple with condition, array, pixel_size_um, num_fields.
    '''
    # error for if no file present
    
    condition = nd2_file.stem
        
    array = nd2.imread(nd2_file, xarray=True, dask=True)
    
    with nd2.ND2File(nd2_file) as file:
        pixel_size_um = file.voxel_size().x
        num_fields = file.sizes['P']
    
    return ND2Data(
        condition=condition,
        array=array,
        pixel_size_um=pixel_size_um,
        num_fields=num_fields
    )

def process_field_of_view_nd2(
    array: xr.DataArray, 
    field_of_view: int,
    brightfield_channel: int,
    spots_channel: int
    ) -> Tuple[xr.DataArray, xr.DataArray]:
    '''Extracts segmentation and spot image from a single field of view of nd2 array.
    
    Args:
        array (xr.DataArray): X Array of multiple channels, z-stacks and fields of view.
        field_of_view (int): Number of which field of view to extract.
        brightfield_channel (int): Index of brightfield channel.
        spots_channel (int): Index of spot channel.
        
    Raises:
    
    Returns:
        Tuple[xr.DataArray, xr.DataArray]: Xarray of seg channel and spot channel.    
    '''
    seg_image = array.isel(P=field_of_view-1, C=brightfield_channel)
    spots_image = array.isel(P=field_of_view-1, C=spots_channel)
    
    return seg_image, spots_image

# %% czi
def czi_group_conditions(
    czi_files: list[Path]
    ) -> dict[str, list[tuple[Path, int]]]:
    '''Extracts condition group names from the file stems of a list of .czi files.
    Args:
        czi_files (list[Path]): A list of .czi files.
        
    Raises:
    
    Returns:
        dict[str, list[tuple[Path, int]]]: A dictionary mapping condition names to lists of tuples. Each tuple contains:
            - Path: the original file path
            - int: the field number extracted from the filename
    '''    
    condition_groups = {}
    
    for file_path in czi_files:
        filename = file_path.stem
        parts = filename.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            condition = '_'.join(parts[:-1])
            field_num = int(parts[-1])
        else:
            condition = filename
            field_num = 1
        
        if condition not in condition_groups:
            condition_groups[condition] = []
        condition_groups[condition].append((file_path, field_num))
    
    # Sort files within each condition by field number
    for condition in condition_groups:
        condition_groups[condition].sort(key=lambda x: x[1])
    
    return condition_groups

def extract_pixelSize_czi(
    czi_file: Path
    ) -> float | None:
    '''Extracts pixel size in um from .czi file.
    Args:
        czi_file (Path): A path to .czi file.
        
    Raises:
    
    Returns:
        float | None: Pixel size in um.
    '''    
    img = AICSImage(czi_file)
    pixel_size_um = img.physical_pixel_sizes.X
    
    del img
    
    return pixel_size_um

def process_field_of_view_czi(
    path: Path, 
    brightfield_channel: int, 
    spots_channel: int
    ) -> Tuple[np.ndarray, np.ndarray]:
    '''Extracts segmentation and spot image from a single field of view of .czi file.
    Args:
        path (Path): Path to .czi file.
        brightfield_channel (int): Index of brightfield channel.
        spots_channel (int): Index of spot channel.
    Raises:
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Numpy array of seg channel and spot channel.   
    '''    
    czi = CziFile(path)
    seg_image, seg_info = czi.read_image(C=brightfield_channel)
    spots_image, spots_info = czi.read_image(C=spots_channel)
    
    del seg_info
    del spots_info
    
    return seg_image, spots_image
                