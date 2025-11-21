import numpy as np
import xarray as xr
from typing import Union, Tuple

def stdev_project(
    array: Union[xr.DataArray, np.ndarray]
    ) -> np.ndarray:
    """Converts xarray to numpy array & squeezes numpy array & stdev projects it

    Args:
        array (xarray | np.ndarray): Input grayscale image as a 2D NumPy array.

    Raises:
        ValueError: If input has less than 2 dimensions.

    Returns:
        np.ndarray: Standard deviation projection image.
    """
    # Convert xarray to numpy if needed
    if isinstance(array, xr.DataArray):
        array = array.values

    # Squeeze singleton dimensions
    array = np.squeeze(array)

    if array.ndim < 2:
        raise ValueError(f"Expected at least 2D input, got shape {array.shape}")

    return np.std(array, axis=0)

def process_field_of_view(
    seg_image: Union[xr.DataArray, np.ndarray],
    spot_image: Union[xr.DataArray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Processes field of view by stdev projecting segmentation and spot images.

    Args:
        seg_image (Union[xr.DataArray, np.ndarray]): Segmentation image stack.
        spot_image (Union[xr.DataArray, np.ndarray]): Spot image stack.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Stdev projected segmentation and spot images. 
    """
    seg_projected = stdev_project(seg_image)
    del seg_image
    
    spots_projected = stdev_project(spot_image)
    del spot_image
    
    return seg_projected, spots_projected