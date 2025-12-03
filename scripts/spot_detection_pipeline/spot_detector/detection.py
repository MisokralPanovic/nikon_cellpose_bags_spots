import numpy as np
from skimage.morphology import white_tophat
from spotiflow.model import Spotiflow
from typing import Tuple, List, Dict
from types import SimpleNamespace

def spot_tophat_correction(
    image: np.ndarray, 
    footprint: np.ndarray
    ) -> np.ndarray:
    """Apply white top-hat morphological filtering to enhance small bright spots in an image.

    Args:
        image (np.ndarray): Input grayscale image as a 2D NumPy array.
        footprint (np.ndarray): Structuring element used for morphological filtering. 
    
    Raises:
        ValueError: If white_tophat returns None. 

    Returns:
        np.ndarray: Image with enhanced bright spots, same shape as input.
    """
    corrected_spots = white_tophat(image, footprint)
    
    if corrected_spots is None:
        raise ValueError("white_tophat returned None. Please check the input image and footprint.")
    
    return corrected_spots

def detect_spots_spotiflow(
    image: np.ndarray,
    min_distance: int = 10
    ) -> Tuple[np.ndarray, SimpleNamespace]:
    """Detect spot-like features in an image using the default Spotiflow model.

    Args:
        image (np.ndarray): Input spot image as a 2D.
        min_distance (int): Minimum distance between detected spots. Defaults to 10.

    Returns:
        Tuple[np.ndarray, SimpleNamespace]: 
            - points (np.ndarray): Array of detected spot coordinates.
            - details (SimpleNamespace): List of metadata dictionaries for each spot, including confidence scores and other attributes.
    """
    model = Spotiflow.from_pretrained("general")
    points, details = model.predict(
        img=image,
        verbose=False,
        min_distance=min_distance
    )
    
    return points, details

def assign_spots_to_mask(
    coordinates: np.ndarray | list, 
    mask: np.ndarray
    ) -> list:
    """Assigns coordinates (spots) to masks (objects).

    Args:
        coordinates (np.ndarray | list): Coordinates of detected spots.
        mask (np.ndarray): Numpy array of object masks from segmentation.

    Returns:
        list: A list of coordinates that falls within the non-zero masks.
    """
    if len(coordinates) == 0:
        return []
    
    # Convert to numpy array if needed
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    
    roi_coords = []
    
    for coord in coordinates:
        y, x = int(round(coord[0])), int(round(coord[1]))
        
        # Check bounds and mask
        if (0 <= y < mask.shape[0] and 
            0 <= x < mask.shape[1] and 
            mask[y, x]):
            roi_coords.append(coord.tolist())  # Convert back to list for consistency
    
    return roi_coords

def calculate_roi_properties(
    mask: np.ndarray,
    pixel_size_um: float | None,
    thickness_um: float
    ) -> tuple[float, float, float]:
    """Calculate ROI area and volume.

    Args:
        mask (np.ndarray): ROI mask.
        pixel_size_um (float): Image pixel size.
        thickness_um (float): Expected thickness of tissue.

    Returns:
        tuple[float, float, float]: Tuple of ROI: area in pixels, area in um2, and volume in um3.
    """
    area_pixels = np.sum(mask)

    if pixel_size_um is None:
        area_um2 = 0.0
    else:
        area_um2 = area_pixels * (pixel_size_um ** 2)

    volume_um3 = area_um2 * thickness_um
    
    return area_pixels, area_um2, volume_um3

def analyze_rois_memory_efficient(
    masks: np.ndarray, 
    coords_spotiflow: np.ndarray | list, 
    pixel_size_um: float | None, 
    thickness_um: float
    ) -> Tuple[List[Dict], List]:
    """Analyze each ROI: count spots, calculate areas/volumes.

    Args:
        mask (np.ndarray): Segmentation masks with integer labels.
        coords_spotiflow (np.ndarray | list): Coordinates of detected spots.        
        pixel_size_um (float): Image pixel size.
        thickness_um (float): Expected thickness of tissue.

    Returns:
        Tuple[List[Dict], List]: Tuple of list of dictionaries with ROI measurements & list of spot coordinates assigned to ROIs.
    """
    num_rois = masks.max()
    results = []
    roi_coords_spotiflow = []
    
    for mask_id in range(1, num_rois + 1):
        single_mask = (masks == mask_id)
        
        roi_spotiflow_coords = assign_spots_to_mask(coords_spotiflow, single_mask)
        spot_count_spotiflow = len(roi_spotiflow_coords)
        
        area_pixels, area_um2, volume_um3 = calculate_roi_properties(single_mask, pixel_size_um, thickness_um)
        
        result = {
            'ROI': mask_id,
            'Spot_Count': spot_count_spotiflow,
            'ROI_Area_pixels': area_pixels,
            'ROI_Area_um2': area_um2,
            'ROI_Volume_um3': volume_um3,
            'Spots_per_Area': spot_count_spotiflow / area_um2 if area_um2 > 0 else 0,
            'Spots_per_Volume': spot_count_spotiflow / volume_um3 if volume_um3 > 0 else 0
        }
        
        results.append(result)
        roi_coords_spotiflow.extend(roi_spotiflow_coords)
        
        del single_mask
    
    return results, roi_coords_spotiflow