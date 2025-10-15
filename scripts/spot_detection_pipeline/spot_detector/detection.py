import numpy as np
from skimage.morphology import white_tophat
from spotiflow.model import Spotiflow
from typing import Tuple, List, Dict

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
    ) -> Tuple[np.ndarray, List[Dict]]:
    """Detect spot-like features in an image using the default Spotiflow model.

    Args:
        image (np.ndarray): Input spot image as a 2D.
        min_distance (int): Minimum distance between detected spots. Defaults to 10.

    Returns:
        Tuple[np.ndarray, List[Dict]]: 
            - points (np.ndarray): Array of detected spot coordinates.
            - details (List[Dict]): List of metadata dictionaries for each spot, including confidence scores and other attributes.
    """
    model = Spotiflow.from_pretrained("general")
    points, details = model.predict(
        img=image,
        verbose=False,
        min_distance=min_distance
    )
    
    return points, details # type: ignore
