import numpy as np
from skimage.measure import block_reduce
from cellpose import models, utils
from spotiflow.model import Spotiflow
from typing import Tuple
from types import SimpleNamespace

# =====================================================================
# Segementation
# =====================================================================

def segment_2d(
    bf_stack: np.ndarray,
    model_cellpose: models.CellposeModel,
    factor: int = 4,
) -> np.ndarray:
    """Run BAG-pretrained Cellpose-SAM in 2D on stdev projection of stack, using image downscaling for faster processing.

    Args:
        bf_stack (np.ndarray): Input 3D BAG image for segmentation.
        model_cellpose (models.CellposeModel): Initiated Cellpose model.
        factor (int): Downscaling factor. Defaults to 4.

    Returns:
        np.ndarray: Segmentation masks, not touching the edges.
    """
    std_proj = bf_stack.std(axis=0).astype(np.float32)
    img_binned = block_reduce(std_proj, block_size=(factor, factor), func=np.mean) # type: ignore[arg-type]
    
    masks, _, _ = model_cellpose.eval(img_binned)
    
    masks_resized = masks.repeat(factor, axis=-2).repeat(factor, axis=-1)
    masks_cleaned = utils.remove_edge_masks(masks_resized, change_index=True)
    
    return masks_cleaned
    

def segment_3d(
    bf_stack: np.ndarray,
    model_cellpose: models.CellposeModel,
    factor: int = 4,
    stitch_threshold: float = 0.4,
) -> np.ndarray:
    """Run BAG-pretrained Cellpose-SAM in pseudo-3D (segementing individual planes and stiching them together) on minimal projection substraced stack, using image downscaling for faster processing.

    Args:
        bf_stack (np.ndarray): Input 3D BAG image for segmentation.
        model_cellpose (models.CellposeModel): Initiated Cellpose model.
        factor (int): Downscaling factor. Defaults to 4.
        stitch_threshold (float): Treshold for stiching segmented planes. Defaults to 0.4.

    Returns:
        np.ndarray: Segmentation masks, not touching the edges.
    """
    min_substracted = bf_stack.astype(np.float32) - np.min(bf_stack, axis=0).astype(np.float32)
    img_binned = block_reduce(min_substracted, block_size=(1,factor, factor), func=np.mean) # type: ignore[arg-type]
    
    masks, _, _ = model_cellpose.eval(
        img_binned,
        do_3D=False,
        z_axis=0,
        stitch_threshold=stitch_threshold
        )
    
    masks_resized = masks.repeat(factor, axis=-2).repeat(factor, axis=-1)
    masks_cleaned = np.zeros_like(masks_resized)
    for z in range(masks_resized.shape[0]):
        masks_cleaned[z] = utils.remove_edge_masks(masks_resized[z], change_index=True)
    
    return masks_cleaned

# =====================================================================
# Spot detection and assigment to masks
# =====================================================================

def detect_spots_spotiflow(
    spot_stack: np.ndarray,
    model_spotiflow: Spotiflow,
    prob_thresh: float,
    min_distance: int,
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
    points, details = model_spotiflow.predict(
        img=spot_stack,
        verbose=False,
        prob_thresh=prob_thresh,
        min_distance=min_distance
    )
    
    return points, details

def assign_spots_to_mask(
    coordinates: np.ndarray | list, 
    masks: np.ndarray
    ) -> np.ndarray:
    """Assigns coordinates (spots) to masks (objects), for 2D and 3D outputs.

    Args:
        coordinates (np.ndarray | list): Coordinates of detected spots.
        masks (np.ndarray): Numpy array of object masks from segmentation.
        
    Raises:
        ValueError: If coordinates and masks dimentions are mismatched.

    Returns:
        np.ndarray: A list of coordinates that falls within the non-zero masks.
    """
    if len(coordinates) == 0:
        return np.array([], dtype=int)
    
    if not isinstance(coordinates, np.ndarray):
        coordinates = np.array(coordinates)
    
    ndim = coordinates.shape[1]
    
    if ndim == 2 and masks.ndim == 2:
        yi = np.clip(np.round(coordinates[:, 0]).astype(int), 0, masks.shape[0] - 1)
        xi = np.clip(np.round(coordinates[:, 1]).astype(int), 0, masks.shape[1] - 1)
        return masks[yi, xi]
    
    elif ndim == 3 and masks.ndim == 3:
        zi = np.clip(np.round(coordinates[:, 0]).astype(int), 0, masks.shape[0] - 1)
        yi = np.clip(np.round(coordinates[:, 1]).astype(int), 0, masks.shape[1] - 1)
        xi = np.clip(np.round(coordinates[:, 2]).astype(int), 0, masks.shape[2] - 1)
        return masks[zi, yi, xi]
    
    else:
        raise ValueError(
            f"Mismatch: coords have {ndim} dims but masks have {masks.ndim} dims."
        )