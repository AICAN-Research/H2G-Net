import openslide as ops
from skimage.morphology import binary_closing, disk
import cv2
import numpy as np


# automatic tissue segmentation
def segment_tissue(raw_wsi_path, threshold_method=0, curr_plane=5, thresh=20, mclose1=5, mclose2=3):
    raw_wsi_reader = ops.OpenSlide(raw_wsi_path)
    raw_wsi = np.array(raw_wsi_reader.read_region((0, 0), curr_plane, raw_wsi_reader.level_dimensions[curr_plane]))[..., :3]
    raw_wsi_reader.close()

    # convert from RGB to HSV and get Saturation image
    tmp = cv2.cvtColor(raw_wsi, cv2.COLOR_RGB2HSV)[..., 1]
    if threshold_method == 1:
        _, tissue = cv2.threshold((tmp * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu's method
        tissue = (tissue > 0).astype(int)
        tissue = binary_closing(tissue, selem=disk(mclose1)).astype(int)
    else:
        tissue = ((tmp * 255).astype(np.uint8) >= thresh).astype(int)  # simple thresholding using predefined threshold
        tissue = binary_closing(tissue, selem=disk(mclose1)).astype(int)
    tissue_orig = (tissue > 0).astype(int)
    tissue = binary_closing(tissue, selem=disk(mclose2)).astype(int)
    return tissue, tissue_orig
