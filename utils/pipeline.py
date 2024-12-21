# %%
# implement pipelining of old photo restoration
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.adjust_sharpness import unsharp_masking
from utils.histogrameq_contrast_brightness import equalize_histogram, adjust_brightness_contrast
from utils.mask_inpaint import inpaint_scratches
from utils.util import display_two_images
from utils.noise_reduction import adaptive_noise_reduction

# %%
# org_image = cv2.imread('image.png')
# org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

# # Step 1: Remove scratches
# scratches_removed = inpaint_scratches(org_image)
# display_two_images(org_image, scratches_removed, 'Original', 'Scratches Removed')

# # Step 2: Noise reduction
# noise_reduced = adaptive_noise_reduction(scratches_removed)
# display_two_images(scratches_removed, noise_reduced, 'Scratches Removed', 'Noise Reduced')

# # Step 3: Histogram equalization
# hist_eq = equalize_histogram(noise_reduced)
# display_two_images(noise_reduced, hist_eq, 'Noice Reduced', 'Histogram Equalized')

# # Step 4: Adjust sharpness
# sharpened = unsharp_masking(hist_eq)
# display_two_images(hist_eq, sharpened, 'Histogram Equalized', 'Sharpened')

# # Step 5: Adjust brightness and contrast
# brightness_contrast = adjust_brightness_contrast(sharpened)
# display_two_images(sharpened, brightness_contrast, 'Sharpened', 'Brightness and Contrast Adjusted')


# # The org_image and the sharpened
# display_two_images(org_image, brightness_contrast, 'Original', 'Final Image')


def pipeline(image, *, status_callback=None):  # Use keyword-only argument
    """
    Apply the complete pipeline of old photo restoration to the input image.
    
    Parameters:
    - image: Input image (NumPy array)
    - status_callback: Optional callback function for status updates
    
    Returns:
    - Final restored image (NumPy array)
    """
    if status_callback:
        status_callback("Removing scratches...")
    scratches_removed = inpaint_scratches(image)
    
    if status_callback:
        status_callback("Implementing noise reduction...")
    noise_reduced = adaptive_noise_reduction(scratches_removed)
    
    if status_callback:
        status_callback("Equalizing histogram...")
    hist_eq = equalize_histogram(noise_reduced)
    
    if status_callback:
        status_callback("Adjusting sharpness...")
    sharpened = unsharp_masking(hist_eq)
    
    if status_callback:
        status_callback("Adjusting brightness and contrast...")
    brightness_contrast = adjust_brightness_contrast(sharpened)
    
    return brightness_contrast