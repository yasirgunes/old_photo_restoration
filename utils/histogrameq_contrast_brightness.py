# %%
import cv2
import matplotlib.pyplot as plt
from utils.mask_inpaint import generate_mask
from utils.util import *

# Function to adjust brightness and contrast
def adjust_brightness_contrast(image, alpha=1.1, beta=2):
    """
    Adjust brightness and contrast of an image.

    Parameters:
    - image: Input image (NumPy array).
    - alpha: Contrast control (1.0-3.0).
    - beta: Brightness control (0-100).

    Returns:
    - Adjusted image (NumPy array).
    """
    image = rgb_to_bgr(image)

    # Apply the formula: new_image = alpha * image + beta
    adjusted = convert_scale_abs(image, alpha=alpha, beta=beta)
    adjusted = bgr_to_rgb(adjusted)
    return adjusted




# %%
def get_histogram(image):
    hist = np.zeros(256, dtype=np.int32)
    for pixel in image.flatten():
        hist[int(pixel)] += 1
    return hist

def clip_histogram(hist, clip_limit):
    excess = 0
    clip_limit = int(clip_limit)
    for i in range(len(hist)):
        if hist[i] > clip_limit:
            excess += hist[i] - clip_limit
            hist[i] = clip_limit
    
    redistrib_per_bin = excess // len(hist)
    for i in range(len(hist)):
        hist[i] += redistrib_per_bin
    return hist

def apply_clahe(channel, clip_limit=0.5, grid_size=(8, 8)):
    height, width = channel.shape
    h_regions = grid_size[0]
    w_regions = grid_size[1]
    
    h_steps = height // h_regions
    w_steps = width // w_regions
    result = np.zeros_like(channel)
    
    for i in range(h_regions):
        for j in range(w_regions):
            y_start = i * h_steps
            y_end = y_start + h_steps if i < h_regions-1 else height
            x_start = j * w_steps
            x_end = x_start + w_steps if j < w_regions-1 else width
            
            region = channel[y_start:y_end, x_start:x_end]
            hist = get_histogram(region)
            
            clip_val = int(clip_limit * (region.shape[0] * region.shape[1]) / 256.0)
            hist = clip_histogram(hist, clip_val)
            
            cdf = np.cumsum(hist).astype(np.float32)
            cdf = ((cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())).astype(np.uint8)
            
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    result[y, x] = cdf[int(channel[y, x])]
    
    return result

def equalize_histogram(image, clip_limit=0.5):
    """Apply CLAHE to color image"""
    if len(image.shape) == 3:
        image = rgb_to_bgr(image)
    
    # Convert to YUV
    image_yuv = bgr_to_yuv(image)
    
    # Apply CLAHE to Y channel
    image_yuv[:, :, 0] = apply_clahe(image_yuv[:, :, 0], clip_limit)
    
    # Convert back to RGB
    return yuv_to_rgb(image_yuv)
