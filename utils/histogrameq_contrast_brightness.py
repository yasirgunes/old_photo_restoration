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
def adjust_brightness_contrast(image, alpha=1.1, beta=2):
    image = rgb_to_bgr(image)
    adjusted = convert_scale_abs(image, alpha=alpha, beta=beta)
    adjusted = bgr_to_rgb(adjusted)
    return adjusted

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

def interpolate_mapping(x, y, mappings, weights):
    result = 0
    for mapping, weight in zip(mappings, weights):
        result += mapping[int(x)] * weight
    return result

def apply_clahe(channel, clip_limit=0.5, grid_size=(8, 8)):
    height, width = channel.shape
    h_regions = grid_size[0]
    w_regions = grid_size[1]
    
    h_steps = height // h_regions
    w_steps = width // w_regions
    
    # Create mapping for each grid region
    mappings = {}
    for i in range(h_regions + 1):
        for j in range(w_regions + 1):
            y_start = max(0, i * h_steps - h_steps//2)
            y_end = min(height, (i + 1) * h_steps + h_steps//2)
            x_start = max(0, j * w_steps - w_steps//2)
            x_end = min(width, (j + 1) * w_steps + w_steps//2)
            
            region = channel[y_start:y_end, x_start:x_end]
            hist = get_histogram(region)
            hist = clip_histogram(hist, int(clip_limit * (region.size / 256.0)))
            
            # Create cumulative distribution function
            cdf = np.cumsum(hist).astype(np.float32)
            cdf = ((cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())).astype(np.uint8)
            mappings[(i, j)] = cdf
    
    # Apply interpolated mappings
    result = np.zeros_like(channel)
    for y in range(height):
        for x in range(width):
            # Find grid regions for interpolation
            grid_y = y / h_steps
            grid_x = x / w_steps
            
            grid_y0 = int(grid_y)
            grid_x0 = int(grid_x)
            
            # Calculate interpolation weights
            wy1 = grid_y - grid_y0
            wx1 = grid_x - grid_x0
            wy0 = 1 - wy1
            wx0 = 1 - wx1
            
            # Get surrounding mappings
            m00 = mappings.get((grid_y0, grid_x0), mappings[(min(grid_y0, h_regions), min(grid_x0, w_regions))])
            m01 = mappings.get((grid_y0, grid_x0 + 1), mappings[(min(grid_y0, h_regions), min(grid_x0 + 1, w_regions))])
            m10 = mappings.get((grid_y0 + 1, grid_x0), mappings[(min(grid_y0 + 1, h_regions), min(grid_x0, w_regions))])
            m11 = mappings.get((grid_y0 + 1, grid_x0 + 1), mappings[(min(grid_y0 + 1, h_regions), min(grid_x0 + 1, w_regions))])
            
            # Bilinear interpolation
            pixel_value = channel[y, x]
            result[y, x] = (wy0 * (wx0 * m00[pixel_value] + wx1 * m01[pixel_value]) +
                          wy1 * (wx0 * m10[pixel_value] + wx1 * m11[pixel_value]))
    
    return result.astype(np.uint8)

def equalize_histogram(image, clip_limit=0.5):
    print("Inside equalize_histogram")
    if len(image.shape) == 3:
        image = rgb_to_bgr(image)
        print("Converted to BGR")
    
    # Convert to YUV
    image_yuv = bgr_to_yuv(image)
    print("Converted to YUV")
    
    # Apply CLAHE to Y channel
    y_channel = image_yuv[:, :, 0].astype(np.uint8)
    image_yuv[:, :, 0] = apply_clahe(y_channel, clip_limit)
    
    # Convert back to RGB
    return yuv_to_rgb(image_yuv)
