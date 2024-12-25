import matplotlib.pyplot as plt
import numpy as np
from heapq import heappush, heappop

def display_two_images(image1, image2, title1, title2):
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title(title1)
    plt.imshow(image1)
    plt.axis('off')

    # CLAHE equalized image
    plt.subplot(1, 2, 2)
    plt.title(title2)
    plt.imshow(image2)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
# %%  
def rgb_to_bgr(image):
    """
    Convert RGB image to BGR manually.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
        
    bgr = image.copy()
    # Swap red and blue channels
    bgr[:, :, 0] = image[:, :, 2]  # Blue = Red
    bgr[:, :, 2] = image[:, :, 0]  # Red = Blue
    # Green stays the same
    return bgr

def bgr_to_rgb(image):
    """
    Convert BGR image to RGB manually.
    """
    return rgb_to_bgr(image)

def rgb_to_gray(image):
    """
    Convert RGB image to grayscale manually.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
        
    # Luminance formula
    gray = 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]
    return gray

def gray_to_rgb(image):
    """
    Convert grayscale image to RGB manually.
    """
    if len(image.shape) != 2:
        return image
        
    # Create a 3-channel image by duplicating the grayscale values
    return np.stack((image, image, image), axis=2)

def rgb_to_yuv(image):
    """
    Convert RGB image to YUV manually.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
        
    yuv = np.zeros_like(image, dtype=np.float32)
    # Y channel
    yuv[:, :, 0] = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    # U channel
    yuv[:, :, 1] = -0.147 * image[:, :, 0] - 0.289 * image[:, :, 1] + 0.436 * image[:, :, 2]
    # V channel
    yuv[:, :, 2] = 0.615 * image[:, :, 0] - 0.515 * image[:, :, 1] - 0.100 * image[:, :, 2]
    return yuv

def yuv_to_rgb(image):
    """
    Convert YUV image to RGB manually.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
        
    rgb = np.zeros_like(image, dtype=np.float32)
    # Red channel
    rgb[:, :, 0] = image[:, :, 0] + 1.140 * image[:, :, 2]
    # Green channel
    rgb[:, :, 1] = image[:, :, 0] - 0.395 * image[:, :, 1] - 0.581 * image[:, :, 2]
    # Blue channel
    rgb[:, :, 2] = image[:, :, 0] + 2.032 * image[:, :, 1]
    return rgb

def bgr_to_yuv(image):
    """
    Convert BGR image to YUV manually.
    """
    return rgb_to_yuv(bgr_to_rgb(image))

def yuv_to_bgr(image):
    """
    Convert YUV image to BGR manually.
    """
    return rgb_to_bgr(yuv_to_rgb(image))

# %%
def resize_mask(mask, target_height, target_width):
    """
    Manually resize a mask to target dimensions using bilinear interpolation.
    """
    if mask.shape[0] == target_height and mask.shape[1] == target_width:
        return mask
        
    # Calculate scaling factors
    scale_y = target_height / mask.shape[0]
    scale_x = target_width / mask.shape[1]
    
    resized = np.zeros((target_height, target_width), dtype=mask.dtype)
    
    for y in range(target_height):
        for x in range(target_width):
            # Find corresponding position in original image
            src_y = y / scale_y
            src_x = x / scale_x
            
            # Get integer and fractional parts
            y0 = min(int(src_y), mask.shape[0] - 2)
            x0 = min(int(src_x), mask.shape[1] - 2)
            y1 = y0 + 1
            x1 = x0 + 1
            
            # Calculate interpolation weights
            wy = src_y - y0
            wx = src_x - x0
            
            # Bilinear interpolation
            value = (mask[y0, x0] * (1 - wy) * (1 - wx) +
                    mask[y0, x1] * (1 - wy) * wx +
                    mask[y1, x0] * wy * (1 - wx) +
                    mask[y1, x1] * wy * wx)
            
            resized[y, x] = round(value)
            
    return resized
# %%
def telea_inpaint(image, mask, radius=3):
    """
    Manual implementation of Telea's inpainting algorithm.
    """
    # Initialize arrays
    height, width = mask.shape
    known = mask == 0
    band = np.zeros_like(mask, dtype=bool)
    distance = np.full_like(mask, np.inf, dtype=float)
    heap = []
    
    # Initialize narrow band
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    
    # Find initial boundary points
    for y in range(1, height-1):
        for x in range(1, width-1):
            if not known[y, x]:
                for k in range(8):
                    ny, nx = y + dy[k], x + dx[k]
                    if known[ny, nx]:
                        distance[y, x] = 0
                        band[y, x] = True
                        heappush(heap, (0, (y, x)))
                        break
    
    # Process points in narrow band
    while heap:
        d, (y, x) = heappop(heap)
        if d > distance[y, x]:
            continue
            
        # Calculate weighted average of known neighbors
        total_weight = 0
        weighted_sum = np.zeros(3) if len(image.shape) == 3 else 0
        
        for k in range(8):
            ny, nx = y + dy[k], x + dx[k]
            if 0 <= ny < height and 0 <= nx < width and known[ny, nx]:
                weight = 1 / (1 + abs(y-ny) + abs(x-nx))
                total_weight += weight
                weighted_sum += weight * image[ny, nx]
        
        # Update pixel value
        if total_weight > 0:
            image[y, x] = weighted_sum / total_weight
            known[y, x] = True
            
            # Update neighbors
            for k in range(8):
                ny, nx = y + dy[k], x + dx[k]
                if (0 <= ny < height and 0 <= nx < width and 
                    not known[ny, nx] and not band[ny, nx]):
                    new_dist = distance[y, x] + 1
                    if new_dist < distance[ny, nx]:
                        distance[ny, nx] = new_dist
                        band[ny, nx] = True
                        heappush(heap, (new_dist, (ny, nx)))
    
    return image
# %%
def gaussian_kernel(size, sigma=1):
  """Generates a Gaussian kernel manually."""
  kernel = np.zeros((size, size), dtype=np.float32)
  center = size // 2

  for x in range(size):
      for y in range(size):
          diff = np.square(x - center) + np.square(y - center)
          kernel[x, y] = np.exp(-diff / (2 * np.square(sigma)))

  return kernel / np.sum(kernel)

def apply_gaussian_filter(image, size, sigma):
    """Applies a Gaussian filter to a colored image using manual convolution."""

    kernel = gaussian_kernel(size, sigma)

    # Get image dimensions
    if len(image.shape) == 3:
        image_height, image_width, channels = image.shape
    else:
        image_height, image_width = image.shape
        channels = 1

    kernel_size = kernel.shape[0]
    pad_width = kernel_size // 2

    # Handle both colored and grayscale images
    if channels == 1:
        # Pad the image to handle borders
        padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
        smoothed_image = np.zeros_like(image)

        # Perform convolution
        for i in range(image_height):
            for j in range(image_width):
                region = padded_image[i:i + kernel_size, j:j + kernel_size]
                smoothed_image[i, j] = np.sum(region * kernel)
    else:
        # For colored images, process each channel separately
        smoothed_image = np.zeros_like(image)

        for c in range(channels):
            # Pad the channel
            padded_channel = np.pad(image[:,:,c], pad_width, mode='constant', constant_values=0)

            # Perform convolution for each channel
            for i in range(image_height):
                for j in range(image_width):
                    region = padded_channel[i:i + kernel_size, j:j + kernel_size]
                    smoothed_image[i, j, c] = np.sum(region * kernel)

    return smoothed_image
# %%
def zero_padding(image, kernel_size):
    """
    Apply zero padding to an RGB image without using np.zeros.
    
    :param image: The original RGB image as a NumPy array (H x W x 3).
    :param kernel_size: The size of the kernel (must be odd).
    :return: The padded RGB image as a manually created NumPy array.
    """
    # Compute padding size
    padding_size = kernel_size // 2
    
    # Get original image dimensions
    original_height, original_width, channels = image.shape
    
    # Create the padded image manually
    padded_height = original_height + 2 * padding_size
    padded_width = original_width + 2 * padding_size
    
    # Initialize the padded image with zeros
    padded_image = [[[0 for _ in range(channels)] for _ in range(padded_width)] for _ in range(padded_height)]
    
    # Copy the original image into the center of the padded image
    for c in range(channels):
        for i in range(original_height):
            for j in range(original_width):
                padded_image[i + padding_size][j + padding_size][c] = image[i][j][c]
    
    # Convert back to a NumPy array
    return np.array(padded_image, dtype=image.dtype)

def compute_median(region):
    """
    Compute the median of a flattened region manually.
    
    :param region: The flattened region of interest.
    :return: The median value.
    """
    sorted_region = sorted(region)
    mid = len(sorted_region) // 2
    if len(sorted_region) % 2 == 0:
        # Average of two middle values for even-length list
        median = (sorted_region[mid - 1] + sorted_region[mid]) // 2
    else:
        # Middle value for odd-length list
        median = sorted_region[mid]
    return median


def median_filter(image, kernel_size=5):
    """
    Apply a median filter to the image.

    :param image: The input image as a NumPy array.
    :param kernel_size: The size of the kernel.
    :return: The filtered image.
    """
    pad_size = kernel_size // 2
    padded_image = zero_padding(image, kernel_size)
    height, width, channels = image.shape

    # Initialize an empty filtered image
    filtered_image = [[[0] * channels for _ in range(width)] for _ in range(height)]

    for c in range(channels):  # Iterate over color channels
        for i in range(height):  # Iterate over rows
            for j in range(width):  # Iterate over columns
                # Extract the region of interest
                region = padded_image[i:i + kernel_size, j:j + kernel_size, c].flatten()
                # Compute the median manually
                filtered_image[i][j][c] = compute_median(region)

    return np.array(filtered_image, dtype=image.dtype)
# %%
def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
    """
    Manual implementation of bilateral filter.
    """
    padded = np.pad(image, ((diameter//2, diameter//2), (diameter//2, diameter//2), (0, 0)), mode='reflect')
    result = np.zeros_like(image)
    
    radius = diameter // 2
    space_kernel = np.zeros((diameter, diameter))
    
    # Precompute spatial gaussian weights
    for i in range(diameter):
        for j in range(diameter):
            distance = np.sqrt((i-radius)**2 + (j-radius)**2)
            space_kernel[i, j] = np.exp(-(distance**2) / (2*sigma_space**2))
    
    for i in range(radius, padded.shape[0]-radius):
        for j in range(radius, padded.shape[1]-radius):
            for c in range(image.shape[2]):
                window = padded[i-radius:i+radius+1, j-radius:j+radius+1, c]
                center_val = padded[i, j, c]
                
                # Color weights
                color_diff = window - center_val
                color_weights = np.exp(-(color_diff**2) / (2*sigma_color**2))
                
                # Combined weights
                weights = color_weights * space_kernel
                norm_weights = weights / np.sum(weights)
                
                # Weighted sum
                result[i-radius, j-radius, c] = np.sum(window * norm_weights)
    
    return result.astype(np.uint8)

# %%
def add_weighted(img1, alpha, img2, beta, gamma=0):
    """
    Manual implementation of weighted image blending
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
        
    result = np.clip(img1.astype(float) * alpha + 
                    img2.astype(float) * beta + 
                    gamma, 0, 255).astype(np.uint8)
    return result

# %%
def laplacian(img):
    """Manual Laplacian operator implementation"""
    kernel = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype=np.float64)
                      
    padded = np.pad(img, 1, mode='reflect')
    result = np.zeros_like(img, dtype=np.float64)
    
    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            window = padded[i-1:i+2, j-1:j+2]
            result[i-1, j-1] = np.sum(window * kernel)
            
    return result

# %%
def sobel(img, dx, dy, ksize=3):
    """
    Manual Sobel operator implementation
    dx, dy: order of derivatives in x,y direction
    """
    # Sobel kernels for x and y directions
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    
    kernel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float64)
    
    padded = np.pad(img, 1, mode='reflect')
    result = np.zeros_like(img, dtype=np.float64)
    
    if dx == 1:
        kernel = kernel_x
    elif dy == 1:
        kernel = kernel_y
    
    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):
            window = padded[i-1:i+2, j-1:j+2]
            result[i-1, j-1] = np.sum(window * kernel)
            
    return result

# %%
def convert_scale_abs(image, alpha=1.0, beta=0):
    """Manual implementation of convertScaleAbs"""
    # Scale with alpha and add beta
    scaled = image.astype(float) * alpha + beta
    
    # Take absolute values
    abs_scaled = np.abs(scaled)
    
    # Clip to uint8 range and convert
    return np.clip(abs_scaled, 0, 255).astype(np.uint8)