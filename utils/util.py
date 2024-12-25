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

def gaussian_kernel(size, sigma=1):
  """Generates a Gaussian kernel manually."""
  kernel = np.zeros((size, size), dtype=np.float32)
  center = size // 2

  for x in range(size):
      for y in range(size):
          diff = np.square(x - center) + np.square(y - center)
          kernel[x, y] = np.exp(-diff / (2 * np.square(sigma)))

  return kernel / np.sum(kernel)

# %%
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