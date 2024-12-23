# %%
# implement pipelining of old photo restoration
import cv2
import numpy as np
import matplotlib.pyplot as plt

# from utils.adjust_sharpness import unsharp_masking
# from utils.histogrameq_contrast_brightness import equalize_histogram, adjust_brightness_contrast
# from utils.mask_inpaint import inpaint_scratches
# from utils.util import display_two_images


# %%
def noise_reduction(image):
    """
    Apply noise reduction to the image
    :param image: input image
    :return: noise reduced image
    """
    img_noise = rgb_to_bgr(image)
    img_noise_rgb = bgr_to_rgb(img_noise)

    gaussian_blur = cv2.GaussianBlur(img_noise, (5, 5), 0)
    gaussian_blur_rgb = bgr_to_rgb(gaussian_blur)

    median_blur = cv2.medianBlur(img_noise, 5)
    median_blur_rgb = bgr_to_rgb(median_blur)

    bilateral_filter = cv2.bilateralFilter(img_noise, 9, 75, 75)
    bilateral_filter_rgb = bgr_to_rgb(bilateral_filter)

    nonlocal_mean = cv2.fastNlMeansDenoisingColored(img_noise, None, 10, 10, 7, 21)
    nonlocal_mean_rgb = bgr_to_rgb(nonlocal_mean)

    plt.figure(figsize=(14,8))
    tup = [
        ("Noise Image", img_noise_rgb),
        ("Gaussian Blur", gaussian_blur_rgb),
        ("Median Blur", median_blur_rgb),
        ("Bilateral Filter", bilateral_filter_rgb),
        ("Non-Local Means", nonlocal_mean_rgb)
    ]

    for i, (name, img) in enumerate(tup):
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(name)
        plt.axis('off')

    return nonlocal_mean_rgb

    

# %%
# noise_reduction("old_photo_02.jpg")

import numpy as np

def rgb_to_bgr(image):
    """Convert RGB to BGR"""
    return image[:, :, ::-1]

def bgr_to_rgb(image):
    """Convert BGR to RGB"""
    return image[:, :, ::-1]

def rgb_to_gray(image):
    """Convert RGB to grayscale"""
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

def estimate_noise(image):
    """Estimate noise using gradient analysis"""
    gray = rgb_to_gray(image)
    # Calculate gradients
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    noise_score = np.mean(np.abs(gx) + np.abs(gy))
    return min(1.0, noise_score / 100.0)

def estimate_detail_level(image):
    """Estimate detail level using gradient magnitude"""
    gray = rgb_to_gray(image)
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    return min(1.0, np.mean(gradient_magnitude) / 50.0)

def median_filter(image, kernel_size=3):
    """
    Apply median filter to reduce noise
    """
    height, width, channels = image.shape
    pad = kernel_size // 2
    result = np.zeros_like(image)
    
    # Pad the image
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                window = padded[i:i+kernel_size, j:j+kernel_size, c]
                result[i, j, c] = np.median(window)
    
    return result

def gaussian_kernel(size, sigma):
    """
    Create a Gaussian kernel
    """
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    return kernel / kernel.sum()

def bilateral_filter(image, kernel_size=5, sigma_space=30, sigma_color=30):
    """
    Custom implementation of bilateral filter
    """
    height, width, channels = image.shape
    pad = kernel_size // 2
    result = np.zeros_like(image)
    
    # Pad the image
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    # Create spatial Gaussian kernel
    spatial_kernel = gaussian_kernel(kernel_size, sigma_space)
    
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                window = padded[i:i+kernel_size, j:j+kernel_size, c]
                center_value = image[i, j, c]
                
                # Color difference Gaussian
                color_diff = window - center_value
                color_weight = np.exp(-(color_diff**2) / (2 * sigma_color**2))
                
                # Combine spatial and color weights
                weights = spatial_kernel * color_weight
                weights = weights / weights.sum()
                
                # Apply filter
                result[i, j, c] = np.sum(window * weights)
    
    return result.astype(np.uint8)

def non_local_means(image, search_window=21, patch_size=7, h=10):
    """
    Custom implementation of non-local means denoising
    """
    height, width, channels = image.shape
    pad_search = search_window // 2
    pad_patch = patch_size // 2
    result = np.zeros_like(image)
    
    # Pad the image
    padded = np.pad(image, ((pad_search, pad_search), (pad_search, pad_search), (0, 0)), mode='reflect')
    
    for i in range(height):
        for j in range(width):
            i_pad = i + pad_search
            j_pad = j + pad_search
            
            # Extract search window
            search_area = padded[i_pad-pad_search:i_pad+pad_search+1,
                               j_pad-pad_search:j_pad+pad_search+1]
            
            weights = np.zeros((search_window, search_window))
            
            # Reference patch
            ref_patch = padded[i_pad-pad_patch:i_pad+pad_patch+1,
                             j_pad-pad_patch:j_pad+pad_patch+1]
            
            # Compare patches within search window
            for si in range(search_window):
                for sj in range(search_window):
                    comp_patch = padded[i_pad-pad_search+si-pad_patch:i_pad-pad_search+si+pad_patch+1,
                                      j_pad-pad_search+sj-pad_patch:j_pad-pad_search+sj+pad_patch+1]
                    
                    # Calculate patch difference
                    diff = np.sum((ref_patch - comp_patch)**2)
                    weights[si, sj] = np.exp(-diff / h**2)
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Apply weights
            for c in range(channels):
                result[i, j, c] = np.sum(search_area[:, :, c] * weights)
    
    return result.astype(np.uint8)

def adaptive_noise_reduction(image):
    """
    Adaptive noise reduction pipeline using only NumPy operations
    """
    # Estimate noise level
    noise_level = estimate_noise(image)
    detail_level = estimate_detail_level(image)
    
    # Adjust filter parameters based on image characteristics
    median_size = 3 if noise_level < 0.5 else 5
    bilateral_sigma = max(10, int(30 * noise_level))
    nlm_h = max(5, int(15 * noise_level))
    
    # Apply filters in sequence
    median_filtered = median_filter(image, kernel_size=median_size)
    bilateral_filtered = bilateral_filter(
        median_filtered, 
        kernel_size=5,
        sigma_space=bilateral_sigma,
        sigma_color=bilateral_sigma
    )
    final_result = non_local_means(
        bilateral_filtered,
        search_window=11,
        patch_size=5,
        h=nlm_h
    )
    
    return final_result

def detect_faces(image):
    """
    Optional: Detect faces to apply different processing to facial regions
    """
    gray = rgb_to_gray(image)
    edges = cv2.Canny(gray, 100, 200)
    return edges


