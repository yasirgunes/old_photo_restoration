import os
import sys
import numpy as np

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from gaussian_filter import gaussian_kernel, apply_gaussian_filter

# Function to apply unsharp masking
def unsharp_masking(image, kernel_size=5, sigma=1.0, strength=0.5):
    """
    Apply unsharp masking to enhance edges.

    Parameters:
    - image: Input image (NumPy array).
    - kernel_size: Size of the Gaussian kernel.
    - sigma: Standard deviation for Gaussian blur.
    - strength: Scaling factor for the sharpness effect.

    Returns:
    - Sharpened image (NumPy array).
    """
    # Create Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Generate a blurred version of the image
    blurred = apply_gaussian_filter(image, kernel)
    
    # Subtract the blurred image from the original
    mask = image - blurred
    
    # Combine the original image with the mask
    sharpened = image + strength * mask
    
    # Clip values to be in the valid range [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened