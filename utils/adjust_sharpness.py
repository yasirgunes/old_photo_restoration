import cv2
from utils.util import *

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
    # Generate a blurred version of the image
    # blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    blurred = apply_gaussian_filter(image, kernel_size, sigma)
    
    # Subtract the blurred image from the original
    # sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    sharpened = add_weighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened