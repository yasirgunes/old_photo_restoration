import numpy as np

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
    # Apply the formula: new_image = alpha * image + beta
    adjusted = alpha * image + beta
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted

def equalize_histogram(image, clipLimit=0.5):
    """
    Apply histogram equalization to the input image.

    Parameters:
    - image: Input image (NumPy array).

    Returns:
    - Equalized image (NumPy array).
    """
    # Convert to YUV color space
    image_yuv = rgb_to_yuv(image)

    # Apply CLAHE to the Y channel
    image_yuv[:, :, 0] = clahe(image_yuv[:, :, 0], clipLimit)

    # Convert back to RGB color space
    image_equalized = yuv_to_rgb(image_yuv)

    return image_equalized

def rgb_to_yuv(image):
    """
    Convert an RGB image to YUV color space.

    Parameters:
    - image: Input RGB image (NumPy array).

    Returns:
    - YUV image (NumPy array).
    """
    m = np.array([[0.299, 0.587, 0.114],
                  [-0.14713, -0.28886, 0.436],
                  [0.615, -0.51499, -0.10001]])
    yuv = np.dot(image, m.T)
    yuv[:, :, 1:] += 128.0
    return yuv

def yuv_to_rgb(image):
    """
    Convert a YUV image to RGB color space.

    Parameters:
    - image: Input YUV image (NumPy array).

    Returns:
    - RGB image (NumPy array).
    """
    m = np.array([[1.0, 0.0, 1.13983],
                  [1.0, -0.39465, -0.58060],
                  [1.0, 2.03211, 0.0]])
    rgb = np.dot(image, m.T)
    rgb[:, :, 1:] -= 128.0
    return np.clip(rgb, 0, 255).astype(np.uint8)

def clahe(channel, clipLimit):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a single channel.

    Parameters:
    - channel: Input channel (NumPy array).
    - clipLimit: Clipping limit for CLAHE.

    Returns:
    - Equalized channel (NumPy array).
    """
    hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Clip the histogram
    cdf_clipped = np.clip(cdf, 0, clipLimit * cdf.max())
    cdf_clipped = cdf_clipped * (cdf.max() / cdf_clipped.max())

    # Normalize the histogram
    cdf_normalized = (cdf_clipped - cdf_clipped.min()) * 255 / (cdf_clipped.max() - cdf_clipped.min())
    cdf_normalized = cdf_normalized.astype('uint8')

    # Map the original channel values to the equalized values
    equalized_channel = cdf_normalized[channel]

    return equalized_channel


