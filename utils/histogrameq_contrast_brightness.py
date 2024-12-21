# %%
import cv2
import matplotlib.pyplot as plt
from utils.mask_inpaint import generate_mask

# # Load the image and preprocess it
# image = cv2.imread("image.png")  # Read as BGR
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

# mask = generate_mask(image)

# image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# # %%
# # Convert to YUV color space
# image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# # Create CLAHE object
# clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))

# # Apply CLAHE to the Y channel
# image_yuv[:, :, 0] = clahe.apply(image_yuv[:, :, 0])

# # Convert back to RGB color space
# image_equalized = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

# # Plot the results
# plt.figure(figsize=(12, 6))

# # Original image
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image_rgb)
# plt.axis('off')

# # CLAHE equalized image
# plt.subplot(1, 2, 2)
# plt.title("Equalized Image (CLAHE)")
# plt.imshow(image_equalized)
# plt.axis('off')

# plt.tight_layout()
# plt.show()


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
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# # Load the input image
# image = cv2.imread("image.png")
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display


# # Adjust the image
# adjusted_image = adjust_brightness_contrast(image_equalized)

# # Plot the results
# plt.figure(figsize=(12, 6))

# # Original image
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image_equalized)
# plt.axis('off')

# # Adjusted image
# plt.subplot(1, 2, 2)
# plt.title(f"Adjusted Image (alpha={1.1}, beta={2})")
# plt.imshow(adjusted_image)
# plt.axis('off')

# plt.tight_layout()
# plt.show()


# %%
def equalize_histogram(image, clipLimit=0.5):
    """
    Apply histogram equalization to the input image.

    Parameters:
    - image: Input image (NumPy array).

    Returns:
    - Equalized image (NumPy array).
    """

    # Convert to YUV color space
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))

    # Apply CLAHE to the Y channel
    image_yuv[:, :, 0] = clahe.apply(image_yuv[:, :, 0])

    # Convert back to RGB color space
    image_equalized = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)


    return image_equalized


