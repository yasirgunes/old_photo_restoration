# %%
# implement pipelining of old photo restoration
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.util import *

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
    gaussian_blur_rgb = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB)

    median_blur = cv2.medianBlur(img_noise, 5)
    median_blur_rgb = cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB)

    bilateral_filter = cv2.bilateralFilter(img_noise, 9, 75, 75)
    bilateral_filter_rgb = cv2.cvtColor(bilateral_filter, cv2.COLOR_BGR2RGB)

    nonlocal_mean = cv2.fastNlMeansDenoisingColored(img_noise, None, 10, 10, 7, 21)
    nonlocal_mean_rgb = cv2.cvtColor(nonlocal_mean, cv2.COLOR_BGR2RGB)

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


def adaptive_noise_reduction(image):
    """
    Adaptive noise reduction that adjusts parameters based on image content and noise levels
    """
    # Convert to grayscale for analysis
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) -- implemented this manually in utils/util.py
    gray = rgb_to_gray(image)
    
    # Get noise and detail estimates
    noise_level = estimate_noise(gray)
    detail_level = estimate_detail_level(gray)
    
    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Create a mask for face regions
    face_mask = np.zeros(gray.shape, dtype=np.uint8)
    for (x, y, w, h) in faces:
        face_mask[y:y+h, x:x+w] = 255
    
    # Adjust parameters based on noise level and presence of faces
    if len(faces) > 0:  # If faces detected, use very conservative parameters
        bilateral_d = 5
        bilateral_sigma = max(5, int(15 * noise_level))
        nlm_h = max(3, int(5 * noise_level))
    else:  # No faces, adjust based on noise and detail levels
        bilateral_d = 7 if noise_level > 0.5 else 5
        bilateral_sigma = max(15, int(50 * noise_level))
        nlm_h = max(5, int(10 * noise_level))
    
    # Apply bilateral filter
    bilateral = cv2.bilateralFilter(
        image, 
        d=bilateral_d,
        sigmaColor=bilateral_sigma,
        sigmaSpace=bilateral_sigma
    )
    
    # Apply non-local means with adjusted parameters
    nlm = cv2.fastNlMeansDenoisingColored(
        bilateral,
        None,
        h=nlm_h,
        hColor=nlm_h,
        templateWindowSize=5,
        searchWindowSize=15
    )
    
    # For face regions, blend more of the original image
    if len(faces) > 0:
        face_mask_3d = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2RGB)
        face_blend = cv2.addWeighted(
            image, 0.6,  # More weight to original in face regions
            nlm, 0.4,
            0
        )
        non_face_blend = cv2.addWeighted(
            image, 0.3,  # Less weight to original in non-face regions
            nlm, 0.7,
            0
        )
        # Combine face and non-face regions
        result = np.where(
            face_mask_3d > 0,
            face_blend,
            non_face_blend
        )
    else:
        # No faces detected, use standard blending
        result = cv2.addWeighted(image, 0.4, nlm, 0.6, 0)
    
    return result

def estimate_noise(gray_img):
    """
    Estimate noise level in the image
    """
    # Laplacian variance method
    lap = cv2.Laplacian(gray_img, cv2.CV_64F)
    noise_score = np.var(lap)
    # Normalize to 0-1 range
    return min(1.0, noise_score / 500.0)

def estimate_detail_level(gray_img):
    """
    Estimate level of detail in the image
    """
    # Use Sobel edges to detect detail
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Calculate detail score
    detail_score = np.mean(gradient_magnitude)
    # Normalize to 0-1 range
    return min(1.0, detail_score / 50.0)

def detect_faces(image):
    """
    Optional: Detect faces to apply different processing to facial regions
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces


