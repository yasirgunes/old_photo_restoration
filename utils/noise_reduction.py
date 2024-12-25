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
# noise_reduction("old_photo_02.jpg")


def adaptive_noise_reduction(image):
    """
    Adaptive noise reduction that adjusts parameters based on image content and noise levels
    """
    print("Inside the adaptive_noise_reduction function")
    # Convert to grayscale for analysis
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) -- implemented this manually in utils/util.py
    # gray = rgb_to_gray(image)
    gray = rgb_to_gray(image).astype(np.uint8)
    print("Converted to grayscale")
    
    # Get noise and detail estimates
    noise_level = estimate_noise(gray)
    print("Estimated noise level")
    detail_level = estimate_detail_level(gray)
    print("Estimated detail level")
    
    # Detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print("Detected faces")
    
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
    
    print("Adjusted parameters to noise level and presence of faces")
    # Apply bilateral filter
    # bilateral = cv2.bilateralFilter(
    #     image, 
    #     d=bilateral_d,
    #     sigmaColor=bilateral_sigma,
    #     sigmaSpace=bilateral_sigma
    # )
    bilateral = bilateral_filter(image, diameter=bilateral_d, sigma_color=bilateral_sigma, sigma_space=bilateral_sigma)
    print("Applied bilateral filter")
    
    
    # Apply non-local means with adjusted parameters
    nlm = cv2.fastNlMeansDenoisingColored(
        bilateral,
        None,
        h=nlm_h,
        hColor=nlm_h,
        templateWindowSize=5,
        searchWindowSize=15
    )
    print("Applied non-local means")
    
    # For face regions, blend more of the original image
    print("Blending face regions")
    if len(faces) > 0:
        print("Faces detected")
        face_mask_3d = gray_to_rgb(face_mask)
        # face_blend = cv2.addWeighted(
        #     image, 0.6,  # More weight to original in face regions
        #     nlm, 0.4,
        #     0
        # )
        face_blend = add_weighted(image, 0.6, nlm, 0.4, 0)
        
        # non_face_blend = cv2.addWeighted(
        #     image, 0.3,  # Less weight to original in non-face regions
        #     nlm, 0.7,
        #     0
        # )
        non_face_blend = add_weighted(image, 0.3, nlm, 0.7, 0)
        # Combine face and non-face regions
        result = np.where(
            face_mask_3d > 0,
            face_blend,
            non_face_blend
        )
    else:
        print("No faces detected")
        # No faces detected, use standard blending
        # result = cv2.addWeighted(image, 0.4, nlm, 0.6, 0)
        result = add_weighted(image, 0.4, nlm, 0.6, 0)
    
    return result

def estimate_noise(gray_img):
    """
    Estimate noise level in the image
    """
    # Laplacian variance method
    # lap = cv2.Laplacian(gray_img, cv2.CV_64F)
    lap = laplacian(gray_img)
    noise_score = np.var(lap)
    # Normalize to 0-1 range
    return min(1.0, noise_score / 500.0)

def estimate_detail_level(gray_img):
    """
    Estimate level of detail in the image
    """
    # Use Sobel edges to detect detail
    # sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = sobel(gray_img, 1, 0, ksize=3)
    # sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    sobely = sobel(gray_img, 0, 1, ksize=3)
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
    gray = rgb_to_gray(image).astype(np.uint8)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces


