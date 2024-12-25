import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from utils.util import *

from contextlib import contextmanager

# Simple PPM image writer for RGB images (no alpha)
def save_ppm(filepath, image_rgb):
    """
    Save an image in raw PPM format (P6).
    image_rgb is assumed to be a NumPy-like 3D list: [height][width][3].
    """
    height = len(image_rgb)
    width = len(image_rgb[0]) if height > 0 else 0
    with open(filepath, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        for row in image_rgb:
            for pixel in row:
                # Each pixel is [R, G, B]
                # Ensure values are in 0-255
                r = max(0, min(255, pixel[0]))
                g = max(0, min(255, pixel[1]))
                b = max(0, min(255, pixel[2]))
                f.write(bytes([r, g, b]))

# Simple PPM reader for grayscale images (P5)
def load_pgm(filepath):
    """
    Load a grayscale image in PGM format (P5).
    Returns a 2D list (height x width) of integer pixel values in [0..255].
    """
    with open(filepath, "rb") as f:
        header = f.readline().decode("ascii").strip()  # e.g. "P5"
        if header != "P5":
            raise ValueError("Not a valid P5 PGM file")
        # Skip comment lines if any
        line = f.readline().decode("ascii").strip()
        while line.startswith("#"):
            line = f.readline().decode("ascii").strip()
        # Now line should have width & height
        width, height = map(int, line.split())
        maxval = int(f.readline().decode("ascii").strip())  # typically 255
        # Read pixel data
        data = f.read(width * height)
        # Build 2D list
        image_gray = []
        idx = 0
        for _ in range(height):
            row = []
            for _ in range(width):
                row.append(data[idx])
                idx += 1
            image_gray.append(row)
        return image_gray

def rgb_to_bgr(image_rgb):
    """
    Convert an image from RGB to BGR by manually swapping channels.
    image_rgb is 3D: [height][width][3].
    """
    height = len(image_rgb)
    width = len(image_rgb[0]) if height > 0 else 0
    image_bgr = []
    for i in range(height):
        row = []
        for j in range(width):
            r, g, b = image_rgb[i][j]
            row.append([b, g, r])  # swapped
        image_bgr.append(row)
    return image_bgr

# Minimal (very naive) scratch inpainting that just copies neighbors
def naive_inpaint(image_rgb, mask_gray):
    """
    Naive inpainting: for each nonzero mask pixel, replace it with average of valid neighbors.
    This is a placeholder for demonstration.
    """
    height = len(image_rgb)
    width = len(image_rgb[0]) if height > 0 else 0
    # Convert to floats for averaging
    new_image = [[[float(c) for c in pixel] for pixel in row] for row in image_rgb]

    for i in range(height):
        for j in range(width):
            if mask_gray[i][j] > 10:  # threshold for "scratch"
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < height and 0 <= nj < width and mask_gray[ni][nj] <= 10:
                            neighbors.append(new_image[ni][nj])
                if neighbors:
                    avg_r = sum(p[0] for p in neighbors) / len(neighbors)
                    avg_g = sum(p[1] for p in neighbors) / len(neighbors)
                    avg_b = sum(p[2] for p in neighbors) / len(neighbors)
                    new_image[i][j] = [avg_r, avg_g, avg_b]

    # Clip and convert back to int
    for i in range(height):
        for j in range(width):
            new_image[i][j] = [
                min(255, max(0, int(new_image[i][j][0]))),
                min(255, max(0, int(new_image[i][j][1]))),
                min(255, max(0, int(new_image[i][j][2]))),
            ]
    return new_image

def load_image_rgb(path):
    """
    Load image from disk in BGR, convert manually to RGB, and return as a list of lists.
    """
    bgr_img = cv.imread(path)  # allowed for loading
    if bgr_img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    # Convert to RGB
    height, width, _ = bgr_img.shape
    image_rgb = []
    for i in range(height):
        row = []
        for j in range(width):
            b, g, r = bgr_img[i, j]
            row.append([r, g, b])
        image_rgb.append(row)
    return image_rgb

def show_image_rgb(image_rgb):
    """
    Display an RGB image using matplotlib.pyplot (allowed for showing).
    """
    arr = np.array(image_rgb, dtype=np.uint8)
    plt.imshow(arr)
    plt.axis('off')
    plt.show()

# %%
@contextmanager
def temporary_directory_change(directory):
    """Context manager for temporary change of working directory."""
    original_directory = os.getcwd()
    print(original_directory)
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(original_directory)


# now we need to create a mask for the scratch
# we will create it with using a trained model

def generate_mask(image_rgb):
    """
    Automates the mask generation process using the Bringing-Old-Photos-Back-to-Life repository.

    """
    # Convert from RGB to BGR
    image_bgr = rgb_to_bgr(image_rgb)

    # Save the image as PPM
    input_dir = os.path.abspath("mask_generation\\input")
    output_dir = os.path.abspath("mask_generation\\output")
    if os.path.exists(input_dir):
        try:
            import shutil
            shutil.rmtree(input_dir)
            os.makedirs(input_dir)
        except PermissionError:
            pass
    if os.path.exists(output_dir):
        try:
            import shutil
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        except PermissionError:
            pass

    # write the image to the path: "mask_generation/input"

    # convert RGB to BGR before saving
    # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) -- implemented this manually in util.py
    image_bgr = rgb_to_bgr(image)


    cv2.imwrite(os.path.join(input_dir, "image.png"), image_bgr)


    # Run the external detection script
    command = [
        "python",
        "detection.py",
        "--test_path", input_dir,
        "--output_dir", output_dir,
        "--input_size", "full_size",
        "--GPU", "-1"
    ]
    script_dir = r"C:\\Users\\yasir\\Desktop\\image_project\\Bringing-Old-Photos-Back-to-Life\\Global"
    
    # Run the command
    with temporary_directory_change(script_dir):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("Mask generation completed successfully.")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error occurred while generating masks:")
            print(e.stderr)
            
    mask = cv2.imread("mask_generation\\output\\mask\\image.png", cv2.IMREAD_GRAYSCALE)
    # Ensure mask matches input dimensions
    if mask.shape[:2] != image.shape[:2]:
        # mask = cv2.resize(mask, (image.shape[1], image.shape[0])) -- implemented this manually in util.py
        mask = resize_mask(mask, image.shape[0], image.shape[1])
    return mask
    

# # %%
# mask = generate_mask(img)

# # display the mask
# plt.imshow(mask, cmap="gray")
# plt.show()


# # %%
# # now we will inpaint the scratch using the mask
# # telea inpainting
# inpainted_image = cv.inpaint(img, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)

# # display the inpainted image and the original image

# fig, ax = plt.subplots(1, 2, figsize=(15, 15))
# ax[0].imshow(img)
# ax[0].set_title("Original Image")
# ax[0].axis("off")

# ax[1].imshow(inpainted_image)
# ax[1].set_title("Inpainted Image")
# ax[1].axis("off")

# plt.show()

# # %%
def inpaint_scratches(image):
    """
    Inpaints the scratches in the input image using a naive approach.
    """
    print("Image shape:", image.shape)
    # generate the mask
    mask = generate_mask(image)
    print("Mask shape:", mask.shape)
    # inpaint the image
    # inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA) -- implemented this manually in util.py
    inpainted_image = telea_inpaint(image, mask)

    return inpainted_image


# %%

