# %%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from contextlib import contextmanager
import subprocess
import os
import shutil

# %%
# Load the image
img = cv.imread("photos\\old_w_scratch\\d.png", cv.IMREAD_COLOR)
# Convert the image to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Check if the image was loaded successfully
if img is None:
    raise FileNotFoundError("The image file could not be found or loaded.")


# %%
# display the image with matplotlib
plt.imshow(img)
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

def generate_mask(image):
    """
    Automates the mask generation process using the Bringing-Old-Photos-Back-to-Life repository.

    Parameters:
        test_path (str): Path to the input images.
        output_dir (str): Path where output masks will be saved.
        input_size (str): Size of the input image, default is 'full_size'.
        gpu (str): GPU setting, default is '-1' (use CPU).
    """

    # get the full path of the directories in the 'mask_generation' folder
    # get the full path of the directory 'input' in the 'mask_generation'
    input_dir = os.path.abspath("mask_generation\\input")
    # get the full path of the directory 'output' in the 'mask_generation'
    output_dir = os.path.abspath("mask_generation\\output")


    # remove the content inside these directories
    for directory in [input_dir, output_dir]:
        try:
            shutil.rmtree(directory)
            os.makedirs(directory)
        except PermissionError:
            print(f"Permission denied: {directory}")

    # write the image to the path: "mask_generation/input"

    # convert RGB to BGR before saving
    image_bgr = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    cv.imwrite(os.path.join(input_dir, "image.png"), image_bgr)


    # Command to execute
    command = [
        "python",
        "detection.py",
        "--test_path", input_dir,
        "--output_dir", output_dir,
        "--input_size", "full_size",
        "--GPU", "-1"
    ]
    
    # Change directory to the script's location
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
            
    mask = cv.imread("mask_generation\\output\\mask\\image.png", cv.IMREAD_GRAYSCALE)
    return mask
    

# %%
mask = generate_mask(img)

# display the mask
plt.imshow(mask, cmap="gray")
plt.show()


# %%
# now we will inpaint the scratch using the mask
# telea inpainting
inpainted_image = cv.inpaint(img, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)

# display the inpainted image and the original image

fig, ax = plt.subplots(1, 2, figsize=(15, 15))
ax[0].imshow(img)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(inpainted_image)
ax[1].set_title("Inpainted Image")
ax[1].axis("off")

plt.show()


