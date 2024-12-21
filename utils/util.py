import matplotlib.pyplot as plt


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
        