# Old Photo Restoration Project

Welcome to the **Old Photo Restoration Project**! This project aims to restore old and damaged photographs using custom image processing techniques. By reducing noise, removing scratches, enhancing contrast and brightness, and sharpening the images, we create visually improved outputs.

---

## **Project Overview**

The project uses Python and popular libraries such as NumPy, Matplotlib, and PIL to implement image restoration from scratch. Techniques include noise reduction, scratch and blemish removal, contrast and brightness adjustments, and sharpness enhancement. Our modular design ensures that each technique is standalone, making it easy to test, refine, and combine into a comprehensive restoration pipeline.

---

## **Features**
- **Noise Reduction:** Gaussian and Median filters for effective denoising.
- **Scratch and Blemish Removal:** Binary mask creation and interpolation-based inpainting.
- **Contrast and Brightness Adjustment:** Histogram equalization and manual adjustments.
- **Sharpness Enhancement:** Custom unsharp masking implementation.
- **Pipeline Functionality:** Combine restoration techniques for a seamless workflow.
- **Evaluation Metrics:** Quantitative (PSNR, SSIM) and qualitative (visual comparison and feedback) assessments.

---

## **Roadmap**
1. **Project Setup and Research**
   - Configure development environment with necessary libraries.
   - Research core techniques for restoration.
2. **Data Collection and Preprocessing**
   - Collect and preprocess a dataset of old and damaged photos.
3. **Implement Custom Restoration Techniques**
   - Noise reduction with Gaussian and Median filters.
   - Scratch removal using manual masking and inpainting techniques.
   - Contrast and brightness adjustments.
   - Sharpness enhancement with unsharp masking.
4. **Combine Techniques**
   - Create a restoration pipeline to apply all techniques sequentially.
5. **Evaluation and Testing**
   - Use PSNR and SSIM for quantitative evaluation.
   - Gather qualitative feedback and refine techniques.
6. **Final Presentation**
   - Prepare a comprehensive presentation and report detailing the methods, results, and challenges.

---

## **Installation**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YourUsername/Old-Photo-Restoration.git
   cd Old-Photo-Restoration
