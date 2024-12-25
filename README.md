# Old Photo Restoration Project

Welcome to the **Old Photo Restoration Project**! This project aims to restore old and damaged photographs using custom image processing techniques. By reducing noise, removing scratches, enhancing contrast and brightness, and sharpening the images, we create visually improved outputs.

---

## Installation

This repository is dependent to some other repositories, so we need to install them first. We need them to automatize mask generation.

First we need to clone this repository in the same folder level of this repository.
```
git clone https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life.git
```

The project folder structure should be like this.

![{F5C77979-DF2E-43AD-9848-C4CA4064DCA7}](https://github.com/user-attachments/assets/460742d1-c7a7-4248-8ef5-752b863673b7)


Then, we should enter the directory 'Bringing-Old-Photos-Back-to-Life'.

After that we should do the followings:

Clone the Synchronized-BatchNorm-PyTorch repository for

```
cd Face_Enhancement/models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../../
```

```
cd Global/detection_models
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

Download the landmark detection pretrained model

```
cd Face_Detection/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
cd ../
```

Download the pretrained model, put the file `Face_Enhancement/checkpoints.zip` under `./Face_Enhancement`, and put the file `Global/checkpoints.zip` under `./Global`. Then unzip them respectively.

```
cd Face_Enhancement/
wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/face_checkpoints.zip
unzip face_checkpoints.zip
cd ../
cd Global/
wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip
unzip global_checkpoints.zip
cd ../
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Go back to our repository's directory and run the GUI:

```bash
cd ../
cd old_photo_restoration
python GUI.py
```

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
