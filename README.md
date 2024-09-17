# Vineyard-Segmentation
Automatic segmentation by cultivation status of vineyards.

## About The Project
This program is built on the Keras framework and uses drone imagery of vineyards to train a U-Net model for automatically identifying the status of farmland. It can detect various conditions, such as cultivated vineyards, abandoned fields, large-scale landslides, and more.

### Build With
Python (3.9.18)
* Deep Learning Framework:
    * TensorFlow (2.17.0)
    * Keras 
* Image Processing and Segmentation Libraries:
    * NumPy (1.26.4)
    * OpenCV (4.10.0.84)
    * PIL(Python Imaging Library) (10.4.0)
    * Patchify (0.2.3)
    * Segmentation Models(sm) (1.0.1)
* Visualization:
    * Matplotlib (3.9.1)
* Data Processing:
    * Scikit-learn (1.5.1)
* Additional Utility Libraries:
    * JSON
    * OS
    * SYS
    * Random

## Getting Started
This repository contain two code files. The main program executes by calling the Unet-model subprogram.

### Directory Struture
The following is an overview of the key files and folders in this project:


### Main Program Workflow

1. Environment Setup:
Ensure the current directory is in the system path.
Set the CUDA_VISIBLE_DEVICES environment variable to specify GPU usage.
Configure TensorFlow’s TF_CONFIG for distributed training using MultiWorkerMirroredStrategy.

2. Libraries Import:
Load libraries such as TensorFlow, OpenCV, NumPy, Matplotlib, and the segmentation model library (segmentation_models).

3. MultiWorkerMirroredStrategy:
Define the MultiWorkerMirroredStrategy for distributed training across workers.

4. Image Preprocessing:
Load and crop images from the JPEGImages folder to dimensions that are multiples of 1024.
Patchify the images (divide them into smaller patches of size 1024x1024) and scale the pixel values between 0 and 1.
Append the processed images to the dataset.

5. Mask Preprocessing:
Similarly, load and crop segmentation masks from the SegmentationClass folder.
Patchify and process the masks, converting them into usable format for training.

6. Label Encoding:

Convert the RGB values in the masks to integer labels representing different classes (e.g., background, cultivated vineyard, etc.).
Expand the labels from 3D to 4D for input into the model.

7. Dataset Splitting:
Split the dataset into training (80%) and testing (20%) groups.
Apply one-hot encoding to the labels.

8. Model Compilation:
Define a custom Unet model for segmentation using the multi_unet_model function.
Compile the model with Adam optimizer, combined dice and focal loss, and metrics including accuracy and Jaccard coefficient.

9. Model Training:

Train the model using MultiWorkerMirroredStrategy to distribute the training across multiple workers.
Set a batch size of 12 and train for 500 epochs.

10. Prediction and Output:
After training, make predictions on the test dataset.
Save the predicted images along with ground truth and test images in an output_images folder.

### Unet-Model Subprogram Workflow

### Recommended User-Ajustable Parameters
* Dataset
* Weight
* Patch_size
* Batch_size
* Epoch
* Training-Testing ratio of dataset

## Reference
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18 (pp. 234-241). Springer International Publishing.

Bhattiprolu, S. (2023). python_for_microscopists. GitHub. https://github.com/bnsreenu/python_for_microscopists/tree/master/228_semantic_segmentation_of_aerial_imagery_using_unet

### What is Unet model?