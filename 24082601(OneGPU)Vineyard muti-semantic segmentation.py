# -*- coding: utf-8 -*-
"""
Created on Wed July 3 16:37:08 2024

@author: Ziv
"""

print("\n\n")
print("╔═══════════════════════════╗")
print("║       START THE CODE12    ║")
print("╚═══════════════════════════╝")

###################  MultiWorkerMirroredStrategy  
import json
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #Specify GPU0 as the GPU to be used.
# os.environ.pop('TF_CONFIG', None) #Remove the environ. var. "TG_CONFIG". If TG_CONFIG unexist, return None. 
os.environ["SM_FRAMEWORK"] = "tf.keras" #Specfy the deep learning framework from Amazon SageMaker to Keras.

if '.' not in sys.path:
  sys.path.insert(0, '.') #Ensure the current directory('.') be added to the top of(0,_) python sys.path(where dictionary import from)
  
import tensorflow as tf

os.environ['TF_CONFIG'] = json.dumps({
    "cluster": {
        "worker": ["viscluster80:1111"]
    },
    "task": {"type": "worker", "index": 0}
})

print("TF_CONFIG:", os.getenv("TF_CONFIG"))
###################  MultiWorkerMirroredStrategy  


import cv2
import numpy as np
from PIL import Image
from patchify import patchify
import segmentation_models as sm 
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.metrics import MeanIoU
from sklearn.preprocessing import MinMaxScaler, StandardScaler


###################  MultiWorkerMirroredStrategy  
# MultiWorkerMirroredStrategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()
# strategy.scope()
###################  MultiWorkerMirroredStrategy  



scaler = MinMaxScaler()
root_directory = 'job_24-2024_08_29_15_10_06-segmentation mask 1.1'
patch_size = 1024

############################################################################
"""
<<Patching image for processing>>
1. Walk through the 'images' & 'masks' files as NumPy array.
2. Enumerate each of the image file.
3. Crop each of the image file's dimension to the mutiple of 256.
4. Patchify each of the image.
5. Point out each of the patched-subimage, then scale it to 0~1.
6. Append each of the afterscaler-patched-subimage into image dataset.
"""

print("\n\n\n----------------Start loading the image.----------------")   
print("(Start finding image file inside \"JEPGImages folder\")")
image_dataset = []  
#1. Walk through the 'images' & 'masks' files as NumPy array.
for path, subdirs, files in os.walk(root_directory): #Use "os.walk" walk through "root_directory" and assigns the value to path, subdirs and fies.
    print("\t","***Current path is:", path)
    dirname = path.split("/")[-1] #Use"os.path.sep" to obtain the sep-symbol of os, and split the path by the sep-symbol(windows uses backslashes), at last get the final element of the splited path. If the path contain forwardslashes, it won't be identified as a sep-symbol.  
    #print("After /, dirname is:", dirname)
    if "JPEGImages" in path:   #Find all 'images' directories
        images = os.listdir(path)  #List of all entries(files and directories) in subdirectory of path and return to "images(list)".
        print("\t!!Found JEPGImages folder!! The subdirectory of path is :", images)
        #2. Enumerate each of the image file.
        for i, image_name in enumerate(images):   #Go through the files name inside the "images(list)", enumerate give each file an index number and file's name(with .jpg) into "image_name" in order.
            print("\t","Enumerate the subdirectory of the current path :",i,image_name)
            if image_name.endswith(".JPG"):   #Only read jpg images...
                print("\t","(Images path is:",path+"/"+image_name,")")
                image = cv2.imread(path+"/"+image_name, 1)  #Read each image at "path+"/"+image_name, 1" as BGR
                SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest multiple of 256 = (ImageWidth/256)*256
                SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest multiple of 256 = (ImageHeigh/256)*256
                image = Image.fromarray(image) #"Image.fromarray()" function is to create an image object from NumPy array "image".(NOTICE! it must be capital I here since this fuction is "from PIL import Image")
                #3. Crop each of the image file's dimension to the mutiple of 256.
                image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner (0, 0) to right lower (SIZE_X, SIZE_Y).
                image = np.array(image)    
                #4. Patchify each of the image.
                patches_img = patchify(image, (patch_size, patch_size, 3), step= patch_size)  #Use function patchify(Step=256 for 256 patches means no overlap, 3 is RGB) from "image" and return each of the patch to "patches_img".
                #5. Point out each of the afterpatchify-subimage, then scale it to 0~1.
                for i in range(patches_img.shape[0]): #The total number of patches on Y-direction(heigh). (SyntaxNote:"variavble.shape[]")
                    for j in range(patches_img.shape[1]): #The total number of patches on X-direction(width)
                        #5. Point out each of the afterpatchify-subimage, then scale it to 0~1.
                        single_patch_img = patches_img[i,j,:,:] #Point out the patch in order and return to single_patch_img. (SyntaxNote:"variable[i,j,:,:]")
                        reshaped_patch_img = single_patch_img.reshape(-1, single_patch_img.shape[-1])
                        scaled_patch_img = scaler.fit_transform(reshaped_patch_img)
                        single_patch_img = scaled_patch_img.reshape(single_patch_img.shape)
                        single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds. From (1, 256, 256, 3) to (256, 256, 3)       
                        #6. Append each of the afterscaler-afterpatchify-subimage into image dataset.
                        image_dataset.append(single_patch_img)
                        
image_dataset = np.array(image_dataset) #Make the dispersed array format into 1 array.  
import random
import numpy as np
image_number = random.randint(0, len(image_dataset)) #Here, lengh of image_dataset is 1305 represent 1350 patched images.
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3))) 
plt.show()
print("\n\n\n----------------Start loading the mask-----------")   
print("(Start finding mask file inside \"SegmentationClass\")")                    
mask_dataset = []  
for path, subdirs, files in os.walk(root_directory):
    print("\t","***Current path is:", path)
    dirname = path.split("/")[-1]
    if "SegmentationClass" in path:   #Find all 'images' directories
        masks = os.listdir(path)  #List of all image names in this subdirectory
        print("\t!!Found SegmentationClass folder!! The subdirectory of path is :", masks)
        for i, mask_name in enumerate(masks):  
            print("\t","Enumerate the subdirectory of the current path :",i,mask_name)
            if mask_name.endswith(".png"):   #Only read png images... (masks in this dataset)
                print("\t","(Mask path is:",path+"/"+mask_name,")")
                mask = cv2.imread(path+"/"+mask_name, 1)  #Read each image as RGB.
                mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                SIZE_Y = (mask.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                mask = Image.fromarray(mask)
                mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                mask = np.array(mask)        
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step= patch_size)  #Step=256 for 256 patches means no overlap
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        single_patch_mask = patches_mask[i,j,:,:]
                        single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.    
                        mask_dataset.append(single_patch_mask) 
                        
image_dataset = np.array(image_dataset) #Make the dispersed array format into 1 array.  
mask_dataset =  np.array(mask_dataset)

import random
import numpy as np
image_number = random.randint(0, len(image_dataset)) #Here, lengh of image_dataset is 1305 represent 1350 patched images.
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3))) 
plt.subplot(122)
plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
plt.show()

import random
import numpy as np
image_number = random.randint(0, len(image_dataset)) #Here, lengh of image_dataset is 1305 represent 1350 patched images.
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3))) 
plt.subplot(122)
plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
plt.show()

import random
import numpy as np
image_number = random.randint(0, len(image_dataset)) #Here, lengh of image_dataset is 1305 represent 1350 patched images.
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3))) 
plt.subplot(122)
plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
plt.show()

############################################################################
"""
<<Convert RGB to Integer>> 
(RGB to HEX: (Hexadecimel --> base 16), 0-9 --> 0-9, 10-15 --> A-F)
1. Convert HEX to RGB array of each lebal.
2. Function converting each lebal with RGB(0-225) array to integer. 
3. Use the function to convert each of the patched-submaskImage from RGB to an integer.
4. Expand the array from 3D to 4D for input into model. (e.g. from (1305, 256, 256) to (1305, 256, 256, 1))
"""
#1. Convert HEX to RGB array of each lebal.
# Building = '#3C1098'.lstrip('#')
# Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
# Land = '#8429F6'.lstrip('#')
# Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
# Road = '#6EC1E4'.lstrip('#') 
# Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228
# Vegetation =  'FEDD3A'.lstrip('#') 
# Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58
# Water = 'E2A929'.lstrip('#') 
# Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41
# Unlabeled = '#9B9B9B'.lstrip('#') 
# Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155

background = [0,0,0]
Cultivated_vindyard = [36,179,83]
Abanded_cleared_farmland = [245,147,49]
Cleared_in_2023_but_not_maintained_farmland = [170,240,209]
Not_maintained_since_2023_fall_farmland = [51,221,255]
Not_cultivated_for_several_years = [115,51,128]
Deterioration_of_walls = [250,50,183]
Large_scale_landslide = [250,50,83]
Others = [143,143,143]

#2. Define a function converting each lebal with RGB(0-225) array to an integer. 
label = single_patch_mask #Dummy label for using below
def rgb_to_2D_label(label):
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label ==background,axis=-1)] = 0
    label_seg [np.all(label==Cultivated_vindyard,axis=-1)] = 1
    label_seg [np.all(label==Abanded_cleared_farmland,axis=-1)] = 2
    label_seg [np.all(label==Cleared_in_2023_but_not_maintained_farmland,axis=-1)] = 3
    label_seg [np.all(label==Not_maintained_since_2023_fall_farmland,axis=-1)] = 4
    label_seg [np.all(label==Not_cultivated_for_several_years,axis=-1)] = 5
    label_seg [np.all(label==Deterioration_of_walls,axis=-1)] = 6
    label_seg [np.all(label==Large_scale_landslide,axis=-1)] = 7
    label_seg [np.all(label==Others,axis=-1)] = 8
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    return label_seg
#3. Use the function to convert each of the patched-submaskImage from RGB to an integer.
labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)  
#4. Expand the array from 3D to 4D for input into model. (e.g. from (1305, 256, 256) to (1305, 256, 256, 1))
labels = np.array(labels)   
labels = np.expand_dims(labels, axis=3) #e.g. from (1305, 256, 256) to (1305, 256, 256, 1)
print("Unique labels in label dataset are: ", np.unique(labels))

import random
import numpy as np
image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:,:,0])
plt.show()

import random
import numpy as np
image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:,:,0])
plt.show()

import random
import numpy as np
image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:,:,0])
plt.show()

############################################################################
"""
1. Converts the class integers to binary class matrix(OneHotEncoder). (e.g. from (1305, 256, 256, 1) to (1305, 256, 256, 6))
2. Split the image_dataseet and labels_cat into traing group and testing group.
"""
n_classes = 9
#1. Converts the class integers to binary class matrix(OneHotEncoder). (e.g. from (1305, 256, 256, 1) to (1305, 256, 256, 6))
from tensorflow.keras.utils import to_categorical
labels_cat = to_categorical(labels, num_classes=n_classes)
#2. Split the image_dataseet and labels_cat into traing group and testing group.
from sklearn.model_selection import train_test_split
split_index = int(len(image_dataset) * 0.8)
X_train = image_dataset[:split_index]
X_test = image_dataset[split_index:]
y_train = labels_cat[:split_index]
y_test = labels_cat[split_index:]
 
############################################################################

weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

from simple_multi_unet_model import multi_unet_model, jacard_coef  
metrics=['accuracy', jacard_coef]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

###################  MultiWorkerMirroredStrategy  
with strategy.scope():
    model = get_model()
    model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
    model.summary()
    
    history1 = model.fit(X_train, y_train, 
                        batch_size = 12, 
                        verbose=1, 
                        epochs=500, 
                        validation_data=(X_test, y_test), 
                        shuffle=False)
###################  MultiWorkerMirroredStrategy  
############################################################################

import os
import matplotlib.pyplot as plt

# Create an output folder
output_folder = 'output_images090601(batchsize_12)'
os.makedirs(output_folder, exist_ok=True)

#IOU
y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)
y_test_argmax=np.argmax(y_test, axis=3)

# Iterate all the testing image
for test_img_number in range(len(X_test)):
    # Extract the testing image, real lebal, predictions
    test_img = X_test[test_img_number]
    ground_truth = y_test_argmax[test_img_number]
    test_img_input = np.expand_dims(test_img, 0)
    prediction = model.predict(test_img_input)
    predicted_img = np.argmax(prediction, axis=3)[0,:,:]

    # Show the images and output to the folder
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img)
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth)
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(predicted_img)
    output_filename = f'test_image_{test_img_number}_prediction.png'
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path)
    plt.close()

print("All images already saved to the output folder:", output_folder)

############################################################################
