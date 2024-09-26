# -*- coding: utf-8 -*-
"""
Created on Mon September 23 16:54:08 2024

@author: Ziv
"""

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def jacard_coef(y_true, y_pred):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection + 1.0)

model_path = 'trained_model.h5'
model = load_model(model_path, custom_objects={'dice_loss': sm.losses.DiceLoss(), 'jacard_coef': jacard_coef})

def preprocess_image(image_path, patch_size=1024):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.crop((0, 0, patch_size, patch_size))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(model, preprocessed_image):
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=-1)
    predicted_class = np.squeeze(predicted_class, axis=0)
    return predicted_class

def visualize_prediction(original_image_path, predicted_class):
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.subplot(1, 2, 2)
    plt.title("Predicted Segmentation")
    plt.imshow(predicted_class, cmap='jet')
    plt.show()

image_path = 'xxx.jpg'
preprocessed_image = preprocess_image(image_path)
predicted_class = predict_image(model, preprocessed_image)
visualize_prediction(image_path, predicted_class)
