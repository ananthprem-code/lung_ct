import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set paths
IMAGE_PATH = r'C:\glt works\Prem Ananth\code\images\CT\lung_region\imgs'
MASK_PATH = r'C:\glt works\Prem Ananth\code\images\CT\lung_region\msks'
IMG_SIZE = (256, 256)  # Resize images to 256x256
BATCH_SIZE = 16
EPOCHS = 50

# Function to apply CLAHE
def apply_CLAHE(image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
    l, a, b = cv2.split(lab)  # Split channels
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHE setup
    cl = clahe.apply(l)  # Apply CLAHE on the L-channel
    lab = cv2.merge((cl, a, b))  # Merge back LAB channels
    final_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # Convert back to BGR
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)  # Convert back to grayscale
    return final_img

# Load images and masks with preprocessing (Resizing + CLAHE)
def load_data(image_path, mask_path):
    images, masks = [], []
    img_filenames = sorted(os.listdir(image_path))
    mask_filenames = sorted(os.listdir(mask_path))
    
    for img_file, mask_file in zip(img_filenames, mask_filenames):
        img = cv2.imread(os.path.join(image_path, img_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_path, mask_file), cv2.IMREAD_GRAYSCALE)
        
        img = cv2.resize(img, IMG_SIZE)  # Resize image
        mask = cv2.resize(mask, IMG_SIZE)  # Resize mask

        img = apply_CLAHE(img)  # Apply CLAHE to image

        img = img / 255.0  # Normalize image
        mask = mask / 255.0  # Normalize mask
        
        images.append(img.reshape(256, 256, 1))  # Reshape for CNN
        masks.append(mask.reshape(256, 256, 1))

    return np.array(images), np.array(masks)

# Load dataset
X, Y = load_data(IMAGE_PATH, MASK_PATH)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define UNet model
def build_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u6 = UpSampling2D(size=(2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D(size=(2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = UpSampling2D(size=(2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = UpSampling2D(size=(2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Train UNet model
unet_model = build_unet()
history = unet_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

# Save model
unet_model.save("lung_segmentation_unet.h5")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('UNet Training Performance')
plt.show()