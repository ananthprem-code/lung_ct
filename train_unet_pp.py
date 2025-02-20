import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from segmentation_models import UnetPlusPlus
from segmentation_models.losses import DiceLoss
from segmentation_models.metrics import iou_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths
IMG_DIR = r'C:\glt works\Prem Ananth\code\images\CT\lung_lession\imgs'
MASK_DIR = r'C:\glt works\Prem Ananth\code\images\CT\lung_lession\masks'
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 50

# Function to Apply CLAHE
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# Load Dataset
def load_data(img_dir, mask_dir, img_size):
    images, masks = [], []
    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_name, mask_name in zip(img_files, mask_files):
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)  # Resize
        img = apply_clahe(img)  # Apply CLAHE
        img = img / 255.0  # Normalize
        images.append(img)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=-1)  # Ensure correct shape
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load Images & Masks
X, Y = load_data(IMG_DIR, MASK_DIR, IMG_SIZE)

# Split Dataset (80% Train, 20% Validation)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build UNet++ Model
model = UnetPlusPlus(backbone_name='resnet34', encoder_weights='imagenet', 
                     input_shape=(256, 256, 3), classes=1, activation='sigmoid')

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=DiceLoss(), metrics=[iou_score])

# Train Model
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)

# Save Model
model.save("lung_lession_unetpp.h5")

# Plot Training Curve
plt.plot(history.history['iou_score'], label='Train IoU')
plt.plot(history.history['val_iou_score'], label='Val IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU Score')
plt.legend()
plt.title('Lung Lesion Segmentation Performance')
plt.show()