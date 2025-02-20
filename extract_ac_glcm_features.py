import os
import cv2
import numpy as np
import pandas as pd
import pickle
import adaptive_contextual_glcm
from skimage.feature import graycomatrix, graycoprops

# Paths
PCT_IMAGE_FOLDER = r"C:\glt works\Prem Ananth\code\images\CT\pct\images"
PCT_MASK_FOLDER = r"C:\glt works\Prem Ananth\code\images\CT\pct\masks"
NCT_IMAGE_FOLDER = r"C:\glt works\Prem Ananth\code\images\CT\nct\images"
NCT_MASK_FOLDER = r"C:\glt works\Prem Ananth\code\images\CT\nct\masks"
FEATURES_FILE = r"C:\glt works\Prem Ananth\code\traindb\features.pkl"
TARGET_FILE = r"C:\glt works\Prem Ananth\code\traindb\target.pkl"

# Function to Extract Features
def extract_ac_glcm_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

    # Apply CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    # Compute GLCM Matrix
    glcm = adaptive_contextual_glcm(img)

    # Extract GLCM Features
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))

    # Adaptive Contextual Statistical Features
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    entropy = -np.sum((img/255.0) * np.log2((img/255.0) + 1e-10))

    return [contrast, correlation, energy, homogeneity, mean_intensity, std_intensity, entropy]

# Function to Process Dataset
def process_dataset(image_folder, mask_folder, label):
    features_list = []
    target_list = []
    
    for filename in os.listdir(mask_folder):  # Use masks for consistency
        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, filename)

        if filename.endswith(('.png', '.jpg', '.jpeg')) and os.path.exists(image_path):
            features = extract_ac_glcm_features(image_path)
            features_list.append(features)
            target_list.append(label)  # 1 for PCT, 0 for NCT

    return features_list, target_list

# Process PCT (Positive) and NCT (Negative) Datasets
pct_features, pct_targets = process_dataset(PCT_IMAGE_FOLDER, PCT_MASK_FOLDER, label=1)
nct_features, nct_targets = process_dataset(NCT_IMAGE_FOLDER, NCT_MASK_FOLDER, label=0)

# Combine Features and Targets
all_features = np.vstack((pct_features, nct_features))
all_targets = np.hstack((pct_targets, nct_targets))

# Save as Pickle
with open(FEATURES_FILE, "wb") as f:
    pickle.dump(all_features, f)
with open(TARGET_FILE, "wb") as f:
    pickle.dump(all_targets, f)