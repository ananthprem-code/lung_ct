import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC  # Using SVM as a baseline for PCNN classification

# Paths
TEST_FOLDER = r"C:\glt works\Prem Ananth\code\test_images"
UNET_MODEL_PATH = r"C:\glt works\Prem Ananth\code\models\unet_lung_seg.h5"
IMAGENET_MODEL_PATH = r"C:\glt works\Prem Ananth\code\models\imagenet_ct_classification.h5"
UNET_PP_MODEL_PATH = r"C:\glt works\Prem Ananth\code\models\unetplusplus_lung_lession.h5"
FEATURES_FILE = r"C:\glt works\Prem Ananth\code\traindb\test_features.pkl"
PCNN_MODEL_PATH = r"C:\glt works\Prem Ananth\code\models\pcnn_model.pkl"

# Load Trained Models
unet_model = load_model(UNET_MODEL_PATH)
imagenet_model = load_model(IMAGENET_MODEL_PATH)
unet_pp_model = load_model(UNET_PP_MODEL_PATH)


# Function to Apply CLAHE & Preprocess Image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = img / 255.0  # Normalize
    return img.reshape(1, 256, 256, 1)

# Function to Extract Lung Region Using UNET
def segment_lung(image_path):
    img = preprocess_image(image_path)
    lung_mask = unet_model.predict(img)[0, :, :, 0]
    lung_mask = (lung_mask > 0.5).astype(np.uint8)
    return lung_mask

# Function to Mask Image with Lung Region
def apply_mask(image_path, mask):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    return img * mask

# Function to Classify CT as Positive/Negative
def classify_ct(image):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = imagenet_model.predict(image)
    return np.argmax(prediction)  # 0 = Negative, 1 = Positive

# Function to Extract Lung Lesion Using UNET++
def segment_lesion(image_path):
    img = preprocess_image(image_path)
    lesion_mask = unet_pp_model.predict(img)[0, :, :, 0]
    lesion_mask = (lesion_mask > 0.5).astype(np.uint8)
    return lesion_mask

# Function to Extract AC-GLCM Features
def extract_ac_glcm_features(masked_image):
    glcm = adaptive_contextual_glcm(masked_image)
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    mean_intensity = np.mean(masked_image)
    std_intensity = np.std(masked_image)
    entropy = -np.sum((masked_image/255.0) * np.log2((masked_image/255.0) + 1e-10))
    return [contrast, correlation, energy, homogeneity, mean_intensity, std_intensity, entropy]

# Function to Process an Entire Folder
def process_folder(test_folder):
    features = []
    labels = []
    
    for category in os.listdir(test_folder):  # Iterate through categories
        category_path = os.path.join(test_folder, category)
        if os.path.isdir(category_path):
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                
                # Step 1: Segment Lung Region
                lung_mask = segment_lung(image_path)
                lung_segmented = apply_mask(image_path, lung_mask)
                
                # Step 2: Classify CT as Positive/Negative
                ct_class = classify_ct(lung_segmented)
                
                if ct_class == 1:  # Positive Case
                    # Step 3: Extract Lung Lesion
                    lesion_mask = segment_lesion(image_path)
                    lesion_segmented = apply_mask(image_path, lesion_mask)
                    
                    # Step 4: Extract Statistical & AC-GLCM Features
                    feature_vector = extract_ac_glcm_features(lesion_segmented)
                    features.append(feature_vector)
                    
                    # Labeling: 1 for Positive
                    labels.append(1)
                else:
                    features.append([0] * 7)  # Zero padding for negative cases
                    labels.append(0)  # Label as Negative
    
    return np.array(features), np.array(labels)

# Process and Extract Features
X_test, y_test = process_folder(TEST_FOLDER)

# Load PCNN Model
with open(PCNN_MODEL_PATH, "rb") as f:
    pcnn_model = pickle.load(f)

# Predict Using PCNN
y_pred = pcnn_model.predict(X_test)

# Evaluate Model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))