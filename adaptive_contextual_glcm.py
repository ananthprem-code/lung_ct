import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

# Define GLCM Parameters
DISTANCES = [1, 2, 3, 4]
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

def compute_adaptive_window(image, x, y, min_size=3, max_size=15):
    """
    Dynamically determine the local window size based on intensity variance.
    """
    h, w = image.shape
    window_size = min_size
    
    for size in range(min_size, max_size, 2):
        x1, x2 = max(0, x - size//2), min(h, x + size//2)
        y1, y2 = max(0, y - size//2), min(w, y + size//2)
        local_patch = image[x1:x2, y1:y2]
        
        if np.std(local_patch) > 10:  # Threshold for adaptive selection
            window_size = size
        else:
            break
    
    return window_size

def adaptive_contextual_glcm(image, distances=DISTANCES, angles=ANGLES):
    """
    Compute Adaptive Contextual Gray Level Co-occurrence Matrix (AC-GLCM)
    for an entire image using local adaptive windows.
    """
    h, w = image.shape
    padded_image = np.pad(image, pad_width=10, mode='reflect')  # Padding for boundary handling
    features = []

    for x in range(h):
        for y in range(w):
            win_size = compute_adaptive_window(image, x, y)
            x1, x2 = x, x + win_size
            y1, y2 = y, y + win_size
            local_patch = padded_image[x1:x2, y1:y2]

            if local_patch.shape[0] < 3 or local_patch.shape[1] < 3:
                continue  # Skip too small regions
            
            # Compute GLCM on adaptive window
            glcm = graycomatrix(local_patch, distances=distances, angles=angles, symmetric=True, normed=True)
            
            # Extract Features
            contrast = np.mean(graycoprops(glcm, 'contrast'))
            correlation = np.mean(graycoprops(glcm, 'correlation'))
            energy = np.mean(graycoprops(glcm, 'energy'))
            homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
            
            # Adaptive Dissimilarity
            dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity')) * (np.std(local_patch) / np.mean(local_patch + 1e-8))

            # Compute Local Entropy
            hist = cv2.calcHist([local_patch.astype(np.uint8)], [0], None, [256], [0, 256])
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist * np.log2(hist + 1e-8))

            features.append([contrast, correlation, energy, homogeneity, entropy, dissimilarity])

    return np.mean(features, axis=0)  # Return the average feature vector