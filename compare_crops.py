import keras
from keras.applications.resnet50 import preprocess_input, decode_predictions
from skimage.feature import local_binary_pattern
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
import cv2
import numpy as np



def extract_crops(pil_img, boxes):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    crops = []
    for (xmin, ymin, xmax, ymax) in boxes.tolist():
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
        size = (xmax - xmin, ymax - ymin)
        crop = cv2.getRectSubPix(img, patchSize=(int(size[0]), int(size[1])), center=(center[0], center[1]))
        crops.append(crop)
    return crops



def normalize_similarity(value, min_val, max_val):
    """Normalizza il valore tra min_val e max_val a un range di 0 a 1."""
    return (value - min_val) / (max_val - min_val)

def compute_histogram_similarity(crop1, crop2):
    """Calcola la similarità tra gli istogrammi di colore normalizzati tra 0 e 1."""
    hist1 = cv2.calcHist([crop1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([crop2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return normalize_similarity(hist_similarity, -1, 1)

def compute_texture_similarity(crop1, crop2):
    """Calcola la similarità tra le texture usando LBP e la normalizza tra 0 e 1."""
    lbp1 = local_binary_pattern(cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY), 8, 1, method="uniform")
    lbp2 = local_binary_pattern(cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY), 8, 1, method="uniform")
    lbp_hist1, _ = np.histogram(lbp1, bins=np.arange(0, 10), range=(0, 9))
    lbp_hist2, _ = np.histogram(lbp2, bins=np.arange(0, 10), range=(0, 9))
    lbp_hist1 = lbp_hist1.astype("float32")
    lbp_hist2 = lbp_hist2.astype("float32")
    lbp_hist1 /= (lbp_hist1.sum() + 1e-6)
    lbp_hist2 /= (lbp_hist2.sum() + 1e-6)
    texture_similarity = cv2.compareHist(lbp_hist1, lbp_hist2, cv2.HISTCMP_CORREL)
    return normalize_similarity(texture_similarity, -1, 1)


def compute_similarity(box1, box2, ssim_score, features1, features2, crop1, crop2):
    iou = compute_iou(box1, box2)

    # Calcolo della similarità delle feature con normalizzazione
    feature_distance = cosine(features1.flatten(), features2.flatten())
    feature_similarity = 1 - feature_distance  # Normalizzazione
    feature_similarity = max(0, feature_similarity)  # Assicura che sia >= 0

    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    center_distance = np.linalg.norm(np.array(center1) - np.array(center2))
    max_dim = max(box1[2] - box1[0], box1[3] - box1[1])
    normalized_center_distance = center_distance / max_dim
    normalized_center_similarity = 1 - normalized_center_distance

    # Calcolo della similarità tra istogrammi di colore
    hist_similarity = compute_histogram_similarity(crop1, crop2)

    # Calcolo della similarità tra texture usando LBP
    texture_similarity = compute_texture_similarity(crop1, crop2)

    combined_similarity = (
        0.4 * feature_similarity + 
        0.2 * iou + 
        0.1 * ssim_score + 
        0.1 * normalized_center_similarity + 
        0.1 * hist_similarity + 
        0.1 * texture_similarity
    )
    
    return combined_similarity



def extract_features(crops, model):
    resized_crops = [cv2.resize(crop, (224, 224)) for crop in crops]  # Ridimensiona tutte le immagini a 224x224
    rgb_crops = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in resized_crops]  # Converti tutte le immagini in RGB
    img_arrays = np.array([keras.utils.img_to_array(crop) for crop in rgb_crops])
    img_arrays = preprocess_input(img_arrays)
    
    features = model.predict(img_arrays)
    return features

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def similarity_between_crops(crop1, crop2):
    crop1_gray = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    crop2_gray = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    if crop1_gray.shape != crop2_gray.shape:
        crop2_gray = cv2.resize(crop2_gray, (crop1_gray.shape[1], crop1_gray.shape[0]))

    score, _ = ssim(crop1_gray, crop2_gray, full=True)
    return score