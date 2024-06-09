import keras
from keras.applications.resnet50 import preprocess_input
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
import cv2
import numpy as np



def extract_crops(pil_img, boxes):
    """
    Estrae i crop data un immagine e la lista delle box (xmin, ymin, xmax, ymax)
    """
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    crops = []
    for (xmin, ymin, xmax, ymax) in boxes.tolist():
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
        size = (xmax - xmin, ymax - ymin)
        crop = cv2.getRectSubPix(img, patchSize=(int(size[0]), int(size[1])), center=(center[0], center[1]))
        crops.append(crop)
    return crops


def normalize_similarity(value, min_val, max_val):
    """
    Normalizza il valore tra min_val e max_val a un range di 0 a 1.
    """
    return (value - min_val) / (max_val - min_val)

def compute_histogram_similarity(crop1, crop2):
    """
    Calcola la similarità tra gli istogrammi di colore normalizzati tra 0 e 1.
    """
    hist1 = cv2.calcHist([crop1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([crop2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return normalize_similarity(hist_similarity, -1, 1)


def compute_similarity(box1, box2, ssim_score, features1, features2, crop1, crop2):
    """
    Calcola la similarità combinata tra:
    - Intersection over Union;
    - features estratte dai crop con resnet50;
    - distanza tra i centri delle box, convertita in similarità;
    - similarità strutturale tra crop (SSIM);
    - similarità tra gli istogrammi di colore

    Tutti i valori sono normalizzati tra 0 e 1, così la similarità restituita sarà 1 per immagini identiche e 0 per immagini diverse.
    """
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

    combined_similarity = (
        0.55 * feature_similarity + 
        0.2 * iou + 
        0.1 * ssim_score + 
        0.1 * normalized_center_similarity + 
        0.05 * hist_similarity
    )
    
    return combined_similarity



def extract_features(crops, model):
    """
    Prepara i crop per passarli al modello resnet50 ed estrae le features dai due crop.
    """
    resized_crops = [cv2.resize(crop, (224, 224)) for crop in crops]  # Ridimensiona tutte le immagini a 224x224
    rgb_crops = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in resized_crops]  # Converti tutte le immagini in RGB
    img_arrays = np.array([keras.utils.img_to_array(crop) for crop in rgb_crops])
    img_arrays = preprocess_input(img_arrays)
    
    features = model.predict(img_arrays)
    return features

def compute_iou(box1, box2):
    """
    Calcola l'Intersection over Union tra due bounding box.
    """
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
    """
    Calcola la similarità strutturale tra due crop.
    """
    crop1_gray = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    crop2_gray = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)

    # Verifica che entrambe le immagini siano almeno 7x7
    min_size = 7
    if crop1_gray.shape[0] < min_size or crop1_gray.shape[1] < min_size or crop2_gray.shape[0] < min_size or crop2_gray.shape[1] < min_size:
        return 0  # Ritorna 0

    if crop1_gray.shape != crop2_gray.shape:
        crop2_gray = cv2.resize(crop2_gray, (crop1_gray.shape[1], crop1_gray.shape[0]))
    
    
    score, _ = ssim(crop1_gray, crop2_gray, full=True)
    return score