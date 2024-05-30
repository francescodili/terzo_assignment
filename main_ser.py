import torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import torchvision.models as models
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
import logging



# Definizione delle classi e dei colori per la visualizzazione
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# Carica il modello ResNet-50 pre-addestrato
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Rimuovi l'ultimo strato di classificazione per ottenere le feature
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))

# Trasformazioni per le immagini di input
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(crop):
    input_tensor = preprocess(crop).unsqueeze(0)
    with torch.no_grad():
        features = resnet(input_tensor)
    return features.squeeze().numpy()

# Funzioni per la gestione delle bounding box
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def draw_boxes(img, boxes, ids, colors=None):
    img = np.array(img)
    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        color = colors[i % len(colors)] if colors else [0, 255, 0]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(img, f"ID: {ids[i]}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img

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

def compute_similarity(box1, box2, crop1, crop2):
    iou = compute_iou(box1, box2)
    ssim_score = similarity_between_crops(crop1, crop2)

    # Estrai le feature dalle immagini ritagliate
    features1 = extract_features(crop1)
    features2 = extract_features(crop2)

    # Calcola la distanza del coseno tra le feature
    feature_distance = cosine(features1.flatten(), features2.flatten())

    # Normalizza la distanza del coseno
    feature_similarity = 1 - feature_distance

    # Calcola la distanza tra i centri delle bounding box
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    center_distance = np.linalg.norm(np.array(center1) - np.array(center2))

    # Normalizza la distanza del centro
    max_dim = max(box1[2] - box1[0], box1[3] - box1[1])
    normalized_center_distance = center_distance / max_dim

    # Calcola una similarità combinata
    combined_similarity = 0.4 * iou + 0.3 * ssim_score + 0.2 * feature_similarity - 0.1 * normalized_center_distance
    return combined_similarity


logging.basicConfig(level=logging.DEBUG)

def match_detections_to_tracks(detections, crops, tracks, iou_threshold=0.3, similarity_threshold=0.5):
    if len(tracks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 4), dtype=int)

    cost_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(tracks):
            similarity = compute_similarity(det, trk.bbox, crops[d], trk.crop)
            cost_matrix[d, t] = 1 - similarity  # Convert similarity to cost

    logging.debug(f'Cost Matrix: \n{cost_matrix}')

    matched_indices = linear_sum_assignment(cost_matrix)

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[0]:
            unmatched_detections.append(d)

    unmatched_tracks = []
    for t in range(len(tracks)):
        if t not in matched_indices[1]:
            unmatched_tracks.append(t)

    matches = []
    for m in zip(matched_indices[0], matched_indices[1]):
        if cost_matrix[m[0], m[1]] > 1 - similarity_threshold:
            unmatched_detections.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m)

    logging.debug(f'Matches: {matches}')
    logging.debug(f'Unmatched Detections: {unmatched_detections}')
    logging.debug(f'Unmatched Tracks: {unmatched_tracks}')

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)




class Track:
    def __init__(self, track_id, bbox, crop, max_frames_to_skip=5):
        self.track_id = track_id
        self.bbox = bbox
        self.crop = crop
        self.max_frames_to_skip = max_frames_to_skip
        self.skipped_frames = 0

    def update(self, bbox, crop):
        self.bbox = bbox
        self.crop = crop
        self.skipped_frames = 0

    def increment_skipped_frames(self):
        self.skipped_frames += 1

    def is_lost(self):
        return self.skipped_frames > self.max_frames_to_skip

class Tracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def add_track(self, bbox, crop):
        self.tracks[self.next_id] = Track(self.next_id, bbox, crop)
        self.next_id += 1

    def update_tracks(self, detected_bboxes, crops):
        current_tracks = list(self.tracks.values())
        matches, unmatched_detections, unmatched_tracks = match_detections_to_tracks(detected_bboxes, crops, current_tracks)
        
        for match in matches:
            detection_idx, track_idx = match
            track_id = current_tracks[track_idx].track_id
            self.tracks[track_id].update(detected_bboxes[detection_idx], crops[detection_idx])

        for track_idx in unmatched_tracks:
            track_id = current_tracks[track_idx].track_id
            self.tracks[track_id].increment_skipped_frames()
            if self.tracks[track_id].is_lost():
                del self.tracks[track_id]

        for detection_idx in unmatched_detections:
            self.add_track(detected_bboxes[detection_idx], crops[detection_idx])

    def get_active_tracks(self):
        return [track for track in self.tracks.values() if not track.is_lost()]



# Funzioni di supporto per il ritaglio e la salvaguardia delle immagini
def extract_crops(pil_img, boxes):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    crops = []
    for (xmin, ymin, xmax, ymax) in boxes.tolist():
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
        size = (xmax - xmin, ymax - ymin)
        crop = cv2.getRectSubPix(img, patchSize=(int(size[0]), int(size[1])), center=(center[0], center[1]))
        crops.append(crop)
    return crops



def save_boxes(tracks, save_path):
    with open(save_path, 'w') as file:
        for track in tracks:
            file.flush()
            os.fsync(file.fileno())

            bbox = track.bbox
            line = f'{track.track_id}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\n'
            file.write(line)
    if not file.closed:
        file.close()

def detect(model, im, transform=None, threshold_confidence=0.7):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    img = transform(im).unsqueeze(0)

    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    outputs = model(img)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold_confidence

    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    person_class_index = CLASSES.index('person')
    person_keep = probas[keep].argmax(-1) == person_class_index

    return probas[keep][person_keep], bboxes_scaled[person_keep]

# Funzione principale
def main():
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    data_path = '../MOT17/train'
    output_path = '../bbox/train_bbox'
    tracker = Tracker()

    for dir in os.listdir(data_path):
        for video_dir in os.listdir(os.path.join(data_path, dir)):
            if video_dir == 'img1':
                for frame_path in os.listdir(os.path.join(data_path, dir, video_dir)):
                    image_path = os.path.join(data_path, dir, video_dir, frame_path)
                    img = Image.open(image_path)
                    prob, bboxes_scaled = detect(model, img)

                    output_dir = os.path.join(output_path, dir)
                    os.makedirs(output_dir, exist_ok=True)

                    save_path = os.path.join(output_dir, frame_path)
                    det_path = os.path.join(output_dir, f'{frame_path}_det.txt')

                    crops = extract_crops(img, bboxes_scaled)
                    tracker.update_tracks(bboxes_scaled.tolist(), crops)
                    active_tracks = tracker.get_active_tracks()

                    save_boxes(active_tracks, det_path)

                    # Disegna le bounding box sull'immagine
                    annotated_img = draw_boxes(img, [track.bbox for track in active_tracks], [track.track_id for track in active_tracks], COLORS)
                    
                    # Salva l'immagine annotata
                    annotated_img_path = os.path.join(output_dir, f'{frame_path}_annotated.jpg')
                    cv2.imwrite(annotated_img_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main()
