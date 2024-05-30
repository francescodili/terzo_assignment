import  torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment



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
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
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

def extract_crops(pil_img, boxes):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    crops = []
    for (xmin, ymin, xmax, ymax) in boxes.tolist():
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
        size = (xmax - xmin, ymax - ymin)
        crop = cv2.getRectSubPix(img, patchSize=(int(size[0]), int(size[1])), center=(center[0], center[1]))
        crops.append(crop)
    return crops

def compute_similarity(box1, box2, ssim_score, features1, features2):
    iou = compute_iou(box1, box2)

    # Calcola la distanza del coseno tra le feature
    feature_distance = cosine(features1.flatten(), features2.flatten())
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



def extract_features(crops, model):
    resized_crops = [cv2.resize(crop, (224, 224)) for crop in crops]  # Ridimensiona tutte le immagini a 224x224
    rgb_crops = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in resized_crops]  # Converti tutte le immagini in RGB
    img_arrays = np.array([keras.utils.img_to_array(crop) for crop in rgb_crops])
    img_arrays = preprocess_input(img_arrays)
    
    features = model.predict(img_arrays)
    return features

def match_detections_to_tracks(detections, crops, tracks, resnet, iou_threshold=0.3, similarity_threshold=0.5):
    if len(tracks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 4), dtype=int)

    # Estrai le feature per tutti i crops delle detection e dei tracks
    detection_features = extract_features(crops, resnet)
    track_crops = [track.crop for track in tracks]
    track_features = extract_features(track_crops, resnet)

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    ssim_scores = np.zeros((len(tracks), len(detections)), dtype=np.float32)

    for t, trk in enumerate(tracks):
        for d, det in enumerate(detections):
            ssim_scores[t, d] = similarity_between_crops(trk.crop, crops[d])
            similarity = compute_similarity(trk.bbox, det, ssim_scores[t, d], track_features[t], detection_features[d])
            cost_matrix[t, d] = 1 - similarity  # Convert similarity to cost

    # Normalizzare i valori della matrice tra 0 e 1
    if cost_matrix.max() > cost_matrix.min():  # Evita la divisione per zero
        cost_matrix = (cost_matrix - cost_matrix.min()) / (cost_matrix.max() - cost_matrix.min())
    print("Cost matrix:\n", cost_matrix)

    matched_indices = linear_sum_assignment(cost_matrix)
    print("Matched indices:", matched_indices)

    matched_indices = list(zip(matched_indices[0], matched_indices[1]))
    unmatched_detections = list(set(range(len(detections))) - set(i for _, i in matched_indices))
    unmatched_tracks = list(set(range(len(tracks))) - set(t for t, _ in matched_indices))

    matches = []
    for t, d in matched_indices:
        if cost_matrix[t, d] <= 1 - similarity_threshold:
            matches.append((t, d))
        else:
            unmatched_detections.append(d)
            unmatched_tracks.append(t)

    print("Matches:", matches)
    print("Unmatched detections:", unmatched_detections)
    print("Unmatched tracks:", unmatched_tracks)

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)




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


def detect(model, im, transform = None, threshold_confidence = 0.7):
    if transform is None:
        # standard PyTorch mean-std input image normalization
        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with a confidence > threshold_confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold_confidence

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # filter to keep only the person class (class index 1)
    person_class_index = CLASSES.index('person')
    person_keep = probas[keep].argmax(-1) == person_class_index

    return probas[keep][person_keep], bboxes_scaled[person_keep]

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
        self.resnet = ResNet50(weights='imagenet', include_top=False)

    def add_track(self, bbox, crop):
        self.tracks[self.next_id] = Track(self.next_id, bbox, crop)
        self.next_id += 1

    def update_tracks(self, detected_bboxes, crops):
        current_tracks = list(self.tracks.values())
        matches, unmatched_detections, unmatched_tracks = match_detections_to_tracks(detected_bboxes, crops, current_tracks, self.resnet)

        print("Current tracks before update:", [(track.track_id, track.bbox) for track in current_tracks])
        print("Matches:", matches)
        print("Unmatched detections:", unmatched_detections)
        print("Unmatched tracks:", unmatched_tracks)

        for match in matches:
            track_idx, detection_idx = match
            if track_idx < len(current_tracks):
                track_id = current_tracks[track_idx].track_id
                self.tracks[track_id].update(detected_bboxes[detection_idx], crops[detection_idx])
            else:
                print(f"Warning: track_idx {track_idx} out of range for current_tracks")

        for track_idx in unmatched_tracks:
            if track_idx < len(current_tracks):
                track_id = current_tracks[track_idx].track_id
                self.tracks[track_id].increment_skipped_frames()
                if self.tracks[track_id].is_lost():
                    del self.tracks[track_id]
            else:
                print(f"Warning: track_idx {track_idx} out of range for current_tracks")

        for detection_idx in unmatched_detections:
            self.add_track(detected_bboxes[detection_idx], crops[detection_idx])

        print("Tracks after update:", [(track.track_id, track.bbox) for track in self.tracks.values()])

    def get_active_tracks(self):
        return [track for track in self.tracks.values() if not track.is_lost()]






def draw_boxes(img, boxes, ids, colors=None):
    img = np.array(img)
    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        color = colors[i % len(colors)] if colors else [0, 255, 0]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(img, f"ID: {ids[i]}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img

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
                    
                    # Crea la directory di output se non esiste
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
                        

                

if __name__=='__main__':
    main()