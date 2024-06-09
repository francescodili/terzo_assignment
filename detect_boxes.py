import  torch
import torchvision.transforms as T
import cv2
import numpy as np
import random
from scipy.optimize import linear_sum_assignment
import compare_crops

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


def draw_boxes(img, boxes, ids, colors):
    img = np.array(img)
    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        color = colors[ids[i]]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(img, f"ID: {ids[i]}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img

def generate_unique_colors(n):
    random.seed(42)
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n)]


def match_detections_to_tracks(detections, crops, tracks, resnet, similarity_threshold):
    if len(tracks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 4), dtype=int)

    detection_features = compare_crops.extract_features(crops, resnet)
    track_crops = [track.crop for track in tracks]
    track_features = compare_crops.extract_features(track_crops, resnet)

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    ssim_scores = np.zeros((len(tracks), len(detections)), dtype=np.float32)

    for t, trk in enumerate(tracks):
        for d, det in enumerate(detections):
            ssim_scores[t, d] = compare_crops.similarity_between_crops(trk.crop, crops[d])
            similarity = compare_crops.compute_similarity(trk.bbox, det, ssim_scores[t, d], track_features[t], detection_features[d], trk.crop, crops[d])
            cost_matrix[t, d] = 1 - similarity  # Convert similarity to cost

    if cost_matrix.max() > cost_matrix.min():  # Evita la divisione per zero
        cost_matrix = (cost_matrix - cost_matrix.min()) / (cost_matrix.max() - cost_matrix.min())


    matched_indices = linear_sum_assignment(cost_matrix)
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

    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)