import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
import os

def initialize_deepsort(model_path):
    max_cosine_distance = 0.2
    nn_budget = None
    encoder = gdet.create_box_encoder(model_path, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    return encoder, tracker

def detect_objects(model, image, transform=None):
    if transform is None:
        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    img = transform(image).unsqueeze(0)
    outputs = model(img)
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], image.size)
    return bboxes_scaled

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def compute_cost_matrix(detections1, detections2, encoder, image):
    features1 = encoder(image, detections1)
    features2 = encoder(image, detections2)
    cost_matrix = np.zeros((len(detections1), len(detections2)), dtype=np.float32)
    for i, f1 in enumerate(features1):
        for j, f2 in enumerate(features2):
            cost_matrix[i, j] = np.linalg.norm(f1 - f2)
    return cost_matrix

def track_objects(tracker, encoder, image, detections):
    features = encoder(image, detections)
    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(detections, features)]
    tracker.predict()
    tracker.update(detections)
    return tracker

def main():
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    data_path = '../MOT17/train'
    model_path = 'deep_sort/mars-small128.pb'
    encoder, tracker = initialize_deepsort(model_path)

    for dir in os.listdir(data_path):
        for video_dir in os.listdir(os.path.join(data_path, dir)):
            if video_dir == 'img1':
                for frame_path in os.listdir(os.path.join(data_path, dir, video_dir)):
                    image_path = os.path.join(data_path, dir, video_dir, frame_path)
                    img = Image.open(image_path)
                    detections = detect_objects(model, img)
                    if len(detections) > 0:
                        tracker = track_objects(tracker, encoder, np.array(img), detections)

if __name__ == '__main__':
    main()
