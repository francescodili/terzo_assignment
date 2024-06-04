import  torch
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import Tracker
import compare_crops
import detect_boxes
import validazione


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




def main():
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    data_path = '../MOT17/train'
    output_path = '../bbox/train_bbox'
    tracker = Tracker.Tracker(similarity_threshold=0.7)
    for dir in os.listdir(data_path):
        for video_dir in os.listdir(os.path.join(data_path, dir)):
            if video_dir == 'img1':
                video_output_path = os.path.join(output_path, dir, 'output.txt')
                os.makedirs(os.path.join(output_path, dir), exist_ok=True)

                with open(video_output_path, 'w') as file:
                    file.write("frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z\n")


                for frame_path in os.listdir(os.path.join(data_path, dir, video_dir)):
                    image_path = os.path.join(data_path, dir, video_dir, frame_path)
                    img = Image.open(image_path)
                    frame_number = int(frame_path.split('.')[0])
                    prob, bboxes_scaled = detect_boxes.detect(model, img)                  

                    crops = compare_crops.extract_crops(img, bboxes_scaled)
                    tracker.update_tracks(bboxes_scaled.tolist(), crops)
                    active_tracks = tracker.get_active_tracks()

                    validazione.save_boxes(active_tracks, frame_number, video_output_path, prob.tolist())

                    # Disegna le bounding box sull'immagine
                    annotated_img = detect_boxes.draw_boxes(img, [track.bbox for track in active_tracks], [track.track_id for track in active_tracks], detect_boxes.COLORS)
                    
                    # Salva l'immagine annotata
                    annotated_img_path = os.path.join(output_path, dir, f'{frame_path}_annotated.jpg')
                    cv2.imwrite(annotated_img_path, cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
                        

                
if __name__=='__main__':
    main()