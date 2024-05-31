import os
import cv2
import numpy as np
import pickle

def read_gt_file(gt_file_path):
    with open(gt_file_path, 'r') as file:
        lines = file.readlines()
    gt_data = [line.strip().split(',') for line in lines]
    return gt_data

def extract_crops_from_frame(frame, bboxes, crop_size=(224, 224)):
    crops = []
    frame_height, frame_width = frame.shape[:2]
    for bbox in bboxes:
        x, y, w, h = map(int, bbox)
        if x < 0 or y < 0 or x + w > frame_width or y + h > frame_height:
            continue  # Salta le bounding box fuori dai limiti
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            continue  # Salta i crop vuoti
        crop_resized = cv2.resize(crop, crop_size)
        crops.append(crop_resized)
    return crops

def create_pairs_and_labels(gt_data, frames_dict, crop_size=(224, 224)):
    pairs = []
    labels = []

    frame_dict = {}
    for data in gt_data:
        frame_id = int(data[0])
        obj_id = int(data[1])
        bbox = list(map(int, data[2:6]))
        if frame_id not in frame_dict:
            frame_dict[frame_id] = []
        frame_dict[frame_id].append((obj_id, bbox))

    for frame_id in sorted(frame_dict.keys()):
        if frame_id not in frames_dict:
            continue
        frame = frames_dict[frame_id]
        obj_bboxes = frame_dict[frame_id]
        crops = extract_crops_from_frame(frame, [bbox for _, bbox in obj_bboxes], crop_size)

        for i in range(len(crops)):
            for j in range(i+1, len(crops)):
                pairs.append([crops[i], crops[j]])
                if obj_bboxes[i][0] == obj_bboxes[j][0]:
                    labels.append(0)
                else:
                    labels.append(1)

    return np.array(pairs), np.array(labels)

def main(data_path, gt_filename, output_filename, crop_size=(224, 224)):
    pairs_list = []
    labels_list = []

    for video_dir in os.listdir(data_path):
        gt_file_path = os.path.join(data_path, video_dir, gt_filename)
        frames_path = os.path.join(data_path, video_dir, 'img1')

        if not os.path.exists(gt_file_path):
            continue

        gt_data = read_gt_file(gt_file_path)
        
        frames_dict = {}
        for frame_file in os.listdir(frames_path):
            frame_id = int(frame_file.split('.')[0])
            frame_path = os.path.join(frames_path, frame_file)
            frame = cv2.imread(frame_path)
            if frame is not None:
                frames_dict[frame_id] = frame

        pairs, labels = create_pairs_and_labels(gt_data, frames_dict, crop_size)
        pairs_list.extend(pairs)
        labels_list.extend(labels)

    X_train = np.array(pairs_list)
    y_train = np.array(labels_list)

    with open(output_filename, 'wb') as f:
        pickle.dump((X_train, y_train), f)

    print(f"Data saved to {output_filename}")

if __name__ == "__main__":
    data_path = '../MOT17/train'
    gt_filename = 'gt/gt.txt'
    output_filename = 'siamese_training_data.pkl'
    main(data_path, gt_filename, output_filename)
