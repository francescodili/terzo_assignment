import os
import compare_crops
import detect_boxes
from PIL import Image
import torch
import Tracker
import validazione
from TrackEval.trackeval import Evaluator, datasets, metrics

def generate_output_test_files(data_path, output_path, similarity_threshold, model, test_dir = 'MOT17-test'):
    output_files = {}
    video_path = os.path.join(data_path, test_dir)
    for dir in os.listdir(video_path):
        for video_dir in os.listdir(os.path.join(video_path, dir)):
            if video_dir == 'img1':
                video_output_path = os.path.join(output_path,  test_dir, f'{dir}.txt')
                print(video_output_path)
                if dir not in output_files:
                    output_files[dir] = []

                if os.path.exists(video_output_path):
                    print(f'{video_output_path} già esiste')
                    continue

                tracker = Tracker.Tracker(similarity_threshold)

                for frame_path in os.listdir(os.path.join(video_path, dir, video_dir)):
                    image_path = os.path.join(video_path, dir, video_dir, frame_path)
                    img = Image.open(image_path)
                    frame_number = int(frame_path.split('.')[0])
                    prob, bboxes_scaled = detect_boxes.detect(model, img)

                    crops = compare_crops.extract_crops(img, bboxes_scaled)
                    tracker.update_tracks(bboxes_scaled.tolist(), crops)
                    active_tracks = tracker.get_active_tracks()

                    validazione.save_boxes(active_tracks, frame_number, video_output_path, prob.tolist())
                    output_files[dir].append(video_output_path)

    if not output_files:
        return None
    return output_files

if __name__ == "__main__":
    data_path = '../MOT17'
    output_path = '../bbox/test_bbox'
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    similarity_threshold = 0.5

    generate_output_test_files(data_path, output_path, similarity_threshold, model)