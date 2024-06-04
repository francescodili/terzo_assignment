import pandas as pd
import compare_crops
import detect_boxes
import os
from PIL import Image
import  torch
import Tracker
from TrackEval.trackeval import Evaluator, datasets, metrics


def read_output_file(output_file):
    return pd.read_csv(output_file, sep=",", header=0, 
                       names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])

def read_gt_file(gt_file):
    return pd.read_csv(gt_file, sep=",", header=None, 
                       names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "score", "class", "visibility"])


def generate_output_files(data_path, output_path, tracker, model, similarity_threshold):
    output_files = []
    for dir in os.listdir(data_path):
        for video_dir in os.listdir(os.path.join(data_path, dir)):
            if video_dir == 'img1':
                video_output_path = os.path.join(output_path, dir, f'output_{similarity_threshold}_sim.txt')
                output_files.append(video_output_path)
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

                    save_boxes(active_tracks, frame_number, video_output_path, prob.tolist())
    return output_files

def take_gt_files(data_path): 
    gt_files = []
    for dir in os.listdir(data_path):
        gt_dir = os.path.join(data_path, dir, 'gt')
        if os.path.isdir(gt_dir):
            gt_file_path = os.path.join(gt_dir, 'gt.txt')
            if os.path.exists(gt_file_path):
                gt_files.append(gt_file_path)
    return gt_files


def save_boxes(tracks, frame, save_path, confs):
    with open(save_path, 'a') as file:
        for track, conf in zip(tracks, confs):
            bbox = track.bbox
            bb_left = bbox[0]
            bb_top = bbox[1]
            bb_width = bbox[2] - bbox[0]
            bb_height = bbox[3] - bbox[1]
            max_conf = max(conf)
            line = f"{frame}, {track.track_id}, {bb_left}, {bb_top}, {bb_width}, {bb_height}, {max_conf}, -1, -1, -1\n"
            file.write(line)

def run_trackeval(gt_folder, results_folder, eval_path):
    """Run TrackEval on the generated output files."""
    # Configuration options for TrackEval
    eval_config = {
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'PRINT_RESULTS': True,
        'PRINT_CONFIG': True,
        'TIME_PROGRESS': True,
        'DISPLAY_LESS_PROGRESS': False,
        'OUTPUT_SUMMARY': True,
        'OUTPUT_EMPTY_CLASSES': True,
        'OUTPUT_DETAILED': True,
        'PLOT_CURVES': False
    }

    dataset_config = {
        'GT_FOLDER': gt_folder,  # Location of GT data
        'TRACKERS_FOLDER': results_folder,  # Trackers location
        'OUTPUT_FOLDER': eval_path,  # Where to save eval results
        'TRACKERS_TO_EVAL': ['default_tracker'],  # List of trackers to evaluate
        'CLASSES_TO_EVAL': ['pedestrian'],  # List of classes to evaluate
        'BENCHMARK': 'MOT17',  # Benchmark to evaluate
    }

    evaluator = Evaluator(eval_config)
    dataset_list = [datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [metrics.HOTA(), metrics.CLEAR()]
    evaluator.evaluate(dataset_list, metrics_list)

def main():
    data_path = '../MOT17/train'
    output_path = '../bbox/train_bbox'
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    #Carico in una lista i percorsi dei file gt
    gt_files = take_gt_files(data_path)

    similarity_thresholds = [0.3, 0.5, 0.7, 0.9]

    for i, similarity_threshold in enumerate(similarity_thresholds):
        tracker = Tracker.Tracker(similarity_threshold)

        output_files = generate_output_files(data_path, output_path, tracker, model, similarity_threshold)

        #TODO
        #Accoppiare output_file e gt_file e passarli in un for a run_trackeval

    # Esegue TrackEval
    run_trackeval('../MOT17/train', '../bbox/train_bbox')


if __name__ == "__main__":
    main()
