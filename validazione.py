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


def generate_output_files(data_path, output_path, similarity_threshold, model, i):
    output_files = {}  # Crea un dizionario vuoto
    threshold_dir = f'sim_0{int(similarity_threshold * 10):01d}'
    for dir in os.listdir(data_path):
        for video_dir in os.listdir(os.path.join(data_path, dir)):
            if video_dir == 'img1':
                video_output_path = os.path.join(output_path, threshold_dir, dir, f'output_{i}_sim.txt')
                print(video_output_path)
                if dir not in output_files:
                    output_files[dir] = []  # Crea una lista vuota per la chiave 'dir'
                os.makedirs(os.path.join(output_path, threshold_dir, dir), exist_ok=True)

                # Se il file esiste già, passa all'iterazione successiva
                if os.path.exists(video_output_path):
                    print(f'{video_output_path} già esiste')
                    continue

                tracker = Tracker.Tracker(similarity_threshold)

                for frame_path in os.listdir(os.path.join(data_path, dir, video_dir)):
                    image_path = os.path.join(data_path, dir, video_dir, frame_path)
                    img = Image.open(image_path)
                    frame_number = int(frame_path.split('.')[0])
                    prob, bboxes_scaled = detect_boxes.detect(model, img)

                    crops = compare_crops.extract_crops(img, bboxes_scaled)
                    tracker.update_tracks(bboxes_scaled.tolist(), crops)
                    active_tracks = tracker.get_active_tracks()

                    save_boxes(active_tracks, frame_number, video_output_path, prob.tolist())
                    output_files[dir].append(video_output_path)  # Aggiungi 'video_output_path' alla lista associata alla chiave 'dir'

    if not output_files:  # Controlla se il dizionario è vuoto
        return None
    return output_files


def take_gt_paths(data_path): 
    gt_folders = {}
    for dir in os.listdir(data_path):
        dir_path = os.path.join(data_path, dir)
        gt_dir = os.path.join(dir_path, 'gt')
        if os.path.isdir(gt_dir):
            gt_folders[dir] = dir_path
    return gt_folders


'''
def take_output_files(data_path, output_path, i):
    output_files = {}  # Crea un dizionario vuoto
    for dir in os.listdir(data_path):
        for video_dir in os.listdir(os.path.join(data_path, dir)):
            if video_dir == 'img1':
                video_output_path = os.path.join(output_path, dir, f'output_{i}_sim.txt')
                if dir not in output_files:
                    output_files[dir] = []  # Crea una lista vuota per la chiave 'dir'
                if os.path.exists(video_output_path):  # Verifica se il file esiste
                    output_files[dir].append(video_output_path)  # Aggiungi 'video_output_path' alla lista associata alla chiave 'dir'

    return output_files'''

def take_output_files(data_path, output_path, similarity_thresholds):
    output_files = {}  # Crea un dizionario vuoto
    for i, similarity_threshold in enumerate(similarity_thresholds):
        threshold_dir = f'sim_0{int(similarity_threshold * 10):01d}'
        for dir in os.listdir(data_path):
            video_output_path = os.path.join(output_path, threshold_dir, dir, f'output_{i}_sim.txt')
            if dir not in output_files:
                output_files[dir] = []  # Crea una lista vuota per la chiave 'dir'
            if os.path.exists(video_output_path):  # Verifica se il file esiste
                output_files[dir].append(video_output_path)  # Aggiungi 'video_output_path' alla lista associata alla chiave 'dir'
    return output_files





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
        'TRACKERS_FOLDER': os.path.dirname(results_folder),  # Trackers location
        'OUTPUT_FOLDER': os.path.dirname(eval_path),  # Where to save eval results
        'TRACKERS_TO_EVAL': ['default_tracker'],  # List of trackers to evaluate
        'CLASSES_TO_EVAL': ['pedestrian'],  # List of classes to evaluate
        'BENCHMARK': 'MOT17',  # Benchmark to evaluate
        'SEQMAP_FILE': '../bbox/seqmaps/MOT17-train.txt',  # Path to seqmap file
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt'
    }

    evaluator = Evaluator(eval_config)
    dataset_list = [datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [metrics.HOTA(), metrics.CLEAR()]
    evaluator.evaluate(dataset_list, metrics_list)


def main(data_path, output_path, similarity_thresholds):
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)

    for i, similarity_threshold in enumerate(similarity_thresholds):
        output_files = generate_output_files(data_path, output_path, similarity_threshold, model, i)
    
    return output_files

def do_valutation(data_path, output_path, similarity_thresholds, output_files=None):
    # Carico in una lista i percorsi delle cartelle gt
    gt_folders = take_gt_paths(data_path)
    
    if output_files is None:
        output_files = {}
        output_files.update(take_output_files(data_path, output_path, similarity_thresholds))
    
    for i, threshold in enumerate(similarity_thresholds):
        for key, value in output_files.items():
            gt_path = gt_folders[key]
            detection_files = [v for v in value if f'output_{i}_sim.txt' in v]
            for detection_file in detection_files:
                print(f'Valutazione per {key} con soglia {threshold}: {detection_file}')
                
                # Esegue TrackEval
                run_trackeval(gt_path, detection_file, f'results/eval_{key}_sim_{int(threshold * 10):01d}.txt')


if __name__ == "__main__":
    data_path = '../MOT17/train'
    output_path = '../bbox/train_bbox'
    similarity_thresholds = [0.3, 0.5, 0.7, 0.9, 0.4]

    flag_gen_files = False

    if flag_gen_files:
        output_files = main(data_path, output_path, similarity_thresholds)
    else:
        output_files = None

    do_valutation(data_path, output_path, similarity_thresholds, output_files)
