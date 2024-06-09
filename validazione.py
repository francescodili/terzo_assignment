import os
import pandas as pd
import compare_crops
import detect_boxes
from PIL import Image
import torch
import Tracker
from TrackEval.trackeval import Evaluator, datasets, metrics


def generate_output_files(data_path, output_path, similarity_threshold, model, train_dir = 'MOT17-train'):
    output_files = {}
    threshold_dir = f'sim_0{int(similarity_threshold * 10):01d}'
    video_path = os.path.join(data_path, train_dir)
    for dir in os.listdir(video_path):
        for video_dir in os.listdir(os.path.join(video_path, dir)):
            if video_dir == 'img1':
                video_output_path = os.path.join(output_path, threshold_dir, train_dir, 'default_tracker', 'data' ,f'{dir}.txt')
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

                    save_boxes(active_tracks, frame_number, video_output_path, prob.tolist())
                    output_files[dir].append(video_output_path)

    if not output_files:
        return None
    return output_files


def take_output_files(output_path, similarity_thresholds):
    output_folders = []
    for similarity_threshold in similarity_thresholds:
        threshold_dir = f'sim_0{int(similarity_threshold * 10):01d}'
        threshold_folder = os.path.join(output_path, threshold_dir)
        output_folders.append(threshold_folder)
    return output_folders

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


def run_trackeval(gt_folder, trackers_folder, eval_path):
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
        'GT_FOLDER': gt_folder,
        'TRACKERS_FOLDER': os.path.abspath(trackers_folder),
        'OUTPUT_FOLDER': os.path.abspath(eval_path),
        'TRACKERS_TO_EVAL': ['default_tracker'],
        'CLASSES_TO_EVAL': ['pedestrian'],
        'BENCHMARK': 'MOT17',
        'SEQMAP_FILE': os.path.abspath('../MOT17/seqmaps/MOT17-train.txt'),
        'SPLIT_TO_EVAL': 'train',  # Valid: 'train', 'test', 'all'
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt'
    }

    print(f"GT_FOLDER: {dataset_config['GT_FOLDER']}")
    print(f"TRACKERS_FOLDER: {dataset_config['TRACKERS_FOLDER']}")
    print(f"SEQMAP_FILE: {dataset_config['SEQMAP_FILE']}")
    print(f"GT_LOC_FORMAT: {dataset_config['GT_LOC_FORMAT']}")

    evaluator = Evaluator(eval_config)
    dataset_list = [datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [metrics.HOTA(), metrics.CLEAR()]
    evaluator.evaluate(dataset_list, metrics_list)



def gen_files(data_path, output_path, similarity_thresholds):
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)

    for similarity_threshold in similarity_thresholds:
        generate_output_files(data_path, output_path, similarity_threshold, model)
    

def do_valutation(gt_folder, output_path, similarity_thresholds):    
    output_folders = take_output_files(output_path, similarity_thresholds)
    
    for i, trackers_folder in enumerate(output_folders):
        threshold = similarity_thresholds[i]
        print(f'Valutazione {i} con soglia {threshold}: {trackers_folder}')
        eval_path = f'results/eval_sim_{int(threshold * 10):01d}.txt'
        
        run_trackeval(gt_folder, trackers_folder, eval_path)

def select_best_tracker():

    file_names = [
        'results/eval_sim_3.txt/default_tracker/pedestrian_summary.txt',
        'results/eval_sim_4.txt/default_tracker/pedestrian_summary.txt',
        'results/eval_sim_5.txt/default_tracker/pedestrian_summary.txt',
        'results/eval_sim_7.txt/default_tracker/pedestrian_summary.txt',
        'results/eval_sim_9.txt/default_tracker/pedestrian_summary.txt'
    ]

    names = [
        'sim_03',
        'sim_04',
        'sim_05',
        'sim_07',
        'sim_09'
    ]

    # Lista per memorizzare i DataFrame
    dfs = []

    # Carica i dati dai file
    for file in file_names:
        df = pd.read_csv(file, sep='\s+')
        dfs.append(df)

    # Funzione per calcolare le metriche medie di interesse
    def calculate_metrics(df):
        required_columns = [
            'HOTA', 'CLR_Re', 'CLR_Pr', 'MOTA', 'MOTP', 'MT', 'ML',
            'CLR_FP', 'CLR_FN', 'IDSW'
        ]
        for col in required_columns:
            if col not in df.columns:
                print(f"La colonna '{col}' non è presente nei dati.")
        
        metrics = {}
        for col in required_columns:
            metrics[col] = df[col].mean()
        
        return metrics

    # Calcola le metriche per ciascun tracker
    metrics = []
    for df in dfs:
        metrics.append(calculate_metrics(df))

    # Crea un DataFrame per confrontare le metriche
    comparison_df = pd.DataFrame(metrics, index=names)

    # Mostra il DataFrame
    print("Confronto delle metriche:")
    print(comparison_df)

    # Normalizzazione delle metriche
    metrics_higher_better = ['HOTA', 'CLR_Re', 'CLR_Pr', 'MOTA', 'MOTP', 'MT']
    metrics_lower_better = ['ML', 'CLR_FP', 'CLR_FN', 'IDSW']

    normalized_df = pd.DataFrame(index=comparison_df.index)

    for col in metrics_higher_better:
        normalized_df[col] = (comparison_df[col] - comparison_df[col].min()) / (comparison_df[col].max() - comparison_df[col].min())

    for col in metrics_lower_better:
        normalized_df[col] = (comparison_df[col].max() - comparison_df[col]) / (comparison_df[col].max() - comparison_df[col].min())

    # Calcolo del punteggio complessivo
    normalized_df['total_score'] = normalized_df.sum(axis=1)

    # Mostra il DataFrame normalizzato
    print("Confronto delle metriche normalizzate:")
    print(normalized_df)

    # Seleziona il tracker con il punteggio complessivo più alto
    best_tracker = normalized_df['total_score'].idxmax()
    print("Il miglior tracker considerando tutte le metriche è:", best_tracker)

    # Salva i risultati in un file
    with open('best_tracker.txt', 'w') as f:
        f.write("Confronto delle metriche:\n")
        f.write(comparison_df.to_string())
        f.write("\n\nConfronto delle metriche normalizzate:\n")
        f.write(normalized_df.to_string())
        f.write("\n\nIl miglior tracker considerando tutte le metriche e':\n")
        f.write(str(best_tracker))

def normalize_path(path):
    return os.path.normpath(os.path.abspath(path))

if __name__ == "__main__":
    data_path = '../MOT17'
    output_path = '../bbox/train_bbox'
    similarity_thresholds = [0.3, 0.4, 0.5, 0.7, 0.9]

    flag_gen_files = False

    if flag_gen_files:
        gen_files(data_path, output_path, similarity_thresholds)


    gt_folder = normalize_path(data_path)
    output_path = normalize_path(output_path)
    #do_valutation(gt_folder, output_path, similarity_thresholds)

    select_best_tracker()