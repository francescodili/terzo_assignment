import os
import compare_crops
import detect_boxes
from PIL import Image
import torch
import Tracker
import validazione
from TrackEval.trackeval import Evaluator, datasets, metrics
import pandas as pd
import detect_boxes
import cv2

def generate_output_test_files(data_path, output_path, similarity_threshold, model, test_dir = 'MOT17-test'):
    output_files = {}
    video_path = os.path.join(data_path, test_dir)
    for dir in os.listdir(video_path):
        for video_dir in os.listdir(os.path.join(video_path, dir)):
            if video_dir == 'img1':
                video_output_path = os.path.join(output_path,  test_dir, 'default_tracker', 'data', f'{dir}.txt')
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

def run_trackeval_test(gt_folder, trackers_folder, eval_path):
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
        'SEQMAP_FILE': os.path.abspath('../MOT17/seqmaps/MOT17-test.txt'),
        'SPLIT_TO_EVAL': 'test',  # Valid: 'train', 'test', 'all'
        'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt'
    }


    evaluator = Evaluator(eval_config)
    dataset_list = [datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [metrics.HOTA(), metrics.CLEAR()]
    evaluator.evaluate(dataset_list, metrics_list)



def valutation_test_videos(gt_folder, trackers_folder):    
    
    print(f'Valutazione con soglia 0.5: {trackers_folder}')
    eval_path = f'results/eval_sim_test.txt'
    
    run_trackeval_test(gt_folder, trackers_folder, eval_path)


def save_results(file, output_file):
    # Leggi il file di input
    df = pd.read_csv(file, sep='\s+')

    # Elenco delle metriche di interesse (HOTA e CLEAR)
    metrics_of_interest = [
        'HOTA', 'CLR_Re', 'CLR_Pr', 'MOTA', 'MOTP', 'MT', 'ML',
        'CLR_FP', 'CLR_FN', 'IDSW'
    ]

    # Seleziona le colonne di interesse se esistono nel dataframe
    df_metrics = df[metrics_of_interest]

    # Imposta il nome della riga
    df_metrics.index = ['test_results']

    # Salva i risultati in un nuovo file
    df_metrics.to_csv(output_file, sep='\t')

    print(f'I risultati sono stati salvati in {output_file}')


def select_best_worst_video(file_path, output_file):
    # Carica il file CSV
    df = pd.read_csv(file_path)
    
    # Elenco delle metriche di interesse
    metrics = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)', 'MOTA', 'MOTP', 'CLR_Re', 'CLR_Pr', 'sMOTA']
    
    # Seleziona solo le colonne di interesse
    selected_columns = ['seq'] + metrics
    df_selected = df[selected_columns]
    
    # Normalizza le metriche
    df_normalized = df_selected.copy()
    for metric in metrics:
        df_normalized[metric] = (df_selected[metric] - df_selected[metric].min()) / (df_selected[metric].max() - df_selected[metric].min())
    
    # Calcola il punteggio complessivo
    df_normalized['total_score'] = df_normalized[metrics].sum(axis=1)
    
    # Ordina i video in base al punteggio complessivo
    df_sorted = df_normalized.sort_values(by='total_score', ascending=False)
    
    # Seleziona il miglior e il peggior video
    best_video = df_sorted.iloc[0]
    worst_video = df_sorted.iloc[-1]
    
    # Mostra i risultati
    print("Il miglior video è:")
    print(best_video[['seq', 'total_score']])
    print("\nIl peggior video è:")
    print(worst_video[['seq', 'total_score']])
    
    # Salva i risultati in un file
    with open(output_file, 'a') as f:
        f.write("\n\nIl miglior video e':\n")
        f.write(best_video[['seq', 'total_score']].to_string(header=False, index=False))
        f.write("\n\nIl peggior video e':\n")
        f.write(worst_video[['seq', 'total_score']].to_string(header=False, index=False))

    print(f'I risultati sono stati salvati in {output_file}')

def generate_frames(videos_path, original_videos):
    for video in os.listdir(videos_path):
        video_path = os.path.join(videos_path, video)
        with open(video_path, 'r') as f:
            lines = f.readlines()

        # Creare una mappa di colori unica per ogni ID
        ids = list(set([int(line.split(',')[1]) for line in lines]))
        id_color_map = {id_: color for id_, color in zip(ids, detect_boxes.generate_unique_colors(len(ids)))}

        # Directory per salvare i frame del video
        output_dir = f'../videos/{video.split(".")[0]}'
        os.makedirs(output_dir, exist_ok=True)

        frames = {}
        for line in lines:
            frame_number, id_person, bb_left, bb_top, bb_width, bb_height, conf, x, y, z = map(float, line.split(','))
            frame_number = int(frame_number)
            id_person = int(id_person)

            if frame_number not in frames:
                frames[frame_number] = []
            frames[frame_number].append((id_person, bb_left, bb_top, bb_width, bb_height))

        video_name = video.split(".")[0]
        video_dir = os.path.join(original_videos, video_name, 'img1')
        for frame_number, boxes in frames.items():
            img_path = os.path.join(video_dir, f'{frame_number:06d}.jpg')
            if not os.path.exists(img_path):
                continue
            
            img = Image.open(img_path)
            boxes_coords = [(bb_left, bb_top, bb_left + bb_width, bb_top + bb_height) for _, bb_left, bb_top, bb_width, bb_height in boxes]
            ids = [id_person for id_person, _, _, _, _ in boxes]

            img_with_boxes = detect_boxes.draw_boxes(img, boxes_coords, ids, id_color_map)

            output_img_path = os.path.join(output_dir, f'{frame_number:06d}.jpg')
            cv2.imwrite(output_img_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))

def create_videos_from_frames(output_dir, fps=30):
    for folder_name in os.listdir(output_dir):
        folder_path = os.path.join(output_dir, folder_name)
        if os.path.isdir(folder_path):
            frame_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            frame_files.sort()  # Assicurati che i frame siano ordinati

            # Verifica che ci siano frame nella cartella
            if len(frame_files) == 0:
                continue

            # Ottieni le dimensioni dei frame
            first_frame_path = os.path.join(folder_path, frame_files[0])
            first_frame = cv2.imread(first_frame_path)
            height, width, layers = first_frame.shape

            # Definisci il percorso di output per il video
            video_output_path = os.path.join(output_dir, f'{folder_name}.mp4')

            # Inizializza il video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

            for frame_file in frame_files:
                frame_path = os.path.join(folder_path, frame_file)
                frame = cv2.imread(frame_path)
                video_writer.write(frame)

            video_writer.release()
            print(f'Video salvato: {video_output_path}')

if __name__ == "__main__":
    data_path = '../MOT17'
    output_path = '../bbox/test_bbox'
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    similarity_threshold = 0.7

    #generate_output_test_files(data_path, output_path, similarity_threshold, model)

    gt_folder = validazione.normalize_path(data_path)
    output_path = validazione.normalize_path(output_path)
    #valutation_test_videos(gt_folder, output_path)

    file_results = 'results/eval_sim_test.txt/default_tracker/pedestrian_summary.txt'
    file_best_worst = 'results/eval_sim_test.txt/default_tracker/pedestrian_detailed.csv'
    output_file = 'test_metrics_summary.txt'

    save_results(file_results, output_file)

    select_best_worst_video(file_best_worst, output_file) 

    original_videos = os.path.join(data_path, 'MOT17-test')
    #generate_frames(os.path.join(output_path, 'MOT17-test', 'default_tracker', 'data'), original_videos)


    output_dir = '../videos'
    #create_videos_from_frames(output_dir)