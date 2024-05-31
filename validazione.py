import pandas as pd
import json
import os
import subprocess

def read_output_file(output_file):
    return pd.read_csv(output_file, sep=",", header=0, 
                       names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])

def read_gt_file(gt_file):
    return pd.read_csv(gt_file, sep=",", header=None, 
                       names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "score", "class", "visibility"])

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxB[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2] + 1) * (boxB[3] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def convert_to_trackeval_format(df, output_path):
    data = {}
    for _, row in df.iterrows():
        frame = int(row['frame'])
        track_id = int(row['id'])
        bbox = [row['bb_left'], row['bb_top'], row['bb_width'], row['bb_height']]
        score = row['conf'] if 'conf' in row else 1.0

        if frame not in data:
            data[frame] = []

        data[frame].append({
            'track_id': track_id,
            'bbox': bbox,
            'score': score
        })

    with open(output_path, 'w') as f:
        json.dump(data, f)



def run_validation(output_files, gt_files, output_json_path, gt_json_path):
    for output_file, gt_file in zip(output_files, gt_files):
        output_df = read_output_file(output_file)
        gt_df = read_gt_file(gt_file)

        output_json_file = output_json_path.replace(".json", f"_{os.path.basename(output_file)}.json")
        gt_json_file = gt_json_path.replace(".json", f"_{os.path.basename(gt_file)}.json")

        convert_to_trackeval_format(output_df, output_json_file)
        convert_to_trackeval_format(gt_df, gt_json_file)
        print(f"Files converted to TrackEval format and saved to {output_json_file} and {gt_json_file}")


def run_trackeval(config_file, output_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    command = [
        "python", "TrackEval/scripts/run_mot_challenge.py",
        f"--GT_FOLDER={config['GT_FOLDER']}",
        f"--TRACKERS_FOLDER={config['TRACKERS_FOLDER']}",
        f"--SPLIT_TO_EVAL={config['SPLIT_TO_EVAL']}",
        f"--METRICS", *config['METRICS'],
        f"--DO_PREPROC={config['DO_PREPROC']}",
        f"--TRACKERS_TO_EVAL", *config['TRACKERS_TO_EVAL'],
        f"--GT_LOC_FORMAT={config['GT_LOC_FORMAT']}",
        f"--THRESHOLD={config['THRESHOLD']}",
        f"--PRINT_CONFIG={config['PRINT_CONFIG']}",
        f"--SEQMAP_FILE={config['SEQMAP_FILE']}"
    ]

    with open(output_file, 'w') as f:
        try:
            subprocess.run(command, shell=False, check=True, stdout=f, stderr=subprocess.STDOUT)
            print(f"Output written to {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e}")
            with open(output_file, 'r') as f:
                print(f.read())

def create_config_file(base_config, similarity_threshold, config_file_path):
    config = base_config.copy()
    config["THRESHOLD"] = similarity_threshold
    with open(config_file_path, 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":
    data_path = '../MOT17/train'
    annotated_path = '../bbox/train_bbox'

    gt_files = []
    output_files = []

    for dir in os.listdir(data_path):
        gt_dir = os.path.join(data_path, dir, 'gt')
        if os.path.isdir(gt_dir):
            gt_file_path = os.path.join(gt_dir, 'gt.txt')
            if os.path.exists(gt_file_path):
                gt_files.append(gt_file_path)

    for dir in os.listdir(annotated_path):
        output_dir = os.path.join(annotated_path, dir)
        if os.path.isdir(output_dir):
            output_file_path = os.path.join(output_dir, 'output.txt')
            if os.path.exists(output_file_path):
                output_files.append(output_file_path)

    output_json_path = '../MOT17/train/trackeval_output.json'
    gt_json_path = '../MOT17/train/trackeval_gt.json'
    base_config_file_path = 'trackeval_config.json'

    base_config = {
        "GT_FOLDER": "../MOT17/train",
        "TRACKERS_FOLDER": "../MOT17/train",
        "SPLIT_TO_EVAL": "val",
        "METRICS": ["HOTA", "CLEAR"],
        "DO_PREPROC": False,
        "TRACKERS_TO_EVAL": ["my_tracker"],
        "GT_LOC_FORMAT": "{gt_folder}/{seq}.json",
        "SEQMAP_FILE": "../MOT17/train/seqmaps/MOT17-val.txt",  # Cambiato per essere una stringa
        "PRINT_CONFIG": True
    }

    similarity_thresholds = [0.3, 0.5, 0.7, 0.9]

    run_validation(output_files, gt_files, output_json_path, gt_json_path)

    for i, similarity_threshold in enumerate(similarity_thresholds):
        config_file_path = f"trackeval_config_{i}.json"
        output_result_file = f"trackeval_results_{i}.txt"

        create_config_file(base_config, similarity_threshold, config_file_path)
        run_trackeval(config_file_path, output_result_file)
