import pandas as pd
import numpy as np
import json
import os
import subprocess

def read_output_file(output_file):
    return pd.read_csv(output_file, sep=",", header=0, 
                       names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])

def read_gt_file(gt_file):
    return pd.read_csv(gt_file, sep=",", header=None, 
                       names=["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y"])

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

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

def run_validation(output_file, gt_file, output_json_path, gt_json_path):
    output_df = read_output_file(output_file)
    gt_df = read_gt_file(gt_file)

    convert_to_trackeval_format(output_df, output_json_path)
    convert_to_trackeval_format(gt_df, gt_json_path)
    print(f"Files converted to TrackEval format and saved to {output_json_path} and {gt_json_path}")

def run_trackeval(config_file):
    command = f"python scripts/run_trackeval.py --CONFIG_FILE {config_file}"
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    output_file_path = 'path/to/output.txt'
    gt_file_path = 'path/to/gt.txt'
    output_json_path = 'path/to/trackeval_output.json'
    gt_json_path = 'path/to/trackeval_gt.json'
    config_file_path = 'path/to/trackeval_config.json'

    run_validation(output_file_path, gt_file_path, output_json_path, gt_json_path)
    run_trackeval(config_file_path)
