import  torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def extract_crops(pil_img, boxes, crops, det_path=None):
    # Convert PIL image to OpenCV format
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if not crops: #Inizializzazione
        id = 0
        for (xmin, ymin, xmax, ymax) in boxes.tolist():
            center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            size = (xmax - xmin, ymax - ymin)
            crop = cv2.getRectSubPix(img, patchSize=(int(size[0]), int(size[1])), center=(center[0], center[1]))
            crops[id] = crop
            id += 1
    else: 
        for (xmin, ymin, xmax, ymax) in boxes.tolist():
            center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            size = (xmax - xmin, ymax - ymin)
            crop = cv2.getRectSubPix(img, patchSize=(int(size[0]), int(size[1])), center=(center[0], center[1]))
            id = compare_crops(crop, crops, boxes, det_path)
            crops[id] = crop

    return crops


def similarity_between_crops(crop1, crop2):
    # Converting images to grayscale
    crop1_gray = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
    crop2_gray = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM between two crops
    score, _ = ssim(crop1_gray, crop2_gray, full=True)
    return score

#def compare_crops(crop, crops, boxes, det_path):
#    similarity = {}
#    for id, img in crops.values():
#        similarity[id] = similarity_between_crops(crop, img) #valore tra 0 e 1 (0 totalmente diverse, 1 uguali)
#
#    for id, sim in similarity.values():
#        if sim > 0.7:


def compare_crops(crop, crops):
    similarity = {}
    for id, img in crops.items():
        similarity[id] = similarity_between_crops(crop, img)  # valore tra 0 e 1 (0 totalmente diverse, 1 uguali)

    max_sim = 0
    max_id = -1
    for id, sim in similarity.items():
        if sim > max_sim:
            max_sim = sim
            max_id = id
    
    if max_sim > 0.7:
        return max_id
    else:
        return max(crops.keys()) + 1

def save_boxes(boxes, idxs, save_path):
    with open(save_path, 'w') as file:
        count = 0
        for box in boxes.tolist():
            file.flush()
            os.fsync(file.fileno())
            
            line = f'{idxs[count]}, {box[0]}, {box[1]}, {box[2]}, {box[3]}\n'
            file.write(line)
            count += 1
    if not file.closed:
            file.close()


    

def detect(model, im, transform = None, threshold_confidence = 0.7):
    if transform is None:
        # standard PyTorch mean-std input image normalization
        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with a confidence > threshold_confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold_confidence

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # filter to keep only the person class (class index 1)
    person_class_index = CLASSES.index('person')
    person_keep = probas[keep].argmax(-1) == person_class_index

    return probas[keep][person_keep], bboxes_scaled[person_keep]


def main():
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    data_path = '../MOT17/train'
    output_path = '../bbox/train_bbox'
    crops = {}
    for dir in os.listdir(data_path):
        for video_dir in os.listdir(os.path.join(data_path, dir)):
            if video_dir == 'img1':
                for frame_path in os.listdir(os.path.join(data_path, dir, video_dir)):
                    image_path = os.path.join(data_path, dir, video_dir, frame_path)
                    img = Image.open(image_path)
                    prob, bboxes_scaled = detect(model, img)
                    
                    # Crea la directory di output se non esiste
                    output_dir = os.path.join(output_path, dir)
                    os.makedirs(output_dir, exist_ok=True)
                    
                    save_path = os.path.join(output_dir, frame_path)
                    det_path = os.path.join(output_dir, f'{frame_path}_det.txt')
                    crops = extract_crops(img, bboxes_scaled, crops, det_path=det_path)
                    save_boxes(bboxes_scaled, det_path)
                        

                

if __name__=='__main__':
    main()