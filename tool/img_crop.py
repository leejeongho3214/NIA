import errno
import json 
import cv2
import os

from tqdm import tqdm

def mkdir(path):
    # if it is the current folder, skip.
    # otherwise the original code will raise FileNotFoundError
    if path == "":
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


folder_path = f"dataset/label"   # Dataset dir path
for equ in os.listdir(folder_path):
    equ_path = os.path.join(folder_path, equ)
    for sub in tqdm(os.listdir(equ_path)):
        sub_path = os.path.join(folder_path, equ, sub)
        for anno_path in os.listdir(sub_path):
            anno_f_path = os.path.join(sub_path, anno_path)
            with open(anno_f_path, "r") as f:
                anno = json.load(f)
                img = cv2.imread(os.path.join("dataset/img", equ, sub, anno["info"]["filename"]))
                if anno["images"]["bbox"] == None: continue
                bbox = list(map(int, anno["images"]["bbox"]))
                
                center_bbox = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                center_bbox = list(map(int, center_bbox))
                mkdir(os.path.join("dataset/cropped_img", equ, sub))
                
                if anno["images"]["facepart"] == 0:
                    cropped_img = img
                else:
                    width, height = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    crop_length = int(max(width, height) / 2)
                    cropped_img = img[max(center_bbox[1] - crop_length, 0) : min(center_bbox[1] + crop_length, img.shape[0]), max(center_bbox[0] - crop_length, 0) : min(center_bbox[0] + crop_length, img.shape[1])]
                
                resized_img = cv2.resize(cropped_img, (256, 256))
                cv2.imwrite(os.path.join("dataset/cropped_img", equ, sub, anno["info"]["filename"][:-4] + f'_{str(anno["images"]["facepart"]).zfill(2)}' + '.jpg'), resized_img)
