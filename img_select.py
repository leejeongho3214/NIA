from collections import defaultdict
import json
import os

import natsort
from tqdm import tqdm 


img_path = 'dataset/img'
json_path = 'dataset/label'
sub_path_list = [
    item
    for item in natsort.natsorted(os.listdir(img_path))
    if not item.startswith(".")
]
for equ_name in sub_path_list:
    if equ_name.startswith(".") or int(equ_name) not in [1, 2, 3]:
        continue
    
    for sub_fold in tqdm(
        natsort.natsorted(os.listdir(os.path.join(img_path, equ_name))),
        desc="path loading..",
    ):
        if sub_fold.startswith(".") or not os.path.exists(
            os.path.join(json_path, equ_name, sub_fold)
        ):
            continue
        folder_path = os.path.join(img_path, equ_name, sub_fold)
        
        for img_name in natsort.natsorted(os.listdir(folder_path)):
            if not img_name.endswith((".png", ".jpg", ".jpeg")):
                continue
            if img_name.split(".")[0].split("_")[-1] != "F":
                continue
            pre_name = "/".join(folder_path.split("/")[2:])
            json_name = os.path.join("dataset/label", pre_name)
            
            bbox_t = defaultdict(int)
            ## Classifying meaningful images for training from various angles of images
            for j_name in os.listdir(json_name):
                if not j_name.startswith(img_name.split('.')[0] + "_"):
                    continue
                with open(os.path.join(json_name, j_name), "r") as f:
                    json_meta = json.load(f)
                    
                    ## Sorting frontal images and img without bbox values
                    if list(json_meta["annotations"].keys())[0] == "acne" or json_meta['images']['bbox'] == None:
                        continue
                    bbox_t[json_meta['images']['facepart']] = (int(json_meta['images']['bbox'][1]) + int(json_meta['images']['bbox'][3])) // 2
                    
            if 3 not in list(bbox_t.keys()):
                bbox_t[3] = bbox_t[4]
                
            elif 4 not in list(bbox_t.keys()):
                bbox_t[4] = bbox_t[3]
                    
            if 5 not in list(bbox_t.keys()):
                bbox_t[5] = bbox_t[6]
                
            elif 6 not in list(bbox_t.keys()):
                bbox_t[6] = bbox_t[5]

            if bbox_t[1] > bbox_t[2]:
                print(json_name)
                
            elif bbox_t[2] > bbox_t[3] or bbox_t[2] > bbox_t[4]:
                print(json_name)
                
            elif bbox_t[3] > bbox_t[5] or bbox_t[3] > bbox_t[6] or bbox_t[4] > bbox_t[5] or bbox_t[3] > bbox_t[6]:
                print(json_name)
                
            elif bbox_t[5] > bbox_t[7] or bbox_t[6] > bbox_t[7]:
                print(json_name)
                
            elif bbox_t[7] > bbox_t[8]:
                print(json_name)
                
                    
                    

