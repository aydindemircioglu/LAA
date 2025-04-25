#!/usr/bin/python3

# https://blog.roboflow.ai/how-to-convert-annotations-from-voc-xml-to-coco-json/
import pandas as pd
import os
import argparse
import cv2
from sklearn.model_selection import KFold
from mmengine.fileio import dump


def get_image_info (filename):
    img_id = os.path.basename(filename)
    img = cv2.imread(filename)
    width = img.shape[1]
    height = img.shape[0]

    # make a true number of it
    image_info = {
        'file_name': os.path.abspath(filename),
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def process(df, cohort, safety_margin = 0):
    print ("## ", cohort)
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    boxIDs = []

    for i, (idx, row) in enumerate(df.iterrows()):
        #,Accession Nr,Upper,Lower,Img,Topo,CT,Split
        img_info = get_image_info (row["Img"])
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)
        upper, lower  = row["Upper"], row["Lower"]
        # add safety margin
        upper = upper - safety_margin
        lower = lower + safety_margin
        cWidth = img_info["width"]
        cHeight = lower-upper

        cBox_x = 0
        cBox_y = upper
        ann = {
            'area': cWidth * cHeight,
            'iscrowd': 0,
            'bbox_mode': 1, #BoxMode.XYWH_ABS,
            'bbox': [cBox_x, cBox_y, cWidth, cHeight],
            'category_id': 1,
            'ignore': 0,
            'segmentation': []  # This script is not for segmentation
        }

        boxID = abs(hash("box" + os.path.basename(row["Img"]))) % (10 ** 16)
        boxIDs.append(boxID)
        ann.update({'image_id': img_id, 'id': boxID})
        output_json_dict['annotations'].append(ann)

    category_info = {'supercategory': 'none', 'id': 1, 'name': "Heart"}
    output_json_dict['categories'].append(category_info)
    output_jsonpath= f"./mmdetect/annotations/{cohort}_{safety_margin}.json"
    dump(output_json_dict, output_jsonpath)
    print ("#boxIDs:", len(boxIDs))
    print ("#unique boxIDs:", len(set(boxIDs)))



def processCSV (csvFile, cohort, safety_margin):
    print ("Reading CSV from", csvFile)
    df = pd.read_csv (csvFile)
    process(df, cohort, safety_margin)



def main():
    annDir = "./mmdetect/annotations"
    os.makedirs(annDir, exist_ok = True)

    # process all
    for cohort in ["UKE.train", "UKE.val", "ELI.test"]:
        print ("Processing cohort", cohort)

        for safety_margin in range(21):
            if  cohort == "UKE.train":
                # recreate fold by fold
                nCV = 5
                if os.path.exists("./data/train_fold_0.csv") == False:
                    # recreate
                    df = pd.read_csv("./data/train.csv")
                    df = df.drop_duplicates(subset='Accession Nr', keep="first")

                    kf = KFold(n_splits = nCV, shuffle = True, random_state = nCV)
                    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
                        cvtrainPats = df.iloc[train_idx]
                        cvtestPats = df.iloc[test_idx]
                        cvtrainPats.to_csv("./data/train_train_fold_" + str(fold)+ ".csv")
                        cvtestPats.to_csv("./data/train_val_fold_" + str(fold)+ ".csv")
                for fold in range(nCV):
                    processCSV("./data/train_train_fold_" + str(fold) + ".csv", "train_train_fold_" + str(fold), safety_margin)
                    processCSV("./data/train_val_fold_" + str(fold) + ".csv", "train_val_fold_" + str(fold), safety_margin)
                processCSV("./data/train.csv", "train", safety_margin)

            if  cohort == "UKE.val":
                processCSV("./data/val.csv", "val", safety_margin)

            if  cohort == "ELI.test":
                processCSV("./data/test.csv", "test", safety_margin)

if __name__ == '__main__':
    main()

#
