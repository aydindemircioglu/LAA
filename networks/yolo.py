from ultralytics import YOLO

import cv2
from glob import glob
import os
import sys
import json
import typer
import shutil

from utils import *

exportPath = "./data"


# unclear if needed this way
def generateYML (fold, margin):
    # read mmdetect annotation
    mmdetectAnnPath = "./mmdetect/annotations/"
    annList = glob(f"{mmdetectAnnPath}/*.json")
    annFiles = [a for a in annList if f"{fold}_{margin}" in a]

    for af in annFiles:
        split = "val" if "_val_fold" in af else "train"
        with open(af, "r") as ff:
            anns = json.load(ff)
        basePath = f'{exportPath}/{fold}_{margin}/{split}/'
        os.makedirs(basePath, exist_ok=True)
        for a in anns["annotations"]:
            coords = a["bbox"]
            img_id = a["image_id"]
            img_meta = [k for k in anns["images"] if k["id"] == img_id]
            assert len(img_meta) == 1
            isize = [img_meta[0]["width"], img_meta[0]["height"]]
            # we never have anything else, so ensure
            assert isize[0] == isize[1]
            fname = img_meta[0]["file_name"]
            tx, ty = coords[0], coords[1]
            tw, th = coords[2], coords[3]
            bbox = [tx+tw/2, ty+th/2, tw, th]
            bbox = [k/isize[j%2] for j,k in enumerate(bbox)]
            text = '0 ' + ' '.join([f'{k}' for k in bbox])
            os.makedirs(basePath + "labels", exist_ok=True)
            with open(f"{basePath}/labels/{img_id.replace('.png', '')}.txt", 'w') as fh:
                fh.write(text)

            # copy topo
            os.makedirs(basePath + "images", exist_ok=True)
            shutil.copy(fname, basePath + "images/")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    y = f'path: {script_dir}' + '\n'
    y += f'train: {os.path.join(script_dir, exportPath, f"{fold}_{margin}", "train")}' + '\n'
    y += f'val: {os.path.join(script_dir, exportPath, f"{fold}_{margin}", "val")}' + '\n'
    y += f'test: {os.path.join(script_dir, exportPath, f"{fold}_{margin}", "val")}' + '\n'  # ignore test for now
    y += f'\n'
    y += f'names:' + '\n'
    y += f'  0: heart'

    with open(f'./{exportPath}/{fold}_{margin}/yolo_data.yaml', 'w') as fh:
        fh.write(y)


def main(margin: int = None, fold: int = None, modelname = None):
    os.makedirs(f"./{exportPath}", exist_ok = True)
    model = YOLO(f"yolo11{modelname}.pt")  # load a pretrained model (recommended for training)
    # create yaml temporarily
    generateYML(fold, margin)
    try:
        shutil.rmtree(f"./runs/obb/train_{fold}_{margin}_{modelname}")
    except:
        pass
    batch = 16
    if modelname == "l" or modelname == "x":
        batch = 8
    results = model.train(data=f"./data/{fold}_{margin}/yolo_data.yaml", device=0, name=f'train_{fold}_{margin}_{modelname}', epochs=100, imgsz=1024, batch = batch)


if __name__ == "__main__":
    typer.run(main)

#
