
import os
import shutil
import pickle
import torch
from glob import glob
from pathlib import Path
from PIL import Image
from ultralytics import YOLO


def validate_and_save(model, img_paths, save_path):
    results_dict = []

    # Iterate over each image
    for img_path in img_paths:
        img = Image.open(img_path)
        results = model(img_path, conf = 0.05)
        try:
            bbox = results[0].boxes.xyxy[0].unsqueeze(0).cpu()
            labels = list(results[0].names.keys())
            score = results[0].boxes.conf.clone().detach()[0].cpu()
        except Exception as e:
            bbox = torch.empty((0, 4))
            labels = torch.empty((0,), dtype=torch.int64)
            score = torch.empty((0, 1))

        # first path for model selection
        img_path = f'/data/data/vorhofohr/slices/{os.path.basename(img_path)}'

        result = {
            'img_shape': img.size,
            'batch_input_shape': img.size,
            'ori_shape': img.size,  # Assuming original shape is same for this example
            'img_path': img_path,
            'scale_factor': (1, 1),  # Assuming no scaling needed
            'img_id': Path(img_path).stem,
            'pad_shape': img.size,
            'pred_instances': {
                'bboxes': bbox,
                'labels': results[0].names,
                'scores': results[0].probs,  # are empty
            }
        }
        results_dict.append(result)

    with open(save_path, 'wb') as f:
        pickle.dump(results_dict, f)


if __name__ == '__main__':
    # instead of glob do it 'systematically'
   for tp in ["n", "s", "m", "l", "x"]:
    # for tp in ["n", "s"]:
        for m in range(4,21,2):
            fw = glob (f"./runs/detect/train_0_{m}_{tp}/weights/best.pt")[0]
            model = YOLO(fw)
            img_paths = glob(f"./data/0_{m}/val/images/*.png")
            save_path = f'../networks.new/mmdetect/results/yolo_yolo{tp}_{m}_0.pkl'
            validate_and_save(model, img_paths, save_path)

#
