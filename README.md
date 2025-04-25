# Detecting the Left Atrial Appendage in CT Localizers Using Deep Learning

This repository contains the code for the paper:
'Detecting the Left Atrial Appendage in CT Localizers Using Deep Learning',
published in Scientific Nature Reports, 2025.

**Note:** The code is primarily focused on the network implementation and
may require some cleanup for real use.

**Important:** Data paths throughout the code need to be updated. Many
currently point to a specific location like `/data/data/vorhofohr/` and must
be adjusted to your environment and data.


## Datasets

The study utilized three datasets: `UKE.train`, `UKE.val`, and `ELI.test`.
`UKE.val` served as the internal test set and was *not* used during training or model selection.

These datasets require manual preparation:

1.  **Identify and Convert CTs:** Identify the relevant CT scans and convert them
to either `.nii.gz` or `.nrrd` format.
2.  **Segment LAA:** Use TotalSegmentator to segment the Left Atrial Appendage (LAA).
Run the following command within the directory containing each scan:
    ```bash
    for x in */; do cd $x; echo $x; [ ! -f ./segmentation.nii ] && CUDA_VISIBLE_DEVICES=0 TotalSegmentator -i CTA.nii.gz -o segmentation -ta aortic_branches_test --ml --preview || echo "exists"; cd ..; done
    ```
3.  **Extract Coordinates:** Map the coordinates of the segmented LAA region back to
the localizer images using the `./extractAnnotationFromTotalSegmentator.py` script.

This process generates `.csv` files stored in `./network/data/`. Each `.csv` file
contains the following columns:


```X,Accession Nr,Upper,Lower,PatientID,Img,Topo,CT,Split```

e.g.,

```0,2091719,143,177,22707146,/data/LAA/slices/topo_2091719.png,/data/LAA/topos/2091719.dcm,/data/LAA/CTs/2091719/CTA.nii.gz,train```


The columns denote (see the script which creates them):
-   `Upper` and `Lower`: y-coordinate boundaries of the LAA.
-   `Img`: Path to the localizer image (converted to PNG using Pseudo-RGB).
-   `Topo`: Path to the original DICOM localizer file (`.dcm`).
-   `CT`: Path to the CT volume (`.nii.gz`).
-   `Split`: Indicates if the sample belongs to the `train` or other dataset split.

**Note:** During training, only the processed localizer PNG images (`Img`) are used,
not the original DICOM files (`Topo`, `CT`).


## Training Data Preparation

The `./networks/createDataset.py` script prepares the datasets for training.
A key step is adding safety margins of varying sizes around the LAA annotations.
Consequently, separate datasets are generated for training and validation for each
specified safety margin.

The script also partitions the data into five folds for cross-validation. However,
due to computational constraints in the original study, only the first fold was used,
effectively resulting in a single 80:20 train/validation split as described in the paper.

This script outputs the data splits in the MMDetection annotation format (`.json`).
After execution, you will find annotation files in `./networks/mmdetect/annotations/`,
following a pattern like `{dataset}_{split_type}_{fold}_{safety_margin}.json`. For instance, `train_val_fold_0_12.json` corresponds to the validation portion of the training
dataset (fold 0) with a 12mm safety margin.


## Network Training

The networks were trained using two different frameworks: MMDetection and YOLO.

To start the training process for *both* frameworks, execute the main training script:

```bash
./run_train.sh
```

This will train both networks.


## MMDetection

**Warning:** MMDetection is deprecated, and installation can be challenging
due to complex dependencies and version incompatibilities. Follow the official
MMDetection installation guides carefully.


The specific package versions used in this project are listed in `requirements_mmdetection.txt`.
You can attempt to install them using `pip install -r requirements_mmdetection.txt`
within a dedicated virtual environment, however, the chances that this will work are low.
Conda seems to be required due to Python3 version incompatibilities.

- The `./networks/mmdetect/` directory contains the core MMDetection code used for training and testing.
- The script `./networks/mmdetect/fixers.py` modifies the network output configuration,
as only one class (LAA) needs to be detected.
- The `train.py` and `predict.py` scripts in the `./networks` directory act as wrappers around
the main MMDetection functions.

**Important:** These wrapper scripts contain hardcoded paths (e.g., for saving checkpoints to /data/data/vorhofohr/experiments/checkpoints). You must modify these paths to match your system configuration.


### YOLO

The YOLO training process uses the same data splits as MMDetection
(defined by the .json annotation files) to ensure fair comparison. The annotation
format is converted from MMDetection's .json to YOLO's required format on-the-fly during training.

YOLO training is handled by the `./networks/yolo.py` script.
The Python dependencies required for YOLO are listed in the main `./requirements.txt` file.

**Note:** The requirements in `./requirements.txt` (for YOLO) are different from those in `requirements_mmdetection.txt`. The original experiments used separate virtual environments
for MMDetection and YOLO to avoid that YOLO messed up the MMDetection environemtn.
It is reasonable you do the same to avoid conflicts.

After YOLO training completes, the predictions must be exported into a format
consistent with MMDetection's output for standardized evaluation.
Run the `./networks/yolo_predict.py` script to perform this conversion after training finishes.



## Model selection

Once both MMDetection and YOLO models have been trained, the best performing model
needs to be selected based on validation metrics.

The model selection script requires additional information extracted
from the DICOM headers (not essential). Ensure you run `./networks/computeExtraInfos.py`
before proceeding, or better, modify to fit your data needs.

The selection process is implemented in the `./networks/modelSelection.py script`.
This script identifies the best model architecture and hyperparameters (including
safety margin) based on
performance on the validation set.

After identifying the best model and its hyperparameters, the selected model
configuration should be retrained on the entire training dataset
(including the data previously held out for validation). This retraining is initiated
using the `run_final.sh` script. You must manually edit `run_final.sh` and insert
the hyperparameters identified as optimal during the model selection phase.
This was done on purpose to avoid any kind of testing during model selection.


## Final evaluation

To generate the final performance metrics (as presented in the paper's tables)
on the test sets (UKE.val and ELI.test), run the `./finalEvaluation.py` script.

To assess clinical utility and provide ground truth for comparison, the LAA
location in the test set localizers needs to be manually annotated.
The script `./AnnotateLAAManually.py` can assist with this annotation process.
It is expected to produce files like `./paper/{dataset_name}.COORDINATES.xlsx`
containing the manually determined ground truth LAA coordinates. These
are then needed during evaluation.



## Questions

If you have questions regarding the code, please open an issue.


## LICENSE

MIT License

Copyright (c) 2025 aydin demircioglu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
