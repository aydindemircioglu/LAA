#!/usr/bin/python3

import json
import os
import pydicom
import cv2
from glob import glob
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from pprint import pprint
import pickle
import sys
import argparse
import os
import torch
import scipy
from scipy.stats import iqr
from evalUtils import *
from utils import recreatePath
sys.path.append("..")
from helpers import getAnnotationFromPNG, getAnnotation

from evalUtils import dice
from modelSelection import *
from computeExtraInfos import *

def getModelName (fm):
    _, model, submodel, margin, _ = fm.split("_")
    model = model.split("/")[-1]
    margin = int(margin)
    return model, submodel, margin


def createExampleImages ():
    fimg = "/data/data/vorhofohr/slices/topo_1736045.png"
    img = cv2.imread(fimg)
    img[:,:,0] = img[:,:,1]
    img[:,:,2] = img[:,:,1]
    img = img[0:512-2*84,84:512-84,:]
    cv2.imwrite("./results/example_topo.png", img)

    #img[122:172,:, :] =     img[122:172,:, :]*0
    #img[122:172,:, 1] = 128
    img[122:172,:, 2] = 0
    img[122:172,:, 0] = 0
    cv2.imwrite("./results/example_topo_ann.png", img)


    # read slices of nii.gz
    CT, ctNullPos, sliceThickness  = loadCT("/data/data/vorhofohr/CT/1736045/CTA.nii.gz")
    itkSeg = sitk.ReadImage("/data/data/vorhofohr/CT/1736045/segmentation.nii")
    Seg = sitk.GetArrayFromImage(itkSeg)
    slices = {}
    for k in range(Seg.shape[0]):
        nPixel = np.sum(Seg[k,:,:] == 8)
        if nPixel > 0:
            #print (k, nPixel)
            slices[k] = nPixel

    # get biggest slices
    maxKey = max(slices, key=slices.get)
    N = 4
    d = 32
    # get N slices and make them nice
    m = np.zeros((512+d*N-d,512+d*N-d,3), dtype = np.uint8)*0+255
    for s in [True, False]:
        for n in range(N):
            sseg = Seg[maxKey-n,:,:].copy()
            sseg = (sseg == 8)

            sct = CT[maxKey-n,:,:].copy()
            sct[:,0:10] = 255
            sct[:,512-10:512] = 255
            sct[0:10,:] = 255
            sct[512-10:512, :] = 255
            if s == False:
                sseg = sseg*0
            m[d*n:d*n+512, d*n:d*n+512, 0] = sct - sseg*48
            m[d*n:d*n+512, d*n:d*n+512, 1] = sct - sseg*48
            m[d*n:d*n+512, d*n:d*n+512, 2] = sct
        cv2.imwrite(f"./results/example_CT_{s}.png", m)




if __name__ == '__main__':
    results = []

    # read val and test
    for dset in ["val", "test"]:
        fm = glob (f"./mmdetect/results/*_{dset}.pkl")
        assert (len(fm) == 1)
        model, submodel, margin = getModelName(fm[0])
        with open(fm[0], 'rb') as f:
            data = pickle.load(f)
        imgPath = "./images/"+os.path.basename(fm[0]).replace(".pkl", '')
        recreatePath (imgPath)

        # this is stupid, maybe change
        dname = {"val": "UKE.val", "test": "ELI.test"}[dset]
        dheart = pd.read_excel(f"../download/data/meta_{dname}_final.xlsx")
        dheart["AccNr"] = dheart["Accession Nr"].astype(str)

        # remove this duplicated (?) pat
        df0 = pd.read_csv(f"./data/{dset}.csv") # just for reference
        # df0 = df0.query("`Accession Nr` != 4270213").copy()
        # df0 = df0.query("`Accession Nr` != '4270213'").copy()
        stats = addStats (data, df0, dheart, margin, imgPath)

        # join with invalid/errors...
        fcoords = f"../coords/{dset}.COORDINATES.xlsx"
        if os.path.exists (fcoords):
            # remove invalids
            coords = pd.read_excel(fcoords)
            invalids = coords.query("invalid == 1")
            inv_ids = invalids.img_id
            stats = stats.query("img_id not in @inv_ids").copy()

        # for stats we only work with those with preds
        sPred = stats.query("Prediction == 1").copy()

        nopreds = np.sum(1-stats["Prediction"].values)
        SL_Heart_clinical = np.mean(stats["SL_Heart_clinical"])
        SL_Heart_ideal = np.mean(stats["SL_Heart_ideal"])
        SL_LAA_ideal = np.mean(stats["SL_LAA_ideal"])
        SL_LAA_with_margin = np.mean(stats["SL_LAA_with_margin"])
        SL_predicted = np.mean(sPred["SL_LAA_pred"])

        cutins_lower = np.mean(sPred["Cutin_Lower"])
        cutins_upper = np.mean(sPred["Cutin_Upper"])
        errors_lower = np.mean(sPred["Diff_Lower"])
        errors_upper = np.mean(sPred["Diff_Upper"])
        errors_lower_sd = np.std(sPred["Diff_Lower"])
        errors_upper_sd = np.std(sPred["Diff_Upper"])

        abs_errors_lower = np.mean(np.abs(sPred["Diff_Lower"]))
        abs_errors_upper = np.mean(np.abs(sPred["Diff_Upper"]))
        abs_errors_lower_sd = np.std(np.abs(sPred["Diff_Lower"]))
        abs_errors_upper_sd = np.std(np.abs(sPred["Diff_Upper"]))

        # fake array, easier this way
        Dice = np.mean(np.array(sPred["Dice"])*100)
        Dice_sd = np.std(np.array(sPred["Dice"])*100)
        acc = np.sum(1-sPred["Cutin"])/len(sPred)
        nerrors = np.sum(sPred["Cutin"])
        Iou = np.mean(np.array(sPred["IoU"])*100)
        Iou_sd = np.std(np.array(sPred["IoU"])*100)

        ED_Heart_clinical = np.mean(np.array(stats["ED_Heart_clinical"]))
        ED_Heart_clinical_sd = np.std(np.array(stats["ED_Heart_clinical"]))
        ED_Expected = np.mean(np.array(stats["ED_Expected"]))
        ED_Expected_sd = np.std(np.array(stats["ED_Expected"]))

        results.append({"Dataset": dset, "Model": f"{model}", "Margin": margin,
                        "No_Prediction": f"{nopreds}",
                        "Cutins_Lower": f"{cutins_lower*100:.1f}",
                        "Cutins_Upper": f"{cutins_upper*100:.1f}",
                        "Error_Lower": f"{errors_lower:.2f} +/- {errors_lower_sd:.2f}",
                        "Error_Upper": f"{errors_upper:.2f} +/- {errors_upper_sd:.2f}",
                        "Abs_Error_Lower": f"{abs_errors_lower:.2f} +/- {abs_errors_lower_sd:.2f}",
                        "Abs_Error_Upper": f"{abs_errors_upper:.2f} +/- {abs_errors_upper_sd:.2f}",
                        "Errors": f"{nerrors}/{len(sPred)}",
                        "Accuracy": f"{100*acc:.1f}",
                        "Dice": f"{Dice:.1f} +/- {Dice_sd:.1f}",
                        "IoU": f"{Iou:.1f} +/- {Iou_sd:.1f}",
                        "ED_Expected_mean": f"{ED_Expected:.1f}",
                        "ED_Expected_SD": f"{ED_Expected_sd:.1f}",
                        "ED_Expected": f"{ED_Expected:.2f} +/- {ED_Expected_sd:.2f}",
                        "ED_Heart_clinical": f"{ED_Heart_clinical:.2f} +/- {ED_Heart_clinical_sd:.2f}",
                        "ED_Heart_clinical_mean": np.round(ED_Heart_clinical,1),
                        "SL_LAA_ideal": f"{SL_LAA_ideal:.1f}",
                        "SL_LAA_with_margin": f"{SL_LAA_with_margin:.1f}",
                        "SL_predicted": f"{SL_predicted:.1f}",
                        "SL_Heart_clinical": f"{SL_Heart_clinical:.1f}",
                        "SL_Heart_ideal": f"{SL_Heart_ideal:.1f}"
                         })

        # p-values; alt hyp: LAA < HAA --> if significant this holds true.
        sPred["ED_Heart_clinical_red"] = sPred["ED_Heart_clinical"] * (1-1/3)
        p = scipy.stats.wilcoxon (sPred["ED_Expected"], sPred["ED_Heart_clinical_red"], alternative = "less")
        print (f"p-Value on {dset}:{p[1]}")


    results = pd.DataFrame(results)
    tModel = {"tood": "TOOD", "cascadeRCNN": "Cascade-RCNN", "vfnet": "VFNet"}
    results["Model"] = [tModel[k] for k in results["Model"].values]
    results.to_excel("./results/final_results.xlsx")

    # for paper as well
    # No_predictions = Clinical accuracy later.
    presults = results[["Dataset",  "ED_Expected", "ED_Heart_clinical", "Accuracy", "Errors", "Dice", "Abs_Error_Upper", "Abs_Error_Lower"]].copy()
    reduction = 1- results["ED_Expected_mean"].astype(np.float64)/results["ED_Heart_clinical_mean"].astype(np.float64)
    presults["Reduction"] = np.round(100*reduction, 1)
    presults = presults[["Dataset",  "ED_Expected", "ED_Heart_clinical", "Reduction", "Accuracy", "Errors", "Dice", "Abs_Error_Upper", "Abs_Error_Lower"]].copy()

    presults.to_excel("../paper/Table_4.xlsx", index = False)


    # also generate images etc
    #
    # make image grey again
    createExampleImages ()


#
