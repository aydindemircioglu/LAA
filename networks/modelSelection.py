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
from scipy.stats import iqr
from evalUtils import *
from utils import recreatePath
sys.path.append("..")
from helpers import getAnnotationFromPNG, getAnnotation
import seaborn as sns
import matplotlib.pyplot as plt

from evalUtils import dice




def addStats (data, df0=None, dheart = None, margin = -1, imgPath = None):
    df0 = df0.copy()
    dpred = pd.DataFrame(data)
    newdf = []
    for i, (idx, row) in enumerate(dpred.iterrows()):
        # find record
        Img = row["img_path"]

        subdf = df0.query("Img == @Img")
        if len(subdf) == 0:
            print (row)
            raise Exception ("No GT found!")
        # does it have any prediction?
        dp = pd.concat([row, subdf.iloc[0]])

        accNr = str(dp["Accession Nr"])
        subdf = dheart.query("`AccNr` == @accNr")
        assert(len(subdf) == 1)
        heartUpper = subdf.iloc[0]["Heart_Upper"]
        heartLower = subdf.iloc[0]["Heart_Lower"]

        trueUpper = dp["True_upper"] = dp["Upper"] - margin
        trueLower = dp["True_lower"] = dp["Lower"] + margin
        gtUpper = dp["GT_upper"] = dp["Upper"]
        gtLower = dp["GT_lower"] = dp["Lower"]


        dp["SL_Heart_clinical"] = subdf.iloc[0]["rExp_CT_ScanLength"]
        dp["SL_Heart_ideal"] = heartLower-heartUpper
        dp["SL_LAA_with_margin"] = trueLower-trueUpper
        dp["SL_LAA_ideal"] = gtLower-gtUpper

        dp["ED_CTDIvol"] = CTDIvol = subdf.iloc[0]["rExp_CTDI"]
        dp["ED_k_factor"] = k = subdf.iloc[0]["rExp_k_factor"]

        dp["ED_Heart_clinical"] = subdf.iloc[0]["rExp_CT_ED"]
        dp["ED_Heart_ideal"] = CTDIvol * k * dp["SL_Heart_ideal"]/10
        dp["ED_LAA_ideal"] = CTDIvol * k * dp["SL_LAA_ideal"]/10
        dp
        if len(dp["pred_instances"]["bboxes"]) == 0:
            dp["Prediction"] = 0
            dp["Cutin"] = 0
            dp["ED_LAA_pred"] = 0
            dp["SL_LAA_pred"] = 0

            dp["Diff_Upper"] = -1
            dp["Diff_Lower"] = -1
        else:
            dp["Prediction"] = 1
            pred_upper = dp["Pred_upper"] = int(dp["pred_instances"]["bboxes"][0][1])
            pred_lower = dp["Pred_lower"] = int(dp["pred_instances"]["bboxes"][0][3])

            # wrt to prediction
            dp["Diff_Upper"] = trueUpper - pred_upper
            dp["Diff_Lower"] = pred_lower - trueLower

            # easier this way
            predArr = np.array([0]*512).reshape(512,1); predArr[pred_upper:pred_lower+1,:] = 1
            trueArr = np.array([0]*512).reshape(512,1); trueArr[trueUpper:trueLower+1,:] = 1
            dp["Dice"] = dice (predArr, trueArr)
            dp["IoU"] = iou (predArr, trueArr)

            dp["SL_LAA_pred"] = pred_lower - pred_upper
            dp["ED_LAA_pred"] = CTDIvol * k * dp["SL_LAA_pred"]/10

            dp["Cutin_Upper"] = int(gtUpper < pred_upper)
            dp["Cutin_Lower"] = int(pred_lower < gtLower)
            dp["Cutin"] = cutin = int(dp["Cutin_Lower"] + dp["Cutin_Upper"] > 0)
            if cutin == 1:
                if imgPath is not None:
                    paintTopo (dp, df0, margin, imgPath)

        # if we have a prediction...
        # ... we could have it done correctly
        Exp_No_cutin = dp["ED_LAA_pred"]
        # if we did not: we have to rescan with the whole heart
        Exp_Cutin = dp["ED_LAA_pred"] + dp["ED_Heart_clinical"]
        Exp = dp["Prediction"] * (dp["Cutin"]*Exp_Cutin + (1-dp["Cutin"])*Exp_No_cutin)

        # if we do not have it, we take the whole heart scan as we had it
        Exp = Exp + (1-dp["Prediction"])*dp["ED_Heart_clinical"]
        dp["ED_Expected"] = Exp

        newdf.append(dp)

    new = pd.DataFrame(newdf)
    return new


# df = true data with given margin
def paintTopo(record, df0=None, margin=-1, fpath="./tmp"):
    fimg = record["img_path"]

    # make image grey again
    img = cv2.imread(fimg)
    img[:,:,0] = img[:,:,1]
    img[:,:,2] = img[:,:,1]

    if len(record["pred_instances"]["bboxes"]) == 0:
        print ("Empty image. Skipping", fimg)
        return None

    # get preds/Gt
    upper = int(record["pred_instances"]["bboxes"][0][1])
    lower = int(record["pred_instances"]["bboxes"][0][3])

    subdf = df0.query("Img == @fimg").copy()
    assert(len(subdf) == 1)
    trueUpper = subdf.iloc[0]["Upper"] - margin
    trueLower = subdf.iloc[0]["Lower"] + margin
    gtUpper = subdf.iloc[0]["Upper"]
    gtLower = subdf.iloc[0]["Lower"]

    # paint RGB G = true, R = true LAA, G = Groundtruth(LAA+maring), B=pred
    for k in [trueUpper, trueLower, trueUpper-1,trueLower+1]:
        cv2.line(img, (0,k), (512,k), (64,64,192), 1)

    for k in [gtUpper, gtLower, gtUpper-1,gtLower+1]:
        cv2.line(img, (0,k), (512,k), (64,192,64), 1)

    for k in [upper, lower, upper-1, lower+1]:
        cv2.line(img, (0,k), (512,k), (192,64,64), 1)

    cv2.imwrite(os.path.join(fpath, os.path.basename(fimg)), img)
    pass



def getModelName (fm):
    model, submodel, margin, fold, _, LR = fm.split("_")
    model = model.split("/")[-1]
    margin = int(margin)
    fold = int(fold)
    LR = LR.replace(".pkl", '')
    return model, submodel, margin, fold, LR



if __name__ == '__main__':
    results = []
    models = glob ("./mmdetect/results/*.pkl")

    df0 = pd.read_csv("./data/train.csv")

    dheart = pd.read_excel("../download/data/meta_UKE.train_final.xlsx")
    dheart["AccNr"] = dheart["Accession Nr"].astype(str)

    recreatePath ("./images")

    for fm in models:
        # there should be no final models there, but during testing of code there were,
        # so we ensure this
        if "lr" not in fm:
            continue

        with open(fm, 'rb') as f:
            data = pickle.load(f)
        imgPath = "./images/"+os.path.basename(fm).replace(".pkl", '')
        recreatePath (imgPath)

        model, submodel, margin, fold, LR = getModelName(fm)
        stats = addStats (data, df0, dheart, margin, imgPath)

        nopreds = np.sum(1-stats["Prediction"].values)

        if nopreds == len(stats):
            # no predictions at all.
            print ("No predictions at all..")
            results.append({"Model": f"{model}", "Margin": margin, "LR": LR,
                            "No_Prediction": f"{nopreds}",
                            "Cutins_Lower": -1,
                            "Cutins_Upper": -1,
                            "Error_Lower": -1,
                            "Error_Upper": -1,
                            "Abs_Error_Lower": -1,
                            "Abs_Error_Upper": -1,
                            "Accuracy": -1,
                            "Dice": -1,
                            "IoU": -1,
                            "ED_Expected_mean": -1,
                            "ED_Expected_SD": -1,
                            "ED_Expected":-1,
                            "ED_Heart_clinical": -1,
                            "SL_LAA_ideal": -1,
                            "SL_LAA_with_margin": -1,
                            "SL_predicted": -1,
                            "SL_Heart_clinical": -1,
                            "SL_Heart_ideal": -1,
                             })
            continue

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
        Iou = np.mean(np.array(sPred["IoU"])*100)
        Iou_sd = np.std(np.array(sPred["IoU"])*100)

        ED_Heart_clinical = np.mean(np.array(stats["ED_Heart_clinical"]))
        ED_Expected = np.mean(np.array(stats["ED_Expected"]))
        ED_Expected_sd = np.std(np.array(stats["ED_Expected"]))

        results.append({"Model": f"{model}", "Margin": margin, "LR": LR,
                        "No_Prediction": f"{nopreds}",
                        "Cutins_Lower": f"{cutins_lower*100:.1f}",
                        "Cutins_Upper": f"{cutins_upper*100:.1f}",
                        "Error_Lower": f"{errors_lower:.2f} +/- {errors_lower_sd:.2f}",
                        "Error_Upper": f"{errors_upper:.2f} +/- {errors_upper_sd:.2f}",
                        "Abs_Error_Lower": f"{abs_errors_lower:.2f} +/- {abs_errors_lower_sd:.2f}",
                        "Abs_Error_Upper": f"{abs_errors_upper:.2f} +/- {abs_errors_upper_sd:.2f}",
                        "Accuracy": f"{100*acc:.1f}",
                        "Dice": f"{Dice:.1f} +/- {Dice_sd:.1f}",
                        "IoU": f"{Iou:.1f} +/- {Iou_sd:.1f}",
                        "ED_Expected_mean": f"{ED_Expected:.1f}",
                        "ED_Expected_SD": f"{ED_Expected_sd:.1f}",
                        "ED_Expected": f"{ED_Expected:.2f} +/- {ED_Expected_sd:.2f}",
                        "ED_Heart_clinical": f"{ED_Heart_clinical:.1f}",
                        "SL_LAA_ideal": f"{SL_LAA_ideal:.1f}",
                        "SL_LAA_with_margin": f"{SL_LAA_with_margin:.1f}",
                        "SL_predicted": f"{SL_predicted:.1f}",
                        "SL_Heart_clinical": f"{SL_Heart_clinical:.1f}",
                        "SL_Heart_ideal": f"{SL_Heart_ideal:.1f}"
                         })


    results = pd.DataFrame(results)
    tModel = {"tood": "TOOD", "cascadeRCNN": "Cascade-R-CNN", "vfnet": "VFNet", "sparseRCNN": "Sparse-R-CNN"}
    results["Model"] = [tModel[k] for k in results["Model"].values]

    results = results.sort_values(["Model", "Margin"])
    results.to_excel("./results/model_selection.xlsx")

    # for model selection remove models with not preds
    results["ED_Expected_mean"] = results["ED_Expected_mean"].astype(np.float32)
    results["Accuracy"] = results["Accuracy"].astype(np.float32)
    results = results.query("Accuracy > 0.0").copy()

    # for paper as well
    presults = []
    # this is stupid, but sort_value and groupby is error-prone
    plotData = []
    for m in set(results["Model"]):
        subdf = results.query("Model == @m").copy()
        subdf = subdf.sort_values(["ED_Expected_mean"])
        presults.append(subdf.iloc[0])
        cLR = subdf.iloc[0]["LR"]
        plotData.append(results.query("Model == @m and LR == @cLR"))
    presults = pd.DataFrame(presults)
    presults
    tmpresults = presults[["Model", "Margin", "LR", "ED_Expected", "Accuracy", "Dice", "Abs_Error_Upper", "Abs_Error_Lower"]].copy()
    tmpresults.to_excel("../paper/Table_3.xlsx", index = False)


    data = pd.concat(plotData).reset_index(drop = True)
    #data["Accuracy"] = data["Accuracy"].astype(np.float32)
    data["ED_Expected_mean"] = data["ED_Expected_mean"].astype(np.float32)
    data["Accuracy"] = data["Accuracy"].astype(np.float32)

    plt.figure()
    sns.set(style="white", context="talk")
    f, axs  = plt.subplots(1, 1, figsize = (12,10)) #gridspec_kw={'width_ratios': [1,2]})
    sns.lineplot(data=data, x="Margin", y="ED_Expected_mean", hue="Model", palette="husl", linewidth = 5)
    axs.set_xlabel("Margin [mm]",fontsize=23)
    axs.set_ylabel("Expected ED [%]",fontsize=23)
    axs.tick_params(axis='x', which='both', labelsize=20)
    axs.tick_params(axis='y', which='both', labelsize=20)
    axs.set_xticks([5,10,15,20]) # setting xticks with values way larger than your index squishes your data


    f.savefig("../paper/Figure_4.png", dpi = 300, bbox_inches='tight')
    plt.close('all')



#
