#
import dicom2nifti
from glob import glob
import os
import pandas as pd
import nibabel as nib
import numpy as np
from PIL import Image
import cv2
import nibabel.processing
from scipy.ndimage import center_of_mass
import pydicom
import SimpleITK as sitk
from evalUtils import *
from utils import recreatePath, pseudoRGB


# UKE.train contains all UKE, including validation
basepathes = {"UKE.train": "/data/data/vorhofohr/CT", "ELI.test": "/data/data/vorhofohr/CT.ELI"}



def bounding_box(img):
    omin, omax = np.where(np.sum(img, axis = (0,1)))[0][[0, -1]]
    rmin, rmax = np.where(np.sum(img, axis = (0,2)))[0][[0, -1]]
    cmin, cmax = np.where(np.sum(img, axis = (1,2)))[0][[0, -1]]
    return omin, omax, rmin, rmax, cmin, cmax # y1, y2, x1, x2


def nrm (vol, minval = None):
    if minval is None:
        minval = np.min(vol)
    v = (vol - minval)/(np.max(vol) - minval)
    v = np.asarray(255*v, dtype = np.uint8)
    return (v)


def cvt (nbvol):
    vol = nbvol.get_fdata()
    v = (vol - np.min(vol))/(np.max(vol) - np.min(vol))
    v = np.asarray(255*v, dtype = np.uint8)
    return (v)

def dsp (z):
    return Image.fromarray(z)



def loadCT(fCT):
    # have a name and uuid
    try:
        ctImgITK = sitk.ReadImage(fCT)
    except:
        print ("Failed", fCT)
        raise Exception ("FAILED")

    ctNullPos = float(ctImgITK.GetOrigin()[2])
    sliceThickness = float(ctImgITK.GetSpacing()[2])

    wmin = -100
    wmax = 500
    ctImg = sitk.Cast(sitk.IntensityWindowing(ctImgITK, windowMinimum=wmin, windowMaximum=wmax, outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)
    ctImg = sitk.GetArrayFromImage(ctImg)
    return ctImg, ctNullPos, sliceThickness


recreatePath("/data/data/vorhofohr/mosaics/")
recreatePath("/data/data/vorhofohr/slices/")


for dset in ["UKE.train", "UKE.val", "ELI.test"]:
    studies = pd.read_excel(f"./data/{dset}_selected.xlsx")
    df = []
    split = dset.split(".")[-1]
    print ("\n\n\n### ", split)
    for i, (idx, row) in enumerate(studies.iterrows()):
        if dset == "ELI.test":
            fCT = f"/data/data/vorhofohr/CT.ELI/{row['Accession Nr']}/CTA.nii.gz"
            fTopo = "/data/data/vorhofohr/topos.ELI/"+ os.path.basename(row["dcm_path"])
        else:
            fCT = f"/data/data/vorhofohr/CT/{str(row['Accession Nr'])}/CTA.nii.gz"
            fTopo = row["topo_dcmpath"]

        if os.path.exists(fCT) == False:
            print (fCT, "missing")
            continue
        if os.path.exists(fTopo) == False:
            print (fTopo, "missing")
            continue

        # read the topo and the CT too to see if it works out all well
        try:
            CT, ctNullPos, sliceThickness  = loadCT(fCT)

            fSeg = os.path.join(os.path.dirname(fCT), "segmentation.nii")
            itkSeg = sitk.ReadImage(fSeg)
            Seg = sitk.GetArrayFromImage(itkSeg)

            if Seg.shape != CT.shape:
                print ("MISMATCH")
                print (Seg.shape, CT.shape)
                continue

            # LAA
            sNr = 8

            slices = {}
            if np.sum(Seg == sNr) < 500:
                continue
            for k in range(Seg.shape[0]):
                nPixel = np.sum(Seg[k,:,:] == sNr)
                if nPixel > 0:
                    #print (k, nPixel)
                    slices[k] = nPixel

            # convert first slice number to coordinate
            topoCoords = [convertCoordinates(fTopo, ctNullPos + sliceThickness*k) for k in list(slices.keys())]

            # get image from topo for training/testing
            topoITK = sitk.ReadImage(fTopo)
            topoImg = sitk.GetArrayFromImage(topoITK)
            topoImg = pseudoRGB (nrm(topoImg[0,:,:],-256), method = "clahe", visualize = False)
            fTopoImg = os.path.join("/data/data/vorhofohr/slices/topo_"+str(row["Accession Nr"])+".png")
            cv2.imwrite(fTopoImg, topoImg)

            # write just an impression
            visImg = topoImg.copy()
            visImg[np.min(topoCoords),:,2] = 255
            visImg[np.max(topoCoords),:,2] = 255
            fSlice = os.path.join("/data/data/vorhofohr/mosaics/topo_"+str(row["Accession Nr"])+".jpg")
            cv2.imwrite(fSlice, visImg)

            upper = np.min(topoCoords)
            lower = np.max(topoCoords)
            df.append({"Accession Nr": row["Accession Nr"], "Upper": upper, "Lower": lower,
                        "PatientID": row["PatientID"],
                        "Img": fTopoImg, "Topo": fTopo, "CT": fCT, "Split": split})

            # add true coordinates to .xlsx for later use
            studies.at[idx, "Upper"] = np.min(topoCoords)
            studies.at[idx, "Lower"] = np.max(topoCoords)
            studies.at[idx, "CT_path"] = fCT
            studies.at[idx, "Topo_path"] = fTopo
            studies.at[idx, "Include"] = 1
        except Exception as e:
            # error
            print ("EXCEPTION", e)
            continue

    studies.to_excel(f"./data/{dset}_final.xlsx")

    net = pd.DataFrame(df)
    net.to_excel(f"./networks/data/{split}.xlsx")
    net.to_csv(f"./networks/data/{split}.csv")

#
