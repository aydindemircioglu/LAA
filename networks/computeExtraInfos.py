#
from glob import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage import center_of_mass
import pydicom
import SimpleITK as sitk

from evalUtils import *
from utils import recreatePath, pseudoRGB



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


if __name__ == '__main__':
    for dset in ["UKE.train", "UKE.val", "ELI.test"]:
        studies = pd.read_excel(f"../download/data/{dset}_final.xlsx")
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
            #try:
            if 1 == 1:
                CT, ctNullPos, sliceThickness  = loadCT(fCT)

                fSeg = os.path.join(os.path.dirname(fCT), "segmentation.nii")
                itkSeg = sitk.ReadImage(fSeg)
                Seg = sitk.GetArrayFromImage(itkSeg)

                if Seg.shape != CT.shape:
                    print ("MISMATCH")
                    print (Seg.shape, CT.shape)
                    continue

                # <Label Key="11" Red="0.7254901960784313" Green="0.6666666666666666" Blue="0.6078431372549019" Alpha="1"><![CDATA[pulmunary_artery]]></Label>
                # <Label Key="12" Red="1.0" Green="0.0" Blue="0.0" Alpha="1"><![CDATA[heart_atrium_left]]></Label>
                sNrs = [11, 12]

                slices = {}
                for sNr in sNrs:
                    for k in range(Seg.shape[0]):
                        nPixel = np.sum(Seg[k,:,:] == sNr)
                        if nPixel > 100:
                            #print (k, nPixel)
                            slices[k] = nPixel

                # convert first slice number to coordinate
                topoCoords = [convertCoordinates(fTopo, ctNullPos + sliceThickness*k) for k in list(slices.keys())]

                upperHeart = np.min(topoCoords)
                lowerHeart = np.max(topoCoords)
                studies.at[idx, "Heart_Upper"] =  upperHeart
                studies.at[idx, "Heart_Lower"] = lowerHeart

                # compute infos for ED
                fCTDcm = glob(os.path.dirname(row["CT_path"]) + "*/*/*")
                accNr = str(row["Accession Nr"])
                lctdi = []
                lSL = []
                for f in fCTDcm:
                    try:
                        ds = pydicom.read_file(f)
                        lctdi.append(ds["CTDIvol"].value)
                        lSL.append(float(ds.SliceLocation))
                    except:
                        pass
                try:
                    SL = (np.max(lSL)-np.min(lSL)+ds.SliceThickness)/10 # in cm
                    CTDI = np.mean(lctdi)
                    #DLP = CTDI*SL

                    k = 0.026
                    if ds.PatientSex == "F":
                        k = 0.045

                    ED = SL*CTDI*k
                    studies.at[idx, "rExp_CT_SliceThickness"] = ds.SliceThickness
                    studies.at[idx, "rExp_k_factor"] =  k
                    studies.at[idx, "rExp_CTDI"] = CTDI
                    studies.at[idx, "rExp_CT_ScanLength"] = SL
                    studies.at[idx, "rExp_CT_ED"] = ED
                except:
                    studies.at[idx, "rExp_CT_SliceThickness"] = -1
                    studies.at[idx, "rExp_k_factor"] =  -1
                    studies.at[idx, "rExp_CTDI"] = -1
                    studies.at[idx, "rExp_CT_ScanLength"] = -1
                    studies.at[idx, "rExp_CT_ED"] = -1

 
        studies.to_excel(f"../download/data/meta_{dset}_final.xlsx")

#
