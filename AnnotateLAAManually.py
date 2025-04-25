#!/usr/bin/env python
import PySimpleGUI as sg
import cv2
import numpy as np

import os
import numpy as np
import cv2
from scipy import ndimage
import sys

import math
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import shutil
from random import sample
import imageio
import pandas as pd
import time
import SimpleITK as sitk


zoomFactor = 1


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (25,256)
fontScale              = 1.0
fontColor              = (255,0,0)
lineType               = 2




def loadCT(row):
    # have a name and uuid
    ctImgITK = None
    ctName = row["instance_UUID"]
    print ("Loading from ")
    print (ctName)
    print ("TRUE UPPER:", row["true_upper"])
    print ("TRUE LOWER:", row["true_lower"])
    print ("MTA UPPER:", row["mta_upper"])
    print ("MTA LOWER:", row["mta_lower"])
    print ("RAD UPPER:", row["rad_upper"])
    print ("RAD LOWER:", row["rad_lower"])
    print ("NET UPPER:", row["net_upper"])
    print ("NET LOWER:", row["net_lower"])
    ctFile = "/data/data/vorhofohr/CT/" + str(ctName) + ".nrrd"
    try:
        ctImgITK = sitk.ReadImage(ctFile)
    except:
        print ("Failed. Trying instead  ")
        ctFile = "/data/data/vorhofohr/CT/" +ctName+"_"+str(row["Accession Nr"]) + ".nrrd"
        ctImgITK = sitk.ReadImage(ctFile)

    ctNullPos = float(ctImgITK.GetOrigin()[2])
    sliceThickness = float(ctImgITK.GetSpacing()[2])

    # PRINT MORE interesting frame coordinates
    print (ctNullPos)
    print (sliceThickness)
    curFrame = (row["true_upper"] - ctNullPos)//sliceThickness
    print ("TRUE UPPER:", curFrame)
    curFrame = (row["true_lower"] - ctNullPos)//sliceThickness
    print ("TRUE LOWER:", curFrame)
    curFrame = (row["mta_upper"] - ctNullPos)//sliceThickness
    print ("MTA UPPER:", curFrame)
    curFrame = (row["mta_lower"] - ctNullPos)//sliceThickness
    print ("MTA LOWER:", curFrame)
    curFrame = (row["rad_upper"] - ctNullPos)//sliceThickness
    print ("RAD UPPER:", curFrame)
    curFrame = (row["rad_lower"] - ctNullPos)//sliceThickness
    print ("RAD LOWER:", curFrame)
    curFrame = (row["net_upper"] - ctNullPos)//sliceThickness
    print ("NET UPPER:", curFrame)
    curFrame = (row["net_lower"] - ctNullPos)//sliceThickness
    print ("NET LOWER:", curFrame)


    wmin = -100
    wmax = 500
    ctImg = sitk.Cast(sitk.IntensityWindowing(ctImgITK, windowMinimum=wmin, windowMaximum=wmax, outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)
    ctImg = sitk.GetArrayFromImage(ctImg)
    return ctImg, ctNullPos, sliceThickness



def paintMarker (displayImg, r, g, b, width = 24, rpos = "left+right"):
    w = displayImg.shape[1]
    if "left" in rpos:
        displayImg[:, 0:width, 0] = r
        displayImg[:, 0:width, 1] = g
        displayImg[:, 0:width, 2] = b
    if "right" in rpos:
        displayImg[:, w-width:w, 0] = r
        displayImg[:, w-width:w, 1] = g
        displayImg[:, w-width:w, 2] = b
    return displayImg


def main():
    dataset = None
    if "UKE" in os.path.abspath(__file__):
        dataset = "val.UKE"
    if "ELI" in os.path.abspath(__file__):
        dataset = "test.ELI"
    if "aydin" in os.path.abspath(__file__):
        dataset = "val.UKE"
    print ("Loading dataset", dataset)

    # read xlsx file
    xl = pd.ExcelFile ("./paper/"+dataset+".COORDINATES.xlsx")
    sheetName = xl.sheet_names[0]
    data = xl.parse(sheetName, header = 0)
    print ("Loaded", data.shape)

    # create window
    sg.theme('Black')

    # define the window layout
    taskbar = [[sg.Button('<', size=(3, 1), font='Helvetica 14'),
               sg.Button('Upper', size=(10, 1), font='Helvetica 14'),
               sg.Button('Lower', size=(10, 1), font='Helvetica 14'),
               sg.Button('>', size=(3, 1), font='Helvetica 14')]]
    layout = [[sg.Image(filename='', key='image', size=(512,512))],
                [sg.Column(taskbar, size = (512, 48), element_justification='c',justification="center")]]

    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration', layout,
                        return_keyboard_events=True)

    curCT = 0
    maxCT = data.shape[0]

    direction = 1
    while True:
        if curCT < 0:
            curCT = 0
        if curCT > maxCT - 1:
            curCT = maxCT - 1
        row = data.iloc[curCT]
        try:
            imgStack, ctNullPos, sliceThickness = loadCT (row)
        except Exception as e:
            print(e)
            # for now advance
            print ("CT does not exist. Trying next file")
            curCT = curCT + direction
            if curCT < 0:
                direction = 1
            if curCT > maxCT - 1:
                direction = -1
            continue
        # save it first

        # coordinates
        maxFrame = imgStack.shape[0]

        jumpToNegAILower = True
        if jumpToNegAILower == True:
            foundNext = False
            print ("Pushing forwards.. from ", curCT)
            curFrame = int(data.at[curCT, "net_upper"])
            curFrame = (curFrame - ctNullPos)//sliceThickness
            if curFrame < maxFrame:
                curCT = curCT + direction
                continue


        jumpToNegAI = False
        if jumpToNegAI == True:
            foundNext = False
            print ("Pushing forwards.. from ", curCT)
            curFrame = (row["net_lower"] - ctNullPos)//sliceThickness
            if curFrame >= 0:
                curCT = curCT + direction
                continue

        data.to_excel(excel_writer="./paper/"+dataset+".COORDINATES.xlsx", index = False)
        if curCT % 15 == 0:
            from datetime import datetime
            now = datetime.now()
            dt_string = now.strftime("%Y%m%d%H%M%S")
            data.to_excel(excel_writer="./paper/"+dataset+".COORDINATES_"+dt_string+".xlsx", index = False)


        if zoomFactor != 1:
            newSize = np.asarray(zoomFactor*np.asarray(imgStack.shape[1:3]), dtype = np.uint32)
            newImgStack = np.zeros ((imgStack.shape[0], newSize[0], newSize[1]))
            for j in range(maxFrame):
                newImgStack[j,:,:] = cv2.resize(imgStack[j,:,:], (newSize[0], newSize[1]))
            imgStack = newImgStack.copy()

        try:
            curFrame = int(data.at[curCT, "net_upper"])
            curFrame = (curFrame - ctNullPos)//sliceThickness
            curFrame = int(curFrame)
            print ("START AT", curFrame, maxFrame)
            if curFrame < 0:
                curFrame = 0
            if curFrame > maxFrame - 1:
                curFrame = maxFrame - 1
            print ("corrected to", curFrame, maxFrame)
        except:
            curFrame = 0
            pass
        upperFrame = None
        lowerFrame = None
        try:
            upperFrame = int(data.at[curCT, "true_upper_frame"])
            if upperFrame is np.nan:
                upperFrame = None
        except:
            pass
        try:
            lowerFrame = int(data.at[curCT, "true_lower_frame"])
            if lowerFrame is np.nan:
                lowerFrame = None
        except:
            pass

        invalid = 0
        try:
            invalid = int(data.at[curCT, "INVALID"])
        except:
            data.at[curCT, "INVALID"] = int(invalid)
        displayed = False
        while True:
            event, values = window.read(timeout = 20)
            if event is None:
                event = "Exit"
            if "TIMEOUT" in event and displayed == True:
                continue

            if event == "MouseWheel:Up":
                curFrame = curFrame - 1
            if event == "MouseWheel:Down":
                curFrame = curFrame + 1
            if  event == "w:25" or event == "w":
                curFrame = curFrame - 1
            if event == "s:39" or event == "s":
                curFrame = curFrame + 1
            if  event == "e:XX" or event == "e":
                curFrame = curFrame - 10
            if event == "d:XX" or event == "d":
                curFrame = curFrame + 10
            if  event == "w:25" or event == "r":
                curFrame = curFrame - 50
            if event == "s:39" or event == "f":
                curFrame = curFrame + 50
            if "BackSpace" in event:
                invalid = 1 - invalid
                data.at[curCT, "INVALID"] = int(invalid)
                print ("invalidating")
            if event == "x":
                upperFrame = None
                lowerFrame = None
                data.at[curCT, "true_upper_frame"] = None
                data.at[curCT, "true_upper"] = None
                data.at[curCT, "true_lower_frame"] = None
                data.at[curCT, "true_lower"] = None
                print ("removed annotation")
                # remove
            # escape
            if curFrame < 0:
                curFrame = 0
            if curFrame > maxFrame - 1:
                curFrame = maxFrame - 1
            if event == 'Exit' or event == sg.WIN_CLOSED or "Escape" in event:
                data.to_excel(excel_writer="./paper/"+dataset+".COORDINATES.xlsx", index = False)
                return
            elif event == "<":
                curCT = curCT - 1
                direction = -1
                break
            elif event == ">":
                curCT = curCT + 1
                direction = +1
                break
            elif event == 'Upper' or event == "o":
                upperFrame = curFrame
                data.at[curCT, "true_upper_frame"] = curFrame
                data.at[curCT, "true_upper"] = curCoordinate
                print ("Upper", curFrame, ":", curCoordinate)
            elif event == 'Lower' or event == "l":
                lowerFrame = curFrame
                data.at[curCT, "true_lower_frame"] = curFrame
                data.at[curCT, "true_lower"] = curCoordinate
                print ("Lower", curFrame, ":", curCoordinate)

            orgdisplayImg = np.zeros(imgStack.shape[1:3]+(3,))
            orgdisplayImg[:,:,0] = imgStack[curFrame,:,:]
            orgdisplayImg[:,:,1] = imgStack[curFrame,:,:]
            orgdisplayImg[:,:,2] = imgStack[curFrame,:,:]
            displayImg = orgdisplayImg.copy()
            curCoordinate = ctNullPos + curFrame*sliceThickness
            cv2.putText(displayImg, str(curFrame), (256-8,24), font, fontScale, fontColor, lineType)

            if upperFrame is not None:
                cv2.putText(displayImg, "U"+str(upperFrame), (256-128,24), font, fontScale, fontColor, lineType)
            if lowerFrame is not None:
                cv2.putText(displayImg, "L"+str(lowerFrame), (256+128,24), font, fontScale, fontColor, lineType)

            if invalid == 1:
                #paintMarker (displayImg, 0, 0, 255, 24, "top+bottom")
                cv2.putText(displayImg, "INVALID", (48,192), font,
                    4, (0,0,255), lineType)

            if curFrame == upperFrame:
                paintMarker (displayImg, 255, 0, 0, 24, "left")

            if curFrame == lowerFrame:
                paintMarker (displayImg, 0, 255, 0, 24, "right")
            imgbytes = cv2.imencode('.png', displayImg)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)
            displayed = True


main()
