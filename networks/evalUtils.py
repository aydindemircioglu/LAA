#!/usr/bin/python3

import json
import os
import pydicom
import cv2
from glob import glob
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

import argparse
import os
#from util import util
#import torch
from scipy.stats import iqr
from utils import recreatePath


def revertCoordinates (dcmpath, c, verbose = False):
    #upper, optimal, lower =  row["upperLimit"],  row["optimalLayer"], row["lowerLimit"]
    m = pydicom.dcmread(dcmpath)
    z = m[0x0020,0x0032].value[2] # z-axis is along the patient  head to feet
    p = m[0x0028,0x0030].value

    if p[0] != p[1]:
        raise Exception ("mmh. pixel spacing not uniform. really now?")

    if verbose == True:
        print (z,c,p[0])

    # c = (z-x)/p  --> x = z - c*p
    x = round(z-c*p[0])
    return x


def convertCoordinates (dcmpath, x, verbose = False):
    #upper, optimal, lower =  row["upperLimit"],  row["optimalLayer"], row["lowerLimit"]
    m = pydicom.dcmread(dcmpath)
    z = m[0x0020,0x0032].value[2] # z-axis is along the patient  head to feet
    p = m[0x0028,0x0030].value

    if p[0] != p[1]:
        raise Exception ("mmh. pixel spacing not uniform. really now?")

    if verbose == True:
        print (z,x,p[0])
    coord = round((z-x)/p[0])
    return coord




def iou(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6
    #print (labels.shape)
    #print (outputs.shape)
    outputs = np.asarray(outputs).astype(bool)
    labels = np.asarray(labels).astype(bool)

    intersection = (outputs & labels).sum((0,1))
    union = (outputs | labels).sum((0, 1))

    iou = (intersection + SMOOTH) / (union + SMOOTH)
    # thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    # thresholded
    return iou  # Or thresholded.mean()



def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        print ("GT:", im1.shape)
        print ("IMG:", im2.shape)
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())




#
