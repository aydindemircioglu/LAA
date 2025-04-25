#!/usr/bin/python3

import json
import os
import cv2
from glob import glob
import numpy as np
#from scipy.signal import find_peaks
from os.path import isfile, join
import shutil
import subprocess
import numpy as np

import argparse
import os
#from util import util
#import torch
from utils import recreatePath


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--model', type=str, default='unet', help='name of the model to evaluate.')
        self.parser.add_argument('--submodel', type=str, default='slim', help='name of the model to evaluate.')
        self.parser.add_argument('--margin', type=int, default=0, help='safety margin in mm')
        self.parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
        self.parser.add_argument('--dry', action='store_true', default=False, help='dry run')
        self.parser.add_argument('--final', action='store_true', default=False, help='predict on final model')
        self.parser.add_argument('--testset', type=str, default='test', help='name of the model to evaluate.')

        # only parameters for tuning
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')

        # input/output sizes
        self.parser.add_argument('--epoch', type=int, default=-1, help='which epoch to evaluate')
        self.initialized = True


    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        return self.opt


def execute(cmd, opt, path = "."):
    if opt.dry == False:
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, cwd = path)
        output, error = process.communicate()
        print (output, error)
        return output, error
    print (cmd)



def clearDir (subpath, opt):
    cmd = "rm -rf /data/data/vorhofohr/"+subpath
    execute(cmd, opt)



if __name__ == '__main__':
    opt = BaseOptions().parse()
    curDir = os.path.dirname(os.path.realpath(__file__))
    rootDir = "/data/data/vorhofohr/"

    # check if config is there
    submodelOpts = "configs/" + opt.model + "_" + opt.submodel + ".py"
    if os.path.exists (os.path.join("./mmdetect", submodelOpts)) == False:
        raise Exception ("Config not found", submodelOpts)

    if opt.final == True:
        print ("Predicting with final model.")
        expName = "final_" + opt.model + "_" + opt.submodel +"_" + str(opt.margin)  # no other parameters-- its the final model
        recreatePath (os.path.join(rootDir, "experiments", "results", expName + "_" + opt.testset), opt.dry)

        # get latest checkpoint
        with open('/data/data/vorhofohr/experiments/checkpoints/' + expName + '/last_checkpoint') as f:
            fmodel = f.readlines()
        fmodel = fmodel[0]
        cmd = "python3 ./test.py  " + submodelOpts + " " + fmodel
        cmd = cmd + " --out ./results/"+expName+"_"+opt.testset+".pkl"
        cmd = cmd + " --margin " + str(opt.margin)
        cmd = cmd + " --fold -1"
        cmd = cmd + " --testset " + opt.testset
        print (cmd)
        execute(cmd, opt, path = os.path.join(curDir, "./mmdetect"))
    else:
        # for now this is not debatable
        nCV = 5
        for fold in range(nCV):
            expName =  opt.model + "_" + str(opt.submodel) + "_" + str(opt.margin) + "_" + str(fold) + "_lr_" + str(opt.lr)
            recreatePath (os.path.join(rootDir, "experiments", "results", expName), opt.dry)

            # get latest checkpoint
            try:
                with open('/data/data/vorhofohr/experiments/checkpoints/' + expName + '/last_checkpoint') as f:
                    fmodel = f.readlines()
            except Exception as e:
                raise (e)
                continue
            fmodel = fmodel[0]
            cmd = "python3 ./test.py  " + submodelOpts + " " + fmodel
            cmd = cmd + " --out ./results/"+expName+".pkl"
            cmd = cmd + " --margin " + str(opt.margin)
            cmd = cmd + " --fold " + str(fold)
            print (cmd)
            execute(cmd, opt, path = os.path.join(curDir, "./mmdetect"))
            break

#

#120737  python3 ./test.py ./configs/cascadeRCNN_x101.py --fold 0 --margin 10 ./work_dirs/cascadeRCNN_x101/epoch_12.pth --out ./results_0_0.pkl
