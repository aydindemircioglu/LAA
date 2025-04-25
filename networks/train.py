#!/usr/bin/python3

import json
import os
from glob import glob
import shutil
import subprocess
import numpy as np
import time
import random
import argparse
import os
from utils import recreatePath, removePath




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
        self.parser.add_argument('--final', action='store_true', default=False, help='train final model')

        # only parameters for tuning
        self.parser.add_argument('--lr', type=float, default=0.0, help='initial learning rate for adam')

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
        print ("Training final model.")
        expName = "final_" + opt.model + "_" + opt.submodel +"_" + str(opt.margin)  # no other parameters-- its the final model
        recreatePath (os.path.join(rootDir, "experiments", "checkpoints", expName), opt.dry)

        cmd = "python3 ./train.py  " + submodelOpts
        cmd = cmd + " --work-dir /data/data/vorhofohr/experiments/checkpoints/" + expName
        cmd = cmd + " --lr " + str(opt.lr)
        cmd = cmd + " --fold -1"
        cmd = cmd + " --margin " + str(opt.margin)
        print ("###########################")
        print ("EXECUTING:")
        print (cmd)
        print ("###########################")
        execute(cmd, opt, path = os.path.join(curDir, "./mmdetect"))
    else:
        nCV = 5
        for fold in range(nCV):
            expName =  opt.model + "_" + str(opt.submodel) + "_" + str(opt.margin) + "_" + str(fold) + "_lr_" + str(opt.lr)
            cmd = "python3 ./train.py  " + submodelOpts + " "
            cmd = cmd + "--work-dir /data/data/vorhofohr/experiments/checkpoints/" + expName + " "
            cmd = cmd + "--lr " + str(opt.lr)
            cmd = cmd + " --fold " + str(fold)
            cmd = cmd + " --margin " + str(opt.margin)
            print ("###########################")
            print ("EXECUTING:")
            print (cmd)
            print ("###########################")
            print (cmd)
            execute(cmd, opt, path = os.path.join(curDir, "./mmdetect"))
            break

    #
