# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")
import os
from glob import glob

import NoduleCropper
import NoduleSerializer

import caffe
import multiprocessing
import threading
import gc

import SimpleITK as sitk
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm

class Train(object):
    # constructor
    def __init__(self, dataPath = "./", netPath = "v1/", batchSize = 2, iterationCount = 100000, queueSize = 32, volSize = 64):
        self.dataPath = dataPath
        self.netPath = netPath
        self.phase = "train"
        self.phaseSubPath = self.phase + "/"
        self.iterationCount = iterationCount
        self.batchSize = batchSize
        self.volSize = volSize
        self.queueSize = queueSize

    # interface
    def train(self):
        caffe.set_device(0)
        caffe.set_mode_gpu()

        trainLoss = np.zeros(self.iterationCount)
        plt.ion()

        solver = caffe.SGDSolver(self.netPath + "solver.prototxt")
        for i in range(self.iterationCount):
            solver.step(1)

            trainLoss[i] = solver.net.blobs["dice_loss"].data
            if np.mod(i, 30) == 0:
                plt.clf()
                plt.grid(True)
                plt.plot(range(0, i), trainLoss[0:i])
                plt.pause(0.00000001)
            matplotlib.pyplot.show()

if __name__ == "__main__":
    trainer = Train("d:/project/tianchi/data/", "v3/", 2)
    trainer.train()