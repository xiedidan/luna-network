# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")
from Plotter import Plotter

import caffe

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib

class Teacher(object):
    # constructor
    def __init__(self, dataPath, netPath, batchSize = 2, snapshotPath = "snapshot/", snapshotFilename = "", iterationCount = 100000, queueSize = 32, volSize = 64):
        self.dataPath = dataPath
        self.netPath = netPath
        self.snapshotPath = snapshotPath
        self.snapshotFilename = snapshotFilename
        self.phase = "train"
        self.phaseSubPath = self.phase + "/"
        self.iterationCount = iterationCount
        self.batchSize = batchSize
        self.volSize = volSize
        self.queueSize = queueSize

    #helper

    # interface
    def train(self):
        caffe.set_device(0)
        caffe.set_mode_gpu()

        solver = caffe.SGDSolver(self.netPath + "solver.prototxt")
        if len(self.snapshotFilename) != 0:
            solver.restore(self.netPath + self.snapshotPath + self.snapshotFilename)
        baseIter = solver.iter

        plotter = Plotter()
        plotter.initLossAndAccu(baseIter, self.iterationCount, self.netPath)
        plotter.initResult(interval = 50)

        for i in range(self.iterationCount - baseIter):
            solver.step(1)

            # plot loss and accu
            loss = solver.net.blobs["loss"].data
            accu = solver.test_nets[0].blobs["accu"].data
            plotter.plotLossAndAccu(loss, accu)

            # plot result and label
            data = solver.net.blobs["data"].data
            data = np.squeeze(data[0, 0, :, :, :])

            result = solver.net.blobs["conv_i64c2o64_output_1"].data
            result = np.squeeze(result[0, 1, :, :, :])
            # print(result)

            label = solver.net.blobs["label"].data
            label = np.squeeze(label[0, 0, :, :, :])

            plotter.plotResult(data, label, result)
