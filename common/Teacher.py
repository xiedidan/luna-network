# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

import caffe

import numpy as np
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

    # interface
    def train(self):
        caffe.set_device(0)
        caffe.set_mode_gpu()

        trainLoss = np.zeros(self.iterationCount)
        testAccu = np.zeros(self.iterationCount)

        plt.ion()
        fig = plt.figure()
        plt.grid(True)
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = ax1.twinx()

        solver = caffe.SGDSolver(self.netPath + "solver.prototxt")
        if len(self.snapshotFilename) != 0:
            solver.restore(self.netPath + self.snapshotPath + self.snapshotFilename)
        baseIter = solver.iter

        for i in range(self.iterationCount):
            solver.step(1)

            trainLoss[i] = solver.net.blobs["loss"].data
            testAccu[i] = solver.test_nets[0].blobs["accu"].data
            if np.mod(i, 30) == 0:
                ax1.plot(range(baseIter, baseIter + i), trainLoss[0:i], "b-", label="Loss", linewidth=1)
                ax2.plot(range(baseIter, baseIter + i), testAccu[0:i], "g-", label="Accu", linewidth=1)
                # plt.show()
                plt.pause(0.00000001)
                plt.savefig(self.netPath + "loss-accu.png")

            matplotlib.pyplot.show()
