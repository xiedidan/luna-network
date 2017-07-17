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
    def drawProcess(self, baseIter, dataQueue):
        plt.ion()
        fig = plt.figure()
        plt.grid(True)
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = ax1.twinx()

        trainLoss = np.zeros(self.iterationCount)
        testAccu = np.zeros(self.iterationCount)

        for i in range(self.iterationCount - baseIter):
            loss, accu = dataQueue.get()
            trainLoss[i] = loss
            testAccu[i] = accu

            if np.mod(i, 300) == 0:
                ax1.plot(range(baseIter, baseIter + i), trainLoss[0:i], "b-", label="Loss", linewidth=1)
                ax2.plot(range(baseIter, baseIter + i), testAccu[0:i], "g-", label="Accu", linewidth=1)
                plt.pause(0.00000001)

            if np.mod(i, 1200) == 0:
                plt.savefig(self.netPath + "loss-accu.png")

            matplotlib.pyplot.show()

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

        for i in range(self.iterationCount - baseIter):
            solver.step(1)

            loss = solver.net.blobs["loss"].data
            accu = solver.test_nets[0].blobs["accu"].data

            plotter.plotLossAndAccu(loss, accu)

        '''''''''
        dataQueue = multiprocessing.Queue(4096)
        drawProc = multiprocessing.Process(target=self.drawProcess, args=(baseIter, dataQueue))
        drawProc.daemon = True
        drawProc.start()

        for i in range(self.iterationCount - baseIter):
            solver.step(1)

            loss = solver.net.blobs["loss"].data
            accu = solver.test_nets[0].blobs["accu"].data
            dataQueue.put(tuple((loss, accu)))
            # print("dataQueue.length: {0}".format(dataQueue.qsize()))
        '''