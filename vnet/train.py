# -*- coding:utf-8 -*-

import sys
sys.path.append("./luna-data-pre-processing")
import NoduleCropper

import caffe
import multiprocessing

class Train(object):
    # constructor
    def __init__(self, dataPath = "./", iterationCount = 100000, batchSize = 2, queueSize = 30):
        self.dataPath = dataPath
        self.iterationCount = iterationCount
        self.batchSize = batchSize
        self.solver = caffe.SGDSolver("solver.prototxt")
        self.dataQueue = multiprocessing.Queue(queueSize)

    # helper
    def dataProcessor(self, dataQueue, dataPath):
        cropper = NoduleCropper(dataPath)
        # crop on the fly since we want randomized input

    def trainProcessor(self, dataQueue, solver):
        for i in range(self.iterationCount):
            # get a batch in each iteration
            nodule, groundTruth = dataQueue.get()
            solver.net.blobs["data"] = nodule
            solver.net.blobs["label"] = groundTruth
            solver.step(1)

    # interface
    def train(self):
        # start dataProcessor
        dataProcess = multiprocessing.Process(target = self.dataProcessor, args = (self.dataQueue, self.dataPath))
        dataProcess.daemon = True
        dataProcess.start()

        # start trainProcessor in main process
        self.trainProcessor(self.dataQueue, self.solver)
