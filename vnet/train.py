# -*- coding:utf-8 -*-

import sys
sys.path.append("./luna-data-pre-processing")
import os
import numpy as np
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


class Train(object):
    # constructor
    def __init__(self, dataPath = "./", iterationCount = 100000, batchSize = 2, queueSize = 30, volSize = 64):
        self.dataPath = dataPath
        self.iterationCount = iterationCount
        self.batchSize = batchSize
        self.volSize = volSize
        self.queueSize = queueSize

    # helper
    def loadSample(self, subPath, filename):
        serializer = NoduleSerializer.NoduleSerializer(self.dataPath)

        sample = {}
        image = serializer.readFromNpy(subPath + "nodules/", filename)
        groundTruth = serializer.readFromNpy(subPath + "groundTruths/", filename)
        sample["image"] = image
        sample["groundTruth"] = groundTruth

        return sample

    def loadAllSamples(self, subPath, filenames):
        samples = []
        for filename in filenames:
            sample = self.loadSample(subPath, filename)
            samples.append(sample)
        return samples

    def randomizedCrop(self, sample, rotateRatio, shiftRatio):
        image = sample["image"]
        groundTruth = sample["groundTruth"]

        if np.random.random() < rotateRatio:
            # rotate - p(3, 3) - 1 possibles
            rotateList = [[1, 0, 2],
                          [1, 2, 0],
                          [2, 0, 1],
                          [2, 1, 0],
                          [0, 2, 1]]
            dir = np.random.randint(0, 4)
            rotate = rotateList[dir]
            image = np.transpose(image, rotate)

        centerRange = [32, 96]
        if np.random.random() < shiftRatio:
            # shift - we shift +-8 max along each axis
            shiftx = np.random.randint(-8, 8)
            shifty = np.random.randint(-8, 8)
            shiftz = np.random.randint(-8, 8)
            xRange = centerRange + [shiftx, shiftx]
            yRange = centerRange + [shifty, shifty]
            zRange = centerRange + [shiftz, shiftz]
        else:
            xRange = centerRange
            yRange = centerRange
            zRange = centerRange

        crop = {}
        crop["image"] = image[zRange[0]:zRange[1], yRange[0]:yRange[1], xRange[0]:xRange[1]]
        crop["groundTruth"] = groundTruth[zRange[0]:zRange[1], yRange[0]:yRange[1], xRange[0]:xRange[1]]
        return crop

    def dataProcessor(self, dataQueue):
        npyFileList = glob(self.dataPath + "npy/nodules/*.npy")
        npyFileList = map(lambda filePath: os.path.basename(filePath), npyFileList)

        # load all samples
        samples = self.loadAllSamples("npy/", npyFileList)

        # crop on the fly since we want randomized input
        np.random.seed()
        for i in range(self.iterationCount):
            for j in range(self.batchSize):
                # get random sample
                noduleIndex = np.random.randint(0, len(samples) - 1)
                sample = samples[noduleIndex]

                # randomized cropping
                sample = self.randomizedCrop(sample, 0.3, 0.3)

                dataQueue.put(tuple((sample["image"], sample["groundTruth"])))
                # print(dataQueue.qsize())

    def trainProcessor(self, dataQueue, solver):
        batchData = np.zeros((self.batchSize, 1, self.volSize, self.volSize, self.volSize))
        batchLabel = np.zeros((self.batchSize, 1, self.volSize, self.volSize, self.volSize))

        trainLoss = np.zeros(self.iterationCount)

        for i in range(self.iterationCount):
            for j in range(self.batchSize):
                # get a batch in each iteration
                [nodule, groundTruth] = dataQueue.get()

                batchData[j, 0, :, :, :] = nodule.astype(dtype = np.float32)
                batchLabel[j, 0, :, :, :] = groundTruth.astype(dtype = np.float32)

            solver.net.blobs["data"].data[...] = batchData.astype(dtype = np.float32)
            solver.net.blobs["label"].data[...] = batchLabel.astype(dtype = np.float32)

            solver.step(1)

            trainLoss[i] = solver.net.blobs["loss"].data
            if np.mod(i, 10) == 0:
                plt.clf()
                plt.plot(range(0, i), trainLoss[0:i])
                plt.pause(0.00000001)

            # matplotlib.pyplot.show()

    # interface
    def train(self):
        dataQueue = multiprocessing.Queue(self.queueSize)

        dataProcess = []
        # start dataProcessor
        for i in range(1):
          dataProcess.append(multiprocessing.Process(target = self.dataProcessor, args = (dataQueue,)))
          dataProcess[i].daemon = True
          # dataProcess = threading.Thread(target = self.dataProcessor, args=(dataQueue,))
          dataProcess[i].start()

        caffe.set_device(0)
        caffe.set_mode_gpu()
        solver = caffe.SGDSolver("solver.prototxt")

        # start trainProcessor in main process
        self.trainProcessor(dataQueue, solver)

if __name__ == "__main__":
    trainer = Train("d:/project/tianchi/data/")
    trainer.train()