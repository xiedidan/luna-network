# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

import os
import numpy as np
from glob import glob

from NoduleCropper import NoduleCropper
from NoduleSerializer import NoduleSerializer
from scan import Scanner

import caffe
import multiprocessing
import threading
import gc

import SimpleITK as sitk
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib

class Test(object):
    # constructor
    def __init__(self, dataPath = "./", netPath = "v1/", snapshot = "_iter_37000.caffemodel", queueSize = 32):
        self.dataPath = dataPath
        self.netPath = netPath
        self.phrase = "test"
        self.phraseSubPath = self.phrase + "/"
        self.snapshot = snapshot
        self.queueSize = queueSize
        self.serializer = NoduleSerializer(self.dataPath, self.phrase)

    # helper
    def loadSample(self, filename):
        serializer = NoduleSerializer(self.dataPath, self.phrase)

        sample = {}
        image = serializer.readFromNpy("nodules/", filename)
        groundTruth = serializer.readFromNpy("groundTruths/", filename)
        sample["image"] = image
        sample["groundTruth"] = groundTruth

        return sample

    def loadAllSamples(self, filenames):
        samples = []
        for filename in filenames:
            sample = self.loadSample(filename)
            samples.append(sample)
        return samples

    def dataProcessor(self, dataQueue):
        npyFileList = glob(self.dataPath + self.phraseSubPath + "nodules/*.npy")
        npyFileList = map(lambda filePath: os.path.basename(filePath), npyFileList)

        # load all samples
        samples = self.loadAllSamples(npyFileList)

        for sample in samples:
            image = sample["image"]

            mean = np.mean(image)
            std = np.std(image)
            print("mean: {0}, std: {1}".format(mean, std))

            image = image.astype(np.float32)
            image -= mean.astype(np.float32)
            image /= std.astype(np.float32)
            sample["image"] = image

        # put data into dataQueue
        for i in range(len(samples)):
            sample = samples[i]
            dataQueue.put(tuple((sample["image"], sample["groundTruth"])))

    def testProcess(self, dataQueue, net, serializer):
        npyFileList = glob(self.dataPath + self.phraseSubPath + "nodules/*.npy")
        npyFileList = list(map(lambda filePath: os.path.basename(filePath), npyFileList))
        fileCount = len(npyFileList)

        totalAccuracy = 0.0
        totalLoss = 0.0
        for i in range(fileCount):
            image, groundTruth = dataQueue.get()
            net.blobs["data"].data[0, 0, :, :, :] = image.astype(dtype = np.float32)
            net.blobs["label"].data[0, 0, :, :, :] = groundTruth.astype(dtype = np.float32)

            out = net.forward()

            accuracy = out["accuracy"]
            loss = out["dice_loss"]
            totalAccuracy += accuracy
            totalLoss += loss
            print("i: {0}, accuracy: {1}, dice loss: {2}".format(i, accuracy, loss))
            # serializer.writeToNpy("results/", "{0}-{1}.npy".format(crop["seriesuid"], crop["number"]), np.squeeze(labels.astype(np.float32)))
            # serializer.writeToNpy("steps/", "{0}-{1}.npy".format(crop["seriesuid"], crop["number"]), crop["steps"])

        avgAcc = totalAccuracy / fileCount
        avgLoss = totalLoss / fileCount
        print("avg accuracy: {0}, avg dice loss: {1}".format(avgAcc, avgLoss))

    # interface
    def test(self):
        dataQueue = multiprocessing.Queue(self.queueSize)
        scanProcess = multiprocessing.Process(target = self.dataProcessor, args = (dataQueue, ))
        scanProcess.daemon = True
        scanProcess.start()

        caffe.set_device(0)
        caffe.set_mode_gpu()

        net =caffe.Net(self.netPath + "test.prototxt", os.path.join(self.netPath + "snapshot/", self.snapshot), caffe.TEST)
        self.testProcess(dataQueue, net, self.serializer)

if __name__ == "__main__":
    tester = Test("d:/project/tianchi/data/", "v1/")
    tester.test()
