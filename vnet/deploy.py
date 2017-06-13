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

class Depoly(object):
    # constructor
    def __init__(self, dataPath = "./", netPath = "v1/", phase = "deploy", snapshot = "_iter_62000.caffemodel", queueSize = 32, volSize = 64):
        self.dataPath = dataPath
        self.netPath = netPath
        self.phase = phase
        self.phaseSubPath = self.phase + "/"
        self.snapshot = snapshot
        self.queueSize = queueSize
        self.volSize = volSize
        self.serializer = NoduleSerializer(self.dataPath, self.phase)

    # helper
    def dataScanner(self, dataPath, phase, dataQueue):
        scanner = Scanner(dataPath, phase, dataQueue)
        scanner.scanAllFiles()

    def testProcess(self, dataQueue, net, serializer):
        while (True):
            crop = dataQueue.get()
            # print("testProcess seriesuid: {0}, number: {1}".format(crop["seriesuid"], crop["number"]))
            net.blobs['data'].data[0, 0, :, :, :] = crop["image"].astype(dtype = np.float32)

            out = net.forward()

            labels = out["argmax_output"]
            labels = np.rint(labels)
            labels = np.squeeze(labels.astype(dtype=np.int8))
            fig = plt.figure()
            ax = fig.add_subplot(221)
            ax.imshow(labels[32])
            plt.show()
            serializer.writeToNpy("results/", "{0}-{1}.npy".format(crop["seriesuid"], crop["number"]), np.squeeze(labels))
            # serializer.writeToNpy("steps/", "{0}-{1}.npy".format(crop["seriesuid"], crop["number"]), crop["steps"])

    # interface
    def speak(self):
        dataQueue = multiprocessing.Queue(self.queueSize)

        scanProcess = multiprocessing.Process(target = self.dataScanner, args = (self.dataPath, self.phase, dataQueue))
        scanProcess.daemon = True
        scanProcess.start()

        caffe.set_device(0)
        caffe.set_mode_gpu()

        net =caffe.Net(self.netPath + "deploy.prototxt", os.path.join(self.netPath, "./snapshot/", self.snapshot), caffe.TEST)
        self.testProcess(dataQueue, net, self.serializer)

if __name__ == "__main__":
    prophet = Depoly("d:/project/tianchi/data/", netPath = "v1/", snapshot = "_iter_62000.caffemodel")
    prophet.speak()
