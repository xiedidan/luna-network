# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

import os
import numpy as np
from glob import glob

import NoduleCropper
import NoduleSerializer
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
    def __init__(self, dataPath = "./", snapshot = "_iter_3700.caffemodel", queueSize = 32, volSize = 64):
        self.dataPath = dataPath
        self.snapshot = snapshot
        self.queueSize = queueSize
        self.volSize = volSize
        self.serializer = NoduleSerializer(self.dataPath)

    # helper
    def dataScanner(self, dataPath, dataQueue):
        scanner = Scanner(dataPath, dataQueue)
        scanner.scanAllFiles()

    def testProcess(self, dataQueue, net, serializer):
        while (True):
            crop = dataQueue.get()
            net.blobs['data'].data[0, 0, :, :, :] = crop["image"].astype(dtype = np.float32)

            out = net.forware()

            labels = out["reshape_label"]
            serializer.writeToNpy("results/", "{0}-{1}.npy".format(crop["seriesuid"], crop["number"]), np.squeeze(labels.astype(np.float32)))
            serializer.writeToNpy("steps/", "{0}-{1}.npy".format(crop["seriesuid"], crop["number"]), crop["steps"])

    # interface
    def test(self):
        dataQueue = multiprocessing.Queue(self.queueSize)
        scanProcess = multiprocessing.Process(target = self.dataScanner, args = (self.dataPath, dataQueue))
        scanProcess.daemon = True
        scanProcess.start()

        caffe.set_device(0)
        caffe.set_mode_gpu()

        net =caffe.Net("test.prototxt", os.path.join("./snapshot/", self.snapshot), caffe.TEST)
        self.testProcess(dataQueue, net, self.serializer)

if __name__ == "__main__":
    tester = Test("d:/project/tianchi/data/")
    tester.test()
