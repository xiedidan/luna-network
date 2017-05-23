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

class Test(object):
    # constructor
    def __init__(self, dataPath="./", volSize=64):
        self.dataPath = dataPath
        self.volSize = volSize

    # helper
    def loadSample(self, subPath, filename):
        serializer = NoduleSerializer.NoduleSerializer(self.dataPath)

        sample = {}
        image = serializer.readFromNpy(subPath + "npy/", filename)
        sample["image"] = image
        sample["filename"] = os.path.basename(filename)

        return sample

    def loadAllSamples(self, subPath, filenames):
        samples = []
        for filename in filenames:
            sample = self.loadSample(subPath, filename)
            samples.append(sample)
        return samples

    # interface
    def test(self):
        samples = self.loadAllSamples("test/")

        caffe.set_device(0)
        caffe.set_mode_gpu()

        net =caffe.Net("test.prototxt", os.path.join("./snapshot/", "_iter_37000.caffemodel"), caffe.TEST)
        results = dict()

        serializer = NoduleSerializer(self.dataPath + "test/")

        for i in range(len(samples)):
            sample = samples[i]
            net.blobs['data'].data[0, 0, :, :, :] = sample["image"]

            out = net.forward()
            labels = out["reshape_label"]
            labelMap = np.squeeze(np.argmax(labels, axis = 1))

            serializer.writeToNpy("results/", sample["filename"], labelMap)

if __name__ == "__main__":
    tester = Test("d:/project/tianchi/data/")
    tester.test()