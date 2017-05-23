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

class Test(obj):
    # constructor
    def __init__(self, dataPath="./", volSize=64):
        self.dataPath = dataPath
        self.volSize = volSize

    # helper
    def loadSample(self, subPath, filename):
        serializer = NoduleSerializer.NoduleSerializer(self.dataPath)

        sample = {}
        image = serializer.readFromNpy(subPath + "nodules/", filename)
        sample["image"] = image

        return sample

    def loadAllSamples(self, subPath, filenames):
        samples = []
        for filename in filenames:
            sample = self.loadSample(subPath, filename)
            samples.append(sample)
        return samples

    # interface
    def test(self):
        self.loadAllSamples("test/")

if __name__ == "__main__":
    tester = Test("d:/project/tianchi/data/")
    tester.test()