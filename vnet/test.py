# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")
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
    def randomizedCrop(self, sample, rotateRatio, shiftRatio):
        image = sample["image"]

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
        return crop

    def normalizeSample(self, sample):
        image = sample["image"]
        mean = np.mean(image)
        std = np.std(image)
        print("sample: {0}, mean: {1}, std: {2}".format(sample["filename"], mean, std))

        image = image.astype(np.float32)
        image -= mean.astype(np.float32)
        image /= std.astype(np.float32)
        sample["image"] = image

    def normalizeAllSamples(self, samples):
        for sample in samples:
            self.normalizeSample(sample)

    def loadSample(self, subPath, filename):
        serializer = NoduleSerializer.NoduleSerializer(self.dataPath)

        sample = {}
        image = serializer.readFromNpy(subPath + "nodules/", filename)
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
        # TODO : load all test samples
        sampleFileList = glob(self.dataPath + "npy/nodules/*.npy")
        sampleFileList = map(lambda filePath: os.path.basename(filePath), sampleFileList)

        samples = self.loadAllSamples("npy/", sampleFileList)
        self.normalizeAllSamples(samples)

        caffe.set_device(0)
        caffe.set_mode_gpu()

        net =caffe.Net("test.prototxt", os.path.join("./snapshot/", "_iter_37000.caffemodel"), caffe.TEST)
        results = dict()

        serializer = NoduleSerializer.NoduleSerializer(self.dataPath + "test/")

        for i in range(len(samples)):
            sample = samples[i]
            crop = self.randomizedCrop(sample, 0, 0)

            net.blobs['data'].data[0, 0, :, :, :] = crop["image"]

            out = net.forward()

            labels = out["reshape_label"]

            # export as vnet output
            labelMap = np.squeeze(np.argmax(labels, axis = 1))
            serializer.writeToNpy("labels/", sample["filename"], labelMap.astype(np.int8))

            # export for resnext
            serializer.writeToNpy("results/", sample["filename"], np.squeeze(labels.astype(np.float32)))

if __name__ == "__main__":
    tester = Test("d:/project/tianchi/data/")
    tester.test()