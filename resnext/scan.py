# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

import os
from glob import glob
import numpy as np
from tqdm import tqdm

from NoduleCropper import NoduleCropper
from NoduleSerializer import NoduleSerializer

import matplotlib.pyplot as plt
import matplotlib

# TODO : crop CTs with the direction of annotations

class Scanner(object):
    def __init__(self, dataPath, phase, dataQueue, stepSize = 32, cropSize = 64):
        self.dataPath = dataPath
        self.phase = phase
        self.phaseSubPath = self.phase + "/"
        self.dataQueue = dataQueue
        self.stepSize = stepSize
        self.cropSize = cropSize

        self.serializer = NoduleSerializer(self.dataPath, self.phase)
        self.cropper = NoduleCropper(self.dataPath, self.phase)

    # helper
    def setWindow(self, image, upperBound = 400.0, lowerBound = -1000.0):
        image[image > upperBound] = upperBound
        image[image < lowerBound] = lowerBound
        return image

    def normalize(self, image):
        mean = np.mean(image)
        std = np.std(image)

        image = image.astype(np.float32)
        image -= mean.astype(np.float32)
        image /= std.astype(np.float32)

        return image

    def scanSingleFile(self, filename):
        seriesuid = os.path.basename(filename).split(".")[0]
        image = self.serializer.readFromNpy("nodules/", filename)

        shape = image.shape
        shape = np.array(shape, dtype = int)
        steps = np.rint(np.ceil(shape / self.stepSize))
        steps = np.array(steps, dtype = int)

        meta = {}
        meta["steps"] = steps
        meta["shape"] = shape
        # print("scanSingleFile seriesuid: {0}, shape: {1}, steps: {2}".format(seriesuid, shape, steps))
        self.serializer.writeToNpy("meta/", seriesuid + ".npy", meta)

        for z in range(steps[0]):
            for y in range(steps[1]):
                for x in range(steps[2]):
                    center = {"coordZ": (z + 1) * self.stepSize, "coordY": (y + 1) * self.stepSize, "coordX": (x + 1) * self.stepSize}
                    crop, minusFlag = self.cropper.cropSingleNodule(image, center, np.array([0, 0, 0]), np.array([1.0, 1.0, 1.0]), self.cropSize)

                    # crop = self.setWindow(crop)
                    crop = self.normalize(crop)

                    data = {}
                    data["number"] = z * steps[1] * steps[2] + y * steps[2] + x
                    data["steps"] = np.array([z, y, x])
                    data["seriesuid"] = seriesuid
                    data["image"] = crop

                    self.dataQueue.put(data)

    # interface
    def scanAllFiles(self):
        self.fileList = glob(self.dataPath + self.phaseSubPath + "nodules/*.npy")
        for file in enumerate(tqdm(self.fileList)):
            self.scanSingleFile(os.path.basename(file[1]))
