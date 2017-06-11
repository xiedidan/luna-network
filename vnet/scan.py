# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

import os
from glob import glob
import numpy as np
from tqdm import tqdm

from NoduleCropper import NoduleCropper
from NoduleSerializer import NoduleSerializer

# scan CTs by 32 stepsize, generate data queue
# suppose CTs have already been re-sampled

class Scanner(object):
    def __init__(self, dataPath, phrase, dataQueue, stepSize = 32, cropSize = 64):
        self.dataPath = dataPath
        self.phrase = phrase
        self.phraseSubPath = self.phrase + "/"
        self.dataQueue = dataQueue
        self.stepSize = stepSize
        self.cropSize = cropSize

        self.serializer = NoduleSerializer(self.dataPath)
        self.cropper = NoduleCropper(self.dataPath)

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
        image = self.serializer.readFromNpy(filename)

        shape = image.shape
        steps = np.rint(np.ceil(shape / self.stepSize))
        steps = np.array(steps, dtype = int)

        meta = {}
        meta["steps"] = steps
        meta["shape"] = shape
        self.serializer.writeToNpy("meta/", seriesuid + ".npy", meta)

        for z in range(steps[0]):
            for y in range(steps[1]):
                for x in range(steps[2]):
                    center = np.array([(z + 1) * self.stepSize, (y + 1) * self.stepSize, (x + 1) * self.stepSize])
                    crop = self.cropper.cropSingleNodule(image, center, np.array[0, 0, 0], np.array[1.0, 1.0, 1.0], self.cropSize)

                    crop = self.setWindow(crop)
                    crop = self.normalize(crop)

                    data = {}
                    data["number"] = z * steps[1] * steps[2] + y * steps[2] + x
                    data["steps"] = np.array([z, y, x])
                    data["seriesuid"] = seriesuid
                    data["image"] = crop

                    self.dataQueue.put(data)

    # interface
    def scanAllFiles(self):
        self.fileList = glob(self.dataPath + self.phraseSubPath + "nodules/*.npy")
        for file in enumerate(tqdm(self.fileList)):
            self.scanSingleFile(file[1])
