# -*- coding:utf-8 -*-

import os
from glob import glob
import numpy as np
from tqdm import tqdm

from NoduleCropper import NoduleCropper
from NoduleSerializer import NoduleSerializer

# scan test CTs by 32 stepsize, generate data queue
# suppose CTs have already been re-sampled

class Scanner(object):
    def __init__(self, dataPath, dataQueue, stepSize = 32, cropSize = 64):
        self.dataPath = dataPath
        self.dataQueue = dataQueue
        self.stepSize = stepSize
        self.cropSize = cropSize

        self.serializer = NoduleSerializer(self.dataPath)
        self.cropper = NoduleCropper(self.dataPath)

    # helper
    def scanSingleFile(self, filename):
        seriesuid = os.path.basename(filename).split(".")[0]
        image = self.serializer.readFromNpy(filename)

        shape = image.shape
        steps = np.rint(np.ceil(shape / self.stepSize))
        steps = np.array(steps, dtype = int)

        for z in range(steps[0]):
            for y in range(steps[1]):
                for x in range(steps[2]):
                    center = np.array([(z + 1) * self.stepSize, (y + 1) * self.stepSize, (x + 1) * self.stepSize])
                    crop = self.cropper.cropSingleNodule(image, center, np.array[0, 0, 0], np.array[1.0, 1.0, 1.0], self.cropSize)

                    data = {}
                    data["number"] = z * steps[1] * steps[2] + y * steps[2] + x
                    data["seriesuid"] = seriesuid
                    data["image"] = crop

                    # TODO : normalize data

                    self.dataQueue.put(data)

    # interface
    def scanAllFiles(self):
        self.fileList = glob(self.dataPath + "*.npy")
        for file in enumerate(tqdm(self.fileList)):
            self.scanSingleFile(file)