# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

from NoduleSerializer import NoduleSerializer

import os
import glob
import numpy as np

from tqdm import tqdm

class Concat(object):
    def __init__(self, dataPath, phase = "deploy", stepSize = 32, volSize = 64):
        self.dataPath = dataPath
        self.phase = phase
        self.phaseSubPath = phase + "/"
        self.stepSize = stepSize
        self.volSize = volSize
        self.serializer = NoduleSerializer(dataPath, phase)

    # helper
    def concatSingleFile(self, seriesuid, steps, shape):
        image = np.zeros(shape)
        for z in range(steps[0]):
            for y in range(steps[1]):
                for x in range(steps[2]):
                    # read result
                    number = z * steps[1] * steps[2] + y * steps[2] + x
                    result = self.serializer.readFromNpy("results/", "{0}-{1}.npy".format(seriesuid, number))
                    offset = np.array([z * self.stepSize, y * self.stepSize, x * self.stepSize])

                    for bz in range(self.volSize):
                        for by in range(self.volSize):
                            for bx in range(self.volSize):
                                coord = [bz, by, bx] + offset
                                if (coord[0] < shape[0]) and (coord[1] < shape[1]) and (coord[2] < shape[2]):
                                    image[coord[0], coord[1], coord[2]] += result[bz, by, bx]
        return image

    # interface
    def concatAllFile(self):
        # read file list from nodules/
        metaPath = self.dataPath + self.phaseSubPath + "meta/"
        metaFileList = glob(metaPath)

        for metaFile in enumerate(tqdm(metaFileList)):
            metaFilename = metaFile[1]
            meta = self.serializer.readFromNpy(metaFilename)
            steps = meta["steps"]
            shape = meta["shape"]
            seriesuid = os.path.basename(metaFilename).split(".")[0]

            image = self.concatSingleFile(seriesuid, steps, shape)
            self.serializer.writeToNpy("concat/", "{0}.npy".format(seriesuid), image)
