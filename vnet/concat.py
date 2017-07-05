# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

from NoduleSerializer import NoduleSerializer

import os
from glob import glob
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
        image = np.zeros(((steps[0] + 1) * self.stepSize, (steps[1] + 1) * self.stepSize, (steps[2] + 1) * self.stepSize)).astype(np.float32)
        for z in range(steps[0]):
            for y in range(steps[1]):
                for x in range(steps[2]):
                    # read result
                    number = z * steps[1] * steps[2] + y * steps[2] + x
                    result = self.serializer.readFromNpy("results/", "{0}-{1}.npy".format(seriesuid, number))
                    result = result.astype(np.float32)
                    offset = np.array([z * self.stepSize, y * self.stepSize, x * self.stepSize])
                    # print("z: {0}, y: {1}, x: {2}".format(z, y, x))
                    image[offset[0]:(offset[0] + self.volSize), offset[1]:(offset[1] + self.volSize), offset[2]:(offset[2] + self.volSize)] += result[:, :, :]

        crop = np.zeros(shape)
        crop[:, :, :] = image[0:shape[0], 0:shape[1], 0:shape[2]]
        return np.rint(crop.astype(np.float32) / 2.)

    def concatSingleFileCrop(self, seriesuid, steps):
        ratio = 2
        cropLength = ((steps[0] / ratio + 1) * self.stepSize, (steps[1] / ratio + 1) * self.stepSize, (steps[2] / ratio + 1) * self.stepSize)

        for zBlock in range(ratio):
            for yBlock in range(ratio):
                for xBlock in range(ratio):
                    image = np.zeros(cropLength)

                    stepRangeZ = np.array(
                        [zBlock * (steps[0] / ratio) - zBlock, (zBlock + 1) * (steps[0] / ratio) + (1 - zBlock)])
                    stepRangeY = np.array(
                        [yBlock * (steps[1] / ratio) - yBlock, (yBlock + 1) * (steps[1] / ratio) + (1 - yBlock)])
                    stepRangeX = np.array(
                        [xBlock * (steps[2] / ratio) - xBlock, (xBlock + 1) * (steps[2] / ratio) + (1 - xBlock)])

                    for z in range(stepRangeZ):
                        for y in range(stepRangeY):
                            for x in range(stepRangeX):
                                number = z * steps[2] * steps[1] + y * steps[2] + x
                                result = self.serializer.readFromNpy("results/", "{0}-{1}.npy".format(seriesuid, number))
                                result = result.astype(np.float32)
                                blockOffset = np.array([(z - zBlock * (steps[0] / ratio)) * self.stepSize, (y - yBlock * (steps[1] / ratio)) * self.stepSize, (x - xBlock * (steps[2] / ratio)) * self.stepSize])
                                image[blockOffset[0]:(blockOffset[0] + self.volSize), blockOffset[1]:(blockOffset[1]+ self.volSize), blockOffset[2]:(blockOffset[2] + self.volSize)] += result[:, :, :]

                    # write to disk
                    series = zBlock * ratio * ratio + yBlock * ratio + xBlock
                    crop = {}
                    crop["position"] = np.array([zBlock, yBlock, xBlock])
                    crop["image"] = image
                    self.serializer.writeToNpy("concat/", "{0}-{1}.npy".format(seriesuid, series), crop)

    # interface
    def concatAllFile(self):
        # read file list from nodules/
        metaPath = self.dataPath + self.phaseSubPath + "meta/*.npy"
        metaFileList = glob(metaPath)

        for metaFile in enumerate(tqdm(metaFileList)):
            metaFilename = metaFile[1]
            metaFile = os.path.basename(metaFilename)
            meta = self.serializer.readFromNpy("meta/", metaFile)
            steps = meta["steps"]
            shape = meta["shape"]
            seriesuid = os.path.basename(metaFilename).split(".")[0]

            image = self.concatSingleFile(seriesuid, steps, shape)
            self.serializer.writeToNpy("concat/", "{0}.npy".format(seriesuid), image.astype(np.int8))

if __name__ == "__main__":
    concator = Concat("d:/project/tianchi/data/", "deploy")
    concator.concatAllFile()