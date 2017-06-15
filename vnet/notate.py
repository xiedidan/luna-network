# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

import os
from glob import glob
import numpy as np
from math import sqrt
from functools import reduce
from skimage import feature, exposure
import SimpleITK as sitk
from tqdm import tqdm

from NoduleSerializer import NoduleSerializer
import blobs_detection
import plot

# get nodule annotation from the output of vnet

class Notate(object):
    # constuctor
    def __init__(self, dataPath, phase = "deploy", stepSize = 32):
        self.dataPath = dataPath
        self.phase = phase # test or deploy
        self.phaseSubPath = self.phase + "/"
        self.stepSize = stepSize
        self.serializer = NoduleSerializer(self.dataPath, self.phase)
        self.sqrt3 = sqrt(3.0)

    # helper
    def calcWorldCoord(self, notations, origin):
        position = np.array([notations[0], notations[1], notations[2]]) # z, y, x
        worldCoord = origin + position

        notations[0] = worldCoord[0]
        notations[1] = worldCoord[1]
        notations[2] = worldCoord[2]
        return notations

    def calcRadius(self, blob):
        # blob is 4D ndarray for 3D image
        notation = blob
        notation[3] = self.sqrt3 * notation[3]

        return notation

    # interface
    def notate(self):
        csvPath = self.dataPath + self.phaseSubPath + "csv/"
        if not os.path.isdir(csvPath):
            os.makedirs(csvPath)

        with open(csvPath + "annotation.csv", "w") as file:
            file.write("seriesuid,coordX,coordY,coordZ,diameter_mm\n")

        with open(csvPath + "seriesuids.csv", "w") as file:
            file.write("seriesuid\n")

        pathList = glob(self.dataPath + self.phaseSubPath + "concat/*.npy")
        for path in enumerate(tqdm(pathList)):
            filename = os.path.basename(path[1])
            seriesuid = filename.split(".")[0]
            print(seriesuid)

            data = self.serializer.readFromNpy("concat/", filename)
            data = np.squeeze(data)

            # notation filter with lung mask
            mask = self.serializer.readFromNpy("mask/", filename)
            data = data * mask

            # blob dectection
            data = exposure.equalize_hist(data)
            blobs = blobs_detection.blob_dog(data, threshold=0.3, min_sigma=self.sqrt3)
            # blobs = feature.blob_dog(data, threshold = 0.3)
            # blobs = feature.blob_log(data)
            # blobs = feature.blob_doh(data)

            # get radius of blobs - now we have notation
            notations = np.zeros(blobs.shape)
            for i in range(len(blobs)):
                notations[i] = self.calcRadius(blobs[i])

            print(notations)
            print(len(notations))

            # convert to world coord
            mhdFile = os.path.join(self.dataPath, self.phaseSubPath, "raw/", seriesuid + ".mhd")
            rawImage = sitk.ReadImage(mhdFile)
            worldOrigin = np.array(rawImage.GetOrigin())[::-1]
            for i in range(len(notations)):
                notations[i] = self.calcWorldCoord(notations[i], worldOrigin)

            # write notations to csv
            csvPath = self.dataPath + self.phaseSubPath + "csv/"
            if not os.path.isdir(csvPath):
                os.makedirs(csvPath)

            with open(csvPath + "annotation.csv", "w+") as file:
                file.write("seriesuid,coordX,coordY,coordZ,diameter_mm\n")
                for i in range(len(notations)):
                    line = "{0},{1},{2},{3},{4}\n".format(seriesuid, notations[i][2], notations[i][1], notations[i][0], notations[i][3] * 2.0)
                    file.write(line)

            with open(csvPath + "seriesuids.csv", "w+") as file:
                file.write("seriesuid")
                for i in range(len(notations)):
                    file.write(seriesuid)

if __name__ == "__main__":
    notater = Notate("d:/project/tianchi/data/experiment/", "deploy")
    notater.notate()