# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

import os
import glob
import numpy as np
from math import sqrt
from functools import reduce
from skimage import feature, exposure
import SimpleITK as sitk
from tqdm import tqdm

import NoduleSerializer

# get nodule annotation from the output of vnet

class Notate(object):
    # constuctor
    def __init__(self, dataPath, phrase = "deploy", stepSize = 32):
        self.dataPath = dataPath
        self.phrase = phrase # test or deploy
        self.phraseSubPath = self.phrase + "/"
        self.stepSize = stepSize
        self.serializer = NoduleSerializer(self.dataPath)
        self.sqrt3 = sqrt(3.0)

    # helper
    def calcWorldCoord(self, notations, origin, steps):
        blockPosition = np.array([notations[0], notations[1], notations[2]]) # z, y, x

        blockOrigin = steps * self.stepSize
        position = blockOrigin + blockPosition

        worldCoord = origin + position

        notations[0] = worldCoord[0]
        notations[1] = worldCoord[1]
        notations[2] = worldCoord[2]
        return notations

    def calcRadius(self, blob):
        # blob is 4D ndarray for 3D image
        notation = blob
        notation[3] = self.sqrt3 * notation[3]

    def filterNotation(self, notation, lungMask):
        return lungMask[notation[0], notation[1], notation[2]] > 0.5

    # interface
    def notate(self):
        pathList = glob(self.dataPath)
        fileList = list(pathList.map(lambda path: os.path.basename(path)))
        for filename in enumerate(tqdm(fileList)):
            id = filename.split(".")[0]
            seriesuid = id.split("-")[0]
            # number = int(id.split("-")[1])

            data = self.serializer.readFromNpy("results/", filename)
            data = np.squeeze(data)

            # blob dectection
            # TODO : try different equalize and blob detect methods
            data = exposure.equalize_hist(data)
            blobs = feature.blob_dog(data, threshold = 0.3)

            # get radius of blobs - now we have notation
            notations = list(blobs.map(lambda blob: self.calcRadius(blob)))

            # convert to world coord
            steps = self.serializer.readFromNpy(self.phrase + "/steps/", filename)

            mhdFile = os.path.join(self.dataPath, self.phraseSubPath, seriesuid, ".mhd")
            rawImage = sitk.ReadImage(mhdFile)
            worldOrigin = np.array(rawImage.GetOrigin())[::-1]

            notations = list(notations.map(lambda notations: self.calcWorldCoord(notations, worldOrigin, steps)))

            # notation filter with lung mask
            mask = self.serializer.readFromNpy("mask/", filename)
            notations = list(notations.filter(lambda notation: self.filterNotation(notation, mask)))

            # TODO : write notations to annotation.csv
            line = "{0},{1},{2},{3},{4}\n".format(seriesuid, notations[2], notations[1], notations[0], notations[3] * 2.0)
            print(line)