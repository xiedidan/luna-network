# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")
from NoduleCropper import NoduleCropper
from NoduleSerializer import NoduleSerializer

import os
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

class FalsePositiveCreator(object):
    def __init__(self, dataPath, phase, volSize):
        self.dataPath = dataPath
        self.phase = phase
        self.phaseSubPath = self.phase + "/"
        self.volSize = volSize

        self.cropper = NoduleCropper(self.dataPath, self.phase, cropSize = self.volSize)
        self.serializer = NoduleSerializer(self.dataPath, self.phase)

        self.resampleSubPath = "resample/"
        self.resamplePath = self.dataPath + self.phaseSubPath + self.resampleSubPath
        self.resampleFileList = glob(self.resampleSubPath + "*.npy")

        self.vnetNotationSubPath = "vnet-csv/"
        self.vnetNotationFile = self.dataPath + self.phaseSubPath + self.vnetNotationSubPath + "annotations.csv"
        self.vnetNotationDf = pd.read_csv(self.vnetNotationFile)
        self.vnetNotationDf["file"] = self.vnetNotationDf["seriesuid"].map(
            lambda seriesuid: self.cropper.getFileFromSeriesuid(self.resampleFileList, seriesuid))
        self.vnetNotationDf.dropna()

        self.labelNotationSubPath = "csv/"
        self.labelNotationFile = self.dataPath + self.phaseSubPath + self.labelNotationSubPath + "annotations.csv"
        self.labelNotationDf = pd.read_csv(self.labelNotationFile)
        self.labelNotationDf["file"] = self.labelNotationDf["seriesuid"].map(
            lambda  seriesuid: self.cropper.getFileFromSeriesuid(self.resampleFileList, seriesuid))
        self.labelNotationDf.dropna()

    # helper
    def checkInside(self, vnetCenter, labelCenter, labelDiameter):
        distance = np.linalg.norm(vnetCenter - labelCenter)
        if distance > labelDiameter / 2.0:
            return False
        return True

    def creatorProcessor(self, resampleFile):
        # read resampled image
        resampleFilename = os.path.basename(resampleFile)
        image = self.serializer.readFromNpy("resample/", resampleFilename)

        # read world origin from mhd file
        seriesuid = resampleFilename.split(".")[0]
        rawFilename = self.dataPath + self.phaseSubPath + "raw/" + seriesuid + ".mhd"
        rawImage = sitk.ReadImage(rawFilename)
        worldOrigin = np.array(rawImage.GetOrigin())[::-1]

        fileVnetNotations = self.vnetNotationDf[self.vnetNotationDf.file == resampleFilename]
        fileLabelNotations = self.labelNotationDf[self.labelNotationDf.file == resampleFilename]
        idx = 0
        for i, vnetNotation in fileVnetNotations:
            falsePositiveFlag = True
            for j, labelNotation in fileLabelNotations:
                vnetCenter = np.array([vnetNotation.coordZ, vnetNotation.coordY, vnetNotation.coordX])
                labelCenter = np.array([labelNotation.coordZ, labelNotation.coordY, labelNotation.coordX])
                labelDiameter = labelNotation.diameter_mm
                if self.checkInside(vnetCenter, labelCenter, labelDiameter) == True:
                    falsePositiveFlag = False
                    break
            if falsePositiveFlag == True:
                idx += 1
                crop, minusFlag = self.cropper.cropSingleNodule(image, vnetCenter, worldOrigin, [1.0, 1.0, 1.0], self.volSize)
                self.serializer.writeToNpy("false-positive/", "{0}-{1}.npy".format(seriesuid, idx), crop)

        self.progressBar.update(1)
        return True

    # interface
    def create(self):
        self.progressBar = tqdm(total=len(self.resampleFileList))

        pool = ThreadPool()
        pool.map(self.creatorProcessor, self.resampleFileList)

        self.progressBar.close()

if __name__ == "__main__":
    creator = FalsePositiveCreator("d:/project/tianchi/data/", "test", 64)
    creator.create()
