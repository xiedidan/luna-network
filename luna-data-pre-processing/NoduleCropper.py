# -*- coding:utf-8 -*-

import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import scipy.ndimage
import os
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from Logger import Logger
import sys
import Utility as util
from NoduleSerializer import NoduleSerializer

class NoduleCropper(object):
    # constructor
    def __init__(self, dataPath = "./", maxDataBound = 400.0, minDataBound = -1000.0, cropSize = 64, levelStr = "Info", logPath = "./log/", logName = "NoduleCropper"):
        # constant
        self.maxDataBound = maxDataBound
        self.minDataBound = minDataBound
        self.cropSize = cropSize

        # path
        self.dataPath = dataPath
        self.annotationCsvPath = os.path.join(self.dataPath, "csv/train/annotations.csv")
        self.annotationMhdPath = os.path.join(self.dataPath, "train/")

        # data
        self.annotationMhdFileList = glob(self.annotationMhdPath + "*.mhd")

        self.annotationDf = pd.read_csv(self.annotationCsvPath)
        self.annotationDf["file"] = self.annotationDf["seriesuid"].map(lambda seriesuid: self.getFileFromSeriesuid(self.annotationMhdFileList, seriesuid))
        self.annotationDf.dropna()

        # progress bar
        self.progressBar = tqdm(total = len(self.annotationMhdFileList))

        # logger
        self.logger = Logger(levelStr, logPath, logName)

    # helper
    def getBox(self, center, size):
        lowerBound = np.rint(np.floor(center - size / 2))
        upperBound = np.rint(np.ceil(center + size / 2))
        lowerBound = np.array(lowerBound, dtype=int)
        upperBound = np.array(upperBound, dtype=int)
        return lowerBound, upperBound

    def fillGroundTruthImage(self, image, nodule, worldOrigin, spacing):
        noduleWorldCenter = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])
        noduleCenter = np.rint((noduleWorldCenter - worldOrigin) / spacing)
        noduleCenter = np.array(noduleCenter, dtype = int)

        noduleCenter = np.absolute(noduleCenter)

        size = np.array([nodule.diameter_mm, nodule.diameter_mm, nodule.diameter_mm])
        lowerBound, upperBound = self.getBox(noduleCenter, size)
        # print("fillGroundTruthImage(): noduleWorldCenter: {0}, worldOrigin: {1}, lower: {2}, upper: {3}".format(noduleWorldCenter, worldOrigin, lowerBound, upperBound))

        radius = np.ceil(nodule.diameter_mm / 2)
        for z in range(lowerBound[0], upperBound[0] + 1):
            for y in range(lowerBound[1], upperBound[1] + 1):
                for x in range(lowerBound[2], upperBound[2] + 1):
                    point = np.array([z, y, x])
                    distance = np.linalg.norm(point - noduleCenter)
                    # print("point: {0}, noduleWorldCenter: {1}, dist: {2}, radius: {3}".format(point, noduleWorldCenter, distance, radius))
                    if distance < radius:
                        image[point[0], point[1], point[2]] = np.int16(1)

        return image

    def resample(self, image, oldSpacing, newSpacing = [1, 1, 1]):
        resizeFactor = oldSpacing / newSpacing
        newRealShape = image.shape * resizeFactor
        newShape = np.round(newRealShape)
        realResizeFactor = newShape / image.shape
        newSpacing = oldSpacing / realResizeFactor

        image = scipy.ndimage.interpolation.zoom(image, realResizeFactor, mode = "nearest")
        return image, newSpacing

    def normalization(self, image):
        dataRange = self.maxDataBound - self.minDataBound
        image = (image - self.minDataBound) / dataRange
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image

    def getFileFromSeriesuid(self, fileList, seriesuid):
        for file in fileList:
            if seriesuid in file:
                return file

    def cropSingleNodule(self, image, nodule, worldOrigin, spacing, size):
        noduleWorldCenter = np.array([nodule.coordZ, nodule.coordY, nodule.coordX])
        noduleCenter = np.rint((noduleWorldCenter - worldOrigin) / spacing)
        noduleCenter = np.array(noduleCenter, dtype = int)

        # print("worldOrigin: {0}, noduleWorldCenter: {1}, noduleCenter: {2}".format(worldOrigin, noduleWorldCenter, noduleCenter))
        minusFlag = False
        if any(noduleCenter < 0):
            minusFlag = True

        noduleCenter = np.absolute(noduleCenter)

        voxelSize = np.array([size, size, size])
        lowerBound, upperBound = self.getBox(noduleCenter, voxelSize)
        # print("cropSingleNodule(): noduleWorldCenter: {0}, worldOrigin: {1}, lower: {2}, upper: {3}".format(noduleWorldCenter, worldOrigin, lowerBound, upperBound))

        noduleCrop = np.zeros((size, size, size), dtype = np.int16)
        imageShape = image.shape
        cropLowerBound = [0, 0, 0]
        cropUpperBound = [size, size, size]
        for i in range(3):
            if lowerBound[i] < 0:
                cropLowerBound[i] = np.absolute(lowerBound[i])
                lowerBound[i] = 0
            if upperBound[i] > imageShape[i]:
                cropUpperBound[i] = cropUpperBound[i] - (upperBound[i] - imageShape[i])
                upperBound[i] = imageShape[i]

        # print("cropLowerBound: {0}, cropUpperBound: {1}, lowerBound: {2}, upperBound: {3}".format(cropLowerBound, cropUpperBound, lowerBound, upperBound))
        # print(noduleCrop[cropLowerBound[0]:cropUpperBound[0], cropLowerBound[1]:cropUpperBound[1], cropLowerBound[2]:cropUpperBound[2]].shape)
        # print(image[lowerBound[0]:upperBound[0], lowerBound[1]:upperBound[1], lowerBound[2]:upperBound[2]].shape)
        noduleCrop[cropLowerBound[0]:cropUpperBound[0], cropLowerBound[1]:cropUpperBound[1], cropLowerBound[2]:cropUpperBound[2]] = image[lowerBound[0]:upperBound[0], lowerBound[1]:upperBound[1], lowerBound[2]:upperBound[2]]
        return noduleCrop, minusFlag

    def cropFileNodule(self, file, size):
        fileNodules = self.annotationDf[self.annotationDf.file == file]
        rawImage = sitk.ReadImage(file)

        worldOrigin = np.array(rawImage.GetOrigin())[::-1]
        oldSpacing = np.array(rawImage.GetSpacing())[::-1]
        image, spacing = self.resample(sitk.GetArrayFromImage(rawImage), oldSpacing)

        image = np.rint(image)
        image = np.array(image, dtype = np.int16)

        # init ground truth image
        groundTruthImage = np.zeros(image.shape, dtype = np.int16)

        groundTruths = []
        nodules = []
        reverseFlag = False
        for idx, nodule in fileNodules.iterrows():
            # ground truth
            groundTruthImage = self.fillGroundTruthImage(groundTruthImage, nodule, worldOrigin, spacing)
            groundTruthCrop, minusFlag = self.cropSingleNodule(groundTruthImage, nodule, worldOrigin, spacing, size)
            groundTruths.append(groundTruthCrop)

            # nodule
            noduleCrop, minusFlag = self.cropSingleNodule(image, nodule, worldOrigin, spacing, size)
            nodules.append(noduleCrop)
            if minusFlag == True:
                reverseFlag = True

        if reverseFlag == True:
            self.logger.warning("cropFileNodule()", "Reversed directions detected in Mhd file: " + file + ".")

        return nodules, groundTruths

    def cropAllNoduleForMhdProcessor(self, file):
        if file not in self.annotationDf.file.values:
            # print("Mhd file: " + file + " is not found in annotations.csv.")
            self.logger.warning("cropAllNoduleForMhdProcessor()", "Mhd file: " + file + " is not found in annotations.csv.")
            self.progressBar.update(1)
            return None

        fileNodules = {}
        fileNodules["seriesuid"] = os.path.basename(file).split(".")[0]
        try:
            noduleCrop, groundTruthCrop = self.cropFileNodule(file, self.cropSize)
            fileNodules["nodules"] = noduleCrop
            fileNodules["groundTruths"] = groundTruthCrop
        except:
            print("cropAllNoduleForMhdProcessor() error: {0}".format(sys.exc_info()[0]))
            self.logger.error("cropAllNoduleForMhdProcessor()", sys.exc_info()[0])
            fileNodules = None
        finally:
            self.progressBar.update(1)
            return fileNodules

    def resampleAndCreateGroundTruthProcessor(self, filename):
        if filename not in self.annotationDf.file.values:
            print("Mhd file: " + filename + " is not found in annotations.csv.")
            self.progressBar.update(1)
            return None

        # load image
        rawImage = sitk.ReadImage(filename)
        worldOrigin = np.array(rawImage.GetOrigin())[::-1]
        oldSpacing = np.array(rawImage.GetSpacing())[::-1]

        #  resample image
        image, spacing = self.resample(sitk.GetArrayFromImage(rawImage), oldSpacing)
        image = np.rint(image)
        image = np.array(image, dtype=np.int16)

        # init ground truth image
        groundTruthImage = np.zeros(image.shape, dtype=np.int16)

        # fill groundTruth
        fileNodules = self.annotationDf[self.annotationDf.file == filename]
        for idx, nodule in fileNodules.iterrows():
            groundTruthImage = self.fillGroundTruthImage(groundTruthImage, nodule, worldOrigin, spacing)

        sample = {}
        sample["seriesuid"] = os.path.basename(filename).split(".")[0]
        sample["image"] = image
        sample["groundTruth"] = groundTruthImage
        # print("sample id: {0}, shape: {1}, spacing: {2}".format(sample["seriesuid"], sample["image"].shape, spacing))

        serializer = NoduleSerializer(self.dataPath)
        serializer.writeToNpy("npy/", sample["seriesuid"], sample["image"], sample["groundTruth"])

        self.progressBar.update(1)

    # interface
    def cropAllNodule(self):
        nodules = []
        for file in enumerate(tqdm(self.annotationMhdFileList)):
            filename = file[1]

            if filename not in self.annotationDf.file.values:
                print("Mhd file: " + filename + " is not found in annotations.csv.")
                continue

            nodules.append(self.cropFileNodule(filename, self.cropSize))
        return nodules

    def cropAllNoduleForMhd(self):
        pool = ThreadPool()
        nodules = pool.map(self.cropAllNoduleForMhdProcessor, self.annotationMhdFileList)
        if None in nodules:
            nodules.remove(None)
        return nodules

    def resampleAndCreateGroundTruth(self):
        pool = ThreadPool()
        pool.map(self.resampleAndCreateGroundTruthProcessor, self.annotationMhdFileList)

        self.progressBar.close()
    