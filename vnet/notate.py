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
import multiprocessing
import time

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
    def calcWorldCoord(self, notation, origin):
        position = np.array([notation[0], notation[1], notation[2]]) # z, y, x
        worldCoord = origin + position

        print("calcWorldCoord", position, origin, worldCoord)

        notation[0] = worldCoord[0]
        notation[1] = worldCoord[1]
        notation[2] = worldCoord[2]
        return notation

    def calcRadius(self, blob):
        # blob is 4D ndarray for 3D image
        notation = blob
        notation[3] = self.sqrt3 * notation[3]

        return notation

    def calcOffset(self, notation, position, steps):
        ratio = 2
        point = np.array([notation[0], notation[1], notation[2]])
        offset = ((steps // ratio - 2) * self.stepSize) * position
        point = point + offset

        print("calcOffset", notation, position, offset, point)

        notation[0] = point[0]
        notation[1] = point[1]
        notation[2] = point[2]
        return notation

    # interface
    def notateProcessor(self, notateQueue):
        while True:
            # get crop from queue
            crop = notateQueue.get()

            # blobs detection
            image = exposure.equalize_hist(crop["image"])
            print("notateProcessor", image.shape)
            blobs = blobs_detection.blob_dog(image, threshold=0.3)

            # get radius of blobs - now we have notation
            notationList = np.zeros(blobs.shape)
            for i in range(len(blobs)):
                notationList[i] = self.calcRadius(blobs[i])

            # get prob - now we have notation
            notations = []
            for i in range(len(notationList)):
                probCube = crop["image"]
                note = notationList[i]
                z = int(note[0])
                y = int(note[1])
                x = int(note[2])
                prob = probCube[z][y][x] / 2
                r = note[3]
                notation = np.array([z, y, x, r, prob])
                notations.append(notation)

            # get world coords
            position = crop["position"]
            metaFilename = crop["seriesuid"] + ".npy"

            meta = self.serializer.readFromNpy("meta/", metaFilename)
            worldCoord = meta["worldOrigin"]
            steps = meta["steps"]

            for i in range(len(notations)):
                notations[i] = self.calcOffset(notations[i], position, steps)
                notations[i] = self.calcWorldCoord(notations[i], worldCoord)

            # write to disk
            print("notateProcessor", crop["seriesuid"], crop["number"])
            self.serializer.writeToNpy("notates/", "{0}-{1}.npy".format(crop["seriesuid"], crop["number"]), notations)


    def startNotate(self):
        # start notate process
        processCount = 5
        queueSize = 1

        notateQueue = multiprocessing.Queue(queueSize)
        notateProcessList = []
        for i in range(processCount):
            notateProcess = multiprocessing.Process(target = self.notateProcessor, args=(notateQueue, ))
            notateProcess.daemon = True
            notateProcess.start()
            notateProcessList.append(notateProcess)

        # get file list from concat/
        concatList = glob(self.dataPath + self.phaseSubPath + "concat/*.npy")
        for concatFile in enumerate(tqdm(concatList)):
            filename = os.path.basename(concatFile[1])
            str = filename.split(".")[0]
            seriesuid = str.split("-")[0] + "-" + str.split("-")[1]
            number = str.split("-")[2]

            crop = self.serializer.readFromNpy("concat/", filename)
            crop["seriesuid"] = seriesuid
            crop["number"] = number
            notateQueue.put(crop)
            if concatFile[0] < processCount:
                time.sleep(5)

    def reduceNotate(self):
        # create csv
        csvPath = self.dataPath + self.phaseSubPath + "vnet-csv/"
        if not os.path.isdir(csvPath):
            os.makedirs(csvPath)

        with open(csvPath + "annotation.csv", "w") as file:
            file.write("seriesuid,coordX,coordY,coordZ,probability\n")

        with open(csvPath + "seriesuids.csv", "w") as file:
            file.write("seriesuid\n")

        # get file list from mask/
        maskList = glob(self.dataPath + self.phaseSubPath + "mask/*.npy")
        for maskFile in enumerate(tqdm(maskList)):
            filename = os.path.basename(maskFile[1])
            seriesuid = filename.split(".")[0]

            #read all notations of a single file
            fileNotations = []
            cropList = glob(self.dataPath + self.phaseSubPath + "notates/{0}-*.npy".format(seriesuid))
            for cropFile in cropList:
                cropFilename = os.path.basename(cropFile)
                cropNotations = self.serializer.readFromNpy("notates/", cropFilename)
                fileNotations.extend(cropNotations)

            # filter notations with mask
            notations = []
            mask = self.serializer.readFromNpy("mask/", filename)
            meta = self.serializer.readFromNpy("meta/", filename)
            worldOrigin = meta["worldOrigin"]
            for notation in fileNotations:
                position = np.array([notation[0], notation[1], notation[2]])
                position -= worldOrigin
                # print(mask.shape, notation, position, worldOrigin)
                if (position[0] < mask.shape[0]) & (position[1] < mask.shape[1]) & (position[2] < mask.shape[2]):
                    if mask[int(position[0]), int(position[1]), int(position[2])] > 0.5:
                        notations.append(notation)

            # prob filter
            notations = []
            for notation in fileNotations:
                if notation[4] >= 0.000001:
                    if notation[4] > 1:
                        # print("before", notation[4])
                        log2 = np.log2(np.ceil(notation[4]))
                        notation[4] = notation[4] / (2 ** log2)
                        # print("after", notation[4])
                    print(notation)
                    if notation[4] > 0.7:
                        notations.append(notation)


            with open(csvPath + "annotation.csv", "a") as file:
                for i in range(len(notations)):
                    line = "{0},{1},{2},{3},{4}\n".format(seriesuid, notations[i][2], notations[i][1],
                                                          notations[i][0], notations[i][4])
                    file.write(line)

    def notate(self):
        csvPath = self.dataPath + self.phaseSubPath + "vnet-csv/"
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
            # print(seriesuid)

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

            # print(notations)
            # print(len(notations))

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
    notater = Notate("d:/project/tianchi/data/", "deploy")
    # notater.startNotate()
    notater.reduceNotate()

    while True:
        time.sleep(1)