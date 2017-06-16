# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")
import NoduleCropper
import NoduleSerializer

import os
from glob import glob
import multiprocessing

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib

import caffe

class ResnetDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ["data", "label"]

        params = eval(self.param_str)
        # check_params(params)

        self.batch_size = params["batch_size"]
        self.vol_size = params["vol_size"]
        self.batch_loader = BatchLoader(params)

        top[0].reshape(self.batch_size, 1, self.vol_size, self.vol_size, self.vol_size)
        top[1].reshape(self.batch_size, 1, 2)

    def forward(self, bottom, top):
        for i in range(self.batch_size):
            image, groundTruth = self.batch_loader.load()

            top[0].data[i, ...] = image
            top[1].data[i, ...] = groundTruth

    def reshape(self, bottom, top):
        pass

    def backward(self, bottom, top):
        pass

class BatchLoader(object):
    def __init__(self, params):
        self.batchSize = params["batch_size"]
        self.volSize = params["vol_size"]
        self.iterationCount = params["iter_count"]
        self.queueSize = params["queue_size"]
        self.shiftRatio = 0. # for resnet, we don't need to shift since it always checks nodule in the center
        self.rotateRatio = params["rotate_ratio"]
        self.histogramShiftRatio = params["histogram_shift_ratio"]

        self.dataPath = params["data_path"]
        self.netPath = params["net_path"]
        self.phase = params["phase"]
        self.phaseSubPath = self.phase + "/"

        self.dataQueue = multiprocessing.Queue(self.queueSize)

        if self.phase == "deploy":
            # scan
            pass
        else:
            # load all nodules and groundTruths
            dataProcess = multiprocessing.Process(target = self.dataProcessor, args = (self.dataQueue,))
            dataProcess.daemon = True
            dataProcess.start()

    # interface
    def load(self):
        [nodule, groundTruth] = self.dataQueue.get()
        return nodule, groundTruth

    # helper
    def loadSample(self, filename, type):
        serializer = NoduleSerializer.NoduleSerializer(self.dataPath, self.phase)

        sample = {}
        if type == "nodule":
            image = serializer.readFromNpy("nodules/", filename)
            # directly generate groundTruth
            groundTruth = np.array([0, 1])
            sample["image"] = image
            sample["groundTruth"] = groundTruth
        else:
            image = serializer.readFromNpy("false-positive/", filename)
            # directly generate groundTruth
            groundTruth = np.array([1, 0])
            sample["image"] = image
            sample["groundTruth"] = groundTruth

        return sample

    def loadAllSamples(self, filenames, type):
        samples = []
        for filename in enumerate(tqdm(filenames)):
            filename = filename[1]
            sample = self.loadSample(filename, type)
            samples.append(sample)
        return samples

    def setWindow(self, image, upperBound=400.0, lowerBound=-1000.0):
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

    def randomizedHistogramShift(self, sample, shiftRatio):
        image = sample["image"]

        if np.random.random() < shiftRatio:
            # shift +- 5%
            shiftPercent = (np.random.random() - 0.5) / 10.0
            image = image * (1.0 + shiftPercent)
            sample["image"] = image

        return sample

    def randomizedCrop(self, sample, rotateRatio, shiftRatio):
        image = sample["image"]

        if np.random.random() < rotateRatio:
            # rotate - p(3, 3) - 1 possibles
            rotateList = [[1, 0, 2],
                          [1, 2, 0],
                          [2, 0, 1],
                          [2, 1, 0],
                          [0, 2, 1]]
            dir = np.random.randint(0, 4)
            rotate = rotateList[dir]
            image = np.transpose(image, rotate)
            # no need to rotate labels
            # groundTruth = np.transpose(groundTruth, rotate)

        # no need to shift
        centerRange = np.array([32, 96])
        xRange = centerRange
        yRange = centerRange
        zRange = centerRange

        crop = {}
        crop["image"] = image[zRange[0]:zRange[1], yRange[0]:yRange[1], xRange[0]:xRange[1]]
        crop["groundTruth"] = sample["groundTruth"]
        return crop

    def dataProcessor(self, dataQueue):
        noduleFileList = glob(self.dataPath + self.phaseSubPath + "nodules/*.npy")
        noduleFileList = map(lambda filePath: os.path.basename(filePath), noduleFileList)

        fpFileList = glob(self.dataPath + self.phaseSubPath + "false-positive/*.npy")
        fpFileList = map(lambda  filePath: os.path.basename(filePath), fpFileList)

        # load all samples
        samples = self.loadAllSamples(noduleFileList, "nodule")
        fpSamples = self.loadAllSamples(fpFileList, "false-positive")
        samples = samples.extend(fpSamples)

        for sample in enumerate(tqdm(samples)):
            sample = sample[1]
            image = sample["image"]
            # image = self.setWindow(image)
            image = self.normalize(image)
            sample["image"] = image

        # crop on the fly since we want randomized input
        np.random.seed()
        for i in range(self.iterationCount):
            for j in range(self.batchSize):
                # get random sample
                noduleIndex = np.random.randint(0, len(samples) - 1)
                sample = samples[noduleIndex]

                # randomized cropping
                if self.phase == "train":
                    sample = self.randomizedCrop(sample, self.rotateRatio, self.shiftRatio)
                    sample = self.randomizedHistogramShift(sample, self.histogramShiftRatio)

                dataQueue.put(tuple((sample["image"], sample["groundTruth"])))