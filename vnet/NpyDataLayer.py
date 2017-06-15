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

class NpyDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ["data", "label"]

        params = eval(self.param_str)
        # check_params(params)

        self.batch_size = params["batch_size"]
        self.vol_size = params["vol_size"]
        self.batch_loader = BatchLoader(params)

        top[0].reshape(self.batch_size, 1, self.vol_size, self.vol_size, self.vol_size)
        top[1].reshape(self.batch_size, 1, self.vol_size, self.vol_size, self.vol_size)

        # print_info("NpyDataLayer", params)

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
        self.shiftRatio = params["shift_ratio"]
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
    def loadSample(self, filename):
        serializer = NoduleSerializer.NoduleSerializer(self.dataPath, self.phase)

        sample = {}
        image = serializer.readFromNpy("nodules/", filename)
        groundTruth = serializer.readFromNpy("groundTruths/", filename)
        sample["image"] = image
        sample["groundTruth"] = groundTruth

        return sample

    def loadAllSamples(self, filenames):
        samples = []
        for filename in enumerate(tqdm(filenames)):
            filename = filename[1]
            sample = self.loadSample(filename)
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
        groundTruth = sample["groundTruth"]

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
            groundTruth = np.transpose(groundTruth, rotate)

        centerRange = np.array([32, 96])
        if np.random.random() < shiftRatio:
            # shift - we shift +-24 max along each axis
            shiftx = np.random.randint(-24, 24)
            shifty = np.random.randint(-24, 24)
            shiftz = np.random.randint(-24, 24)
            xRange = centerRange + np.array([shiftx, shiftx])
            yRange = centerRange + np.array([shifty, shifty])
            zRange = centerRange + np.array([shiftz, shiftz])
        else:
            xRange = centerRange
            yRange = centerRange
            zRange = centerRange

        crop = {}
        crop["image"] = image[zRange[0]:zRange[1], yRange[0]:yRange[1], xRange[0]:xRange[1]]
        crop["groundTruth"] = groundTruth[zRange[0]:zRange[1], yRange[0]:yRange[1], xRange[0]:xRange[1]]
        return crop

    def dataProcessor(self, dataQueue):
        npyFileList = glob(self.dataPath + self.phaseSubPath + "nodules/*.npy")
        npyFileList = map(lambda filePath: os.path.basename(filePath), npyFileList)

        # load all samples
        samples = self.loadAllSamples(npyFileList)

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
                # sample = samples[0]

                # randomized cropping
                if self.phase == "train":
                    sample = self.randomizedCrop(sample, self.rotateRatio, self.shiftRatio)
                    sample = self.randomizedHistogramShift(sample, self.histogramShiftRatio)

                dataQueue.put(tuple((sample["image"], sample["groundTruth"])))
                        # print(dataQueue.qsize())