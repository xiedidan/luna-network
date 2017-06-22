# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

import caffe

import os
import numpy as np

class Prophet(object):
    # constructor
    def __init__(self, dataPath, netPath, snapshot, dataQueue, dataHandler, dataHandlerParameter, phase = "deploy", volSize = 64, batchSize = 4):
        self.dataPath = dataPath
        self.netPath = netPath
        self.dataQueue = dataQueue
        self.phase = phase
        self.phaseSubPath = self.phase + "/"
        self.snapshot = snapshot
        self.dataHandler = dataHandler
        self.dataHandlerParameter = dataHandlerParameter
        self.volSize = volSize
        self.batchSize = batchSize

    # helper
    def testProcess(self, dataQueue, net):
        while (True):
            input = {}
            for i in range(self.batchSize):
                crop = dataQueue.get()
                input[i] = crop
                net.blobs["data"].data[i, 0, :, :, :] = crop["image"].astype(dtype = np.float32)

            out = net.forward()
            # call registered data handler
            self.dataHandler(input, out, self.dataHandlerParameter)

    # interface
    def speak(self):
        caffe.set_device(0)
        caffe.set_mode_gpu()

        net = caffe.Net(self.netPath + "deploy.prototxt", os.path.join(self.netPath, "snapshot/", self.snapshot), caffe.TEST)
        self.testProcess(self.dataQueue, net)
