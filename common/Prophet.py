# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

import caffe

import os
import numpy as np

class Depoly(object):
    # constructor
    def __init__(self, dataPath, netPath, snapshot, dataQueue, dataHandler, dataHandlerParameter, phase = "deploy", volSize = 64, ):
        self.dataPath = dataPath
        self.netPath = netPath
        self.dataQueue = dataQueue
        self.phase = phase
        self.phaseSubPath = self.phase + "/"
        self.snapshot = snapshot
        self.dataHandler = dataHandler
        self.dataHandlerParameter = dataHandlerParameter
        self.volSize = volSize

    # helper
    def testProcess(self, dataQueue, net):
        while (True):
            crop = dataQueue.get()
            net.blobs['data'].data[0, 0, :, :, :] = crop["image"].astype(dtype = np.float32)

            out = net.forward()
            # call registered data handler
            self.dataHandler(crop, out, self.dataHandlerParameter)

    # interface
    def speak(self):
        caffe.set_device(0)
        caffe.set_mode_gpu()

        net = caffe.Net(self.netPath + "deploy.prototxt", os.path.join(self.netPath, "snapshot/", self.snapshot), caffe.TEST)
        self.testProcess(self.dataQueue, net)
