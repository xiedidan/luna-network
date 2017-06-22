# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")
sys.path.append("../common")
from NoduleSerializer import NoduleSerializer
from Prophet import Prophet

import multiprocessing
import numpy as np

from scan import Scanner

def dataScanner(dataPath, phase, dataQueue):
    scanner = Scanner(dataPath, phase, dataQueue)
    scanner.scanAllFiles()

def dataHandler(input, output, parameter):
    serializer = parameter["serializer"]
    batchSize = parameter["batchSize"]

    labels = output["result"]
    # labels = np.squeeze(labels)
    for i in range(batchSize):
        data = input[i]

        label = labels[i]
        out = label[1] - label[0]
        out[out < 0] = 0.
        out = np.rint(out).astype(dtype=np.int8)

        serializer.writeToNpy("results/", "{0}-{1}.npy".format(data["seriesuid"], data["number"]), out)

if __name__ == "__main__":
    dataPath = "d:/project/tianchi/data/"
    netPath = "v1/"
    queueSize = 32
    batchSize = 1

    # start data reader
    dataQueue = multiprocessing.Queue(queueSize)

    scanProcess = multiprocessing.Process(target = dataScanner, args = (dataPath, "test", dataQueue))
    scanProcess.daemon = True
    scanProcess.start()

    # start prophet
    serializer = NoduleSerializer(dataPath, "test")
    dataHandlerParameter = {}
    dataHandlerParameter["serializer"] = serializer
    dataHandlerParameter["batchSize"] = batchSize

    prophet = Prophet(dataPath, netPath, batchSize = batchSize, snapshot = "_iter_20000.caffemodel", dataQueue = dataQueue, dataHandler = dataHandler, dataHandlerParameter = dataHandlerParameter)
    prophet.speak()
