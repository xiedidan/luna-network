# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")
sys.path.append("../lib")
from NoduleSerializer import NoduleSerializer
from Prophet import Prophet

import multiprocessing
import numpy as np

from scan import Scanner

def dataScanner(dataPath, phase, dataQueue):
    scanner = Scanner(dataPath, phase, dataQueue)
    scanner.scanAllFiles()

def dataHandler(input, output, parameter):
    labels = output["results"]
    labels = np.squeeze(labels)
    out = labels[1] - labels[0]
    out[out < 0] = 0.
    out = np.rint(out).astype(dtype=np.int8)

    serializer = parameter
    serializer.writeToNpy("results/", "{0}-{1}.npy".format(input["seriesuid"], input["number"]), out)

if __name__ == "__main__":
    dataPath = "d:/project/tianchi/data/"
    netPath = "resnet_idmap_v1/"
    queueSize = 32

    # start data reader
    dataQueue = multiprocessing.Queue(queueSize)

    scanProcess = multiprocessing.Process(target = dataScanner, args = (dataPath, "deploy", dataQueue))
    scanProcess.daemon = True
    scanProcess.start()

    # start prophet
    serializer = NoduleSerializer(dataPath, "deploy")

    prophet = Prophet(dataPath, netPath, snapshot = "_iter_3000.caffemodel", dataQueue = dataQueue, dataHandler = dataHandler, dataHandlerParameter = serializer)
    prophet.speak()
