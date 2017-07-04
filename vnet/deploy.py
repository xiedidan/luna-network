# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")
sys.path.append("../common")
from NoduleSerializer import NoduleSerializer
from Prophet import Prophet
from notate import Notate

import multiprocessing
import numpy as np
import time

from scan import Scanner

def dataScanner(dataPath, phase, dataQueue):
    scanner = Scanner(dataPath, phase, dataQueue)
    scanner.scanAllFiles()

def dataHandler(input, output, parameter):
    serializer = parameter["serializer"]
    batchSize = parameter["batchSize"]
    notateQueue = parameter["notateQueue"]

    labels = output["result"]
    # labels = np.squeeze(labels)
    for i in range(batchSize):
        data = input[i]

        label = labels[i]
        out = label[1] - label[0]
        out[out < 0] = 0.

        #out = np.rint(out).astype(dtype=np.int8)

        # construct crop
        crop = {}
        crop["seriesuid"] = data["seriesuid"]
        crop["number"] = data["number"]
        crop["steps"] = data["steps"]
        crop["finishFlag"] = False

        crop["image"] = out

        # put crop into queue
        notateQueue.put(crop)

        # serializer.writeToNpy("results/", "{0}-{1}.npy".format(data["seriesuid"], data["number"]), out)

if __name__ == "__main__":
    dataPath = "d:/project/tianchi/data/"
    netPath = "v1/"
    phase = "deploy"
    snapshot = "_iter_20000.caffemodel"
    queueSize = 32
    batchSize = 1
    notateProcessCount = 8

    # start data reader
    dataQueue = multiprocessing.Queue(queueSize)

    scanProcess = multiprocessing.Process(target = dataScanner, args = (dataPath, phase, dataQueue))
    scanProcess.daemon = True
    scanProcess.start()

    # start notate processes
    notater = Notate(dataPath)
    notateQueue = multiprocessing.Queue(queueSize)

    notateProcessList = []
    for i in range(notateProcessCount):
        notateProcess = multiprocessing.Process(target = notater.notateProcess, args = (notateQueue, ))
        notateProcess.daemon = True
        notateProcess.start()
        notateProcessList.append(notateProcess)

    # start prophet
    serializer = NoduleSerializer(dataPath, "test")
    dataHandlerParameter = {}
    dataHandlerParameter["serializer"] = serializer
    dataHandlerParameter["batchSize"] = batchSize
    dataHandlerParameter["notateQueue"] = notateQueue

    prophet = Prophet(dataPath, netPath, batchSize = batchSize, snapshot = snapshot, dataQueue = dataQueue, dataHandler = dataHandler, dataHandlerParameter = dataHandlerParameter)
    prophet.speak()

    # wait data in queue to be processed to exit
    while notateQueue.qsize() > 0:
        time.sleep(1)
