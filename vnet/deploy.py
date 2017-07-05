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

def resultWriter(serializer, resultQueue):
    while True:
        result = resultQueue.get()
        if result["finishFlag"] == True:
            break
        else:
            serializer.writeToNpy("results/", "{0}-{1}.npy".format(result["seriesuid"], result["number"]), result["image"])

def dataHandler(input, output, parameter):
    # serializer = parameter["serializer"]
    batchSize = parameter["batchSize"]
    resultQueue = parameter["resultQueue"]

    labels = output["result"]
    # labels = np.squeeze(labels)
    for i in range(batchSize):
        data = input[i]

        label = labels[i]
        out = label[1]
        out[out < 0] = 0.
        out[out > 1] = 1.

        #out = np.rint(out).astype(dtype=np.int8)

        # construct crop
        crop = {}
        crop["seriesuid"] = data["seriesuid"]
        crop["number"] = data["number"]
        crop["steps"] = data["steps"]
        crop["finishFlag"] = False

        crop["image"] = out

        resultQueue.put(crop)

        # serializer.writeToNpy("results/", "{0}-{1}.npy".format(data["seriesuid"], data["number"]), out)

if __name__ == "__main__":
    dataPath = "d:/project/tianchi/data/"
    netPath = "v1/"
    phase = "deploy"
    snapshot = "_iter_20000.caffemodel"
    queueSize = 512
    batchSize = 1

    # start data reader
    dataQueue = multiprocessing.Queue(queueSize)

    scanProcess = multiprocessing.Process(target = dataScanner, args = (dataPath, phase, dataQueue))
    scanProcess.daemon = True
    scanProcess.start()

    # start result writer
    serializer = NoduleSerializer(dataPath, phase)
    resultQueue = multiprocessing.Queue(queueSize)

    writerProcess = multiprocessing.Process(target = resultWriter, args = (serializer, resultQueue))
    writerProcess.daemon = True
    writerProcess.start()

    # start prophet
    dataHandlerParameter = {}
    # dataHandlerParameter["serializer"] = serializer
    dataHandlerParameter["batchSize"] = batchSize
    dataHandlerParameter["resultQueue"] = resultQueue

    prophet = Prophet(dataPath, netPath, batchSize = batchSize, snapshot = snapshot, dataQueue = dataQueue, dataHandler = dataHandler, dataHandlerParameter = dataHandlerParameter)
    prophet.speak()

