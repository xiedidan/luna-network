# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")

from NoduleSerializer import NoduleSerializer

import os
import glob

from tqdm import tqdm

class Concat(object):
    def __init__(self, dataPath, phase = "deploy", stepSize = 32, volSize = 64):
        self.dataPath = dataPath
        self.phase = phase
        self.phaseSubPath = phase + "/"
        self.stepSize = stepSize
        self.volSize = volSize
        self.serializer = NoduleSerializer(dataPath, phase)

    # helper
    def concatSingleFile(self, seriesuid, steps):

    # interface
    def concatAllFile(self):
        # read file list from nodules/
        metaPath = self.dataPath + self.phaseSubPath + "meta/"
        metaFileList = glob(metaPath)
        for metaFile in enumerate(tqdm(metaFileList)):
            metaFilename = metaFile[1]
            meta = self.serializer.readFromNpy(metaFilename)
            steps = meta["steps"]

            seriesuid = os.path.basename(metaFilename).split(".")[0]
