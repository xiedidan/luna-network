# -*- coding:utf-8 -*-

import os
from glob import glob
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle

# create label for resnext tests
# glob filename from nodules/, write label to groundTruths/
class GroundTruthCreator(object):
    def __init__(self, dataPath, phrase = "train"):
        self.dataPath = dataPath
        self.phrase = phrase
        self.phraseSubPath = self.phrase + "/"
        np.random.seed()

    def createLabel(self):
        groundTruthPath = self.dataPath + self.phraseSubPath + "groundTruths/"
        if not os.path.isdir(groundTruthPath):
            os.makedirs(groundTruthPath)

        noduleFileList = glob(self.dataPath + self.phraseSubPath + "nodules/*.npy")
        print(noduleFileList)
        for noduleFile in noduleFileList:
            label = np.array([0, 0])
            if np.random.random() > 0.5:
                label[0] = 1
            else:
                label[1] = 1
            
            filename = os.path.basename(noduleFile)
            groundTruthFile = groundTruthPath + filename
            print(groundTruthFile)

            with open(groundTruthFile, "wb") as f:
                pickle.dump(label, f)
        
if __name__ == "__main__":
    creator = GroundTruthCreator("d:/project/tianchi/data/experiment/")
    creator.createLabel()
