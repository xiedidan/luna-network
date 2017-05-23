# -*- coding:utf-8 -*-

import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import scipy.ndimage
import os
import array
import math
from tqdm import tqdm
import h5py
try:
    import cPickle as pickle
except:
    import pickle

class NoduleSerializer(object):
    # constructor
    def __init__(self, dataPath):
        # path
        self.dataPath = dataPath
        self.lmdbPath = os.path.join(self.dataPath, "lmdb/")

    # helper
    def writeMhdMetaHeader(self, filename, meta):
        header = ""

        # do not use tags = meta.keys() since the order of tags matters
        tags = ['ObjectType', 'NDims', 'BinaryData',
                'BinaryDataByteOrderMSB', 'CompressedData', 'CompressedDataSize',
                'TransformMatrix', 'Offset', 'CenterOfRotation',
                'AnatomicalOrientation',
                'ElementSpacing',
                'DimSize',
                'ElementType',
                'ElementDataFile',
                'Comment', 'SeriesDescription', 'AcquisitionDate', 'AcquisitionTime', 'StudyDate', 'StudyTime']
        for tag in tags:
            if tag in meta.keys():
                header += "%s = %s\n" % (tag, meta[tag])

        file = open(filename, "w")
        file.write(header)
        file.close()

    def writeMhdRawData(self, filename, data):
        data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
        file = open(filename, "wb")

        arr = array.array("h")
        for item in data:
            arr.fromlist(list(item))
        arr.tofile(file)

        file.close()

    # interface
    def writeToMhd(self, subPath, seriesuid, idx, nodule):
        mhdPath = self.dataPath + subPath
        if not os.path.isdir(mhdPath):
            os.makedirs(mhdPath)

        filename = mhdPath + "nodule-" + seriesuid + "-" + str(idx) + ".mhd"

        dimSize = nodule.shape

        meta = {}
        meta["ObjectType"] = "Image"
        meta["BinaryData"] = "True"
        meta["BinaryDataByteOrderMSB"] = "false"
        meta["ElementType"] = "MET_SHORT"
        meta["NDims"] = str(len(dimSize))
        meta["DimSize"] = " ".join([str(i) for i in dimSize])
        meta["ElementDataFile"] = os.path.split(filename)[1].replace(".mhd", ".raw")
        self.writeMhdMetaHeader(filename, meta)

        pwd = os.path.split(filename)[0]
        if pwd:
            dataPath = pwd + "/" + meta["ElementDataFile"]
        else:
            dataPath = meta["ElementDataFile"]

        self.writeMhdRawData(dataPath, nodule)

    def initHdf5(self, subPath):
        hdfPath = self.dataPath + subPath

        if not os.path.isdir(hdfPath):
            os.makedirs(hdfPath)
        else:
            dataListFile = hdfPath + "data-list.txt"
            if os.path.isfile(dataListFile):
                os.remove(dataListFile)

    def writeFileNoduleToHdf5(self, subPath, fileNodules, gzipFlag = False):
        seriesuid = fileNodules["seriesuid"]
        hdfPath = self.dataPath + subPath

        filename = ""
        if not gzipFlag:
            filename = hdfPath + seriesuid + ".h5"
            with h5py.File(filename, "w") as file:
                file["data"] = fileNodules["nodules"]
                file["label"] = fileNodules["groundTruths"]
        else:
            filename = hdfPath + seriesuid + "_gzip" + ".h5"
            with h5py.File(filename, "w") as file:
                file.create_dataset(
                    "data", data = fileNodules["nodules"],
                    compression = "gzip", compression_opts = 1
                )
                file.create_dataset(
                    "label", data = fileNodules["groundTruths"],
                    compression = "gzip", compression_opts=1
                )

        listFilename = hdfPath + "data-list.txt"
        with open(listFilename, "a") as listFile:
            listFile.write(filename)

    def writeToNpy(self, subPath, filename, image):
        npyPath = self.dataPath + subPath
        if not os.path.isdir(npyPath):
            os.makedirs(npyPath)

        imageFile = open(npyPath + filename, "wb")
        try:
            pickle.dump(image, imageFile)
        finally:
            imageFile.close()

    def readFromNpy(self, subPath, filename):
        npyPath = self.dataPath + subPath

        imageFile = open(npyPath + filename, "rb")
        try:
            image = pickle.load(imageFile)
        finally:
            imageFile.close()

        return image
