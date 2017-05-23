# -*- coding:utf-8 -*-

from NoduleCropper import NoduleCropper

dataPath = "d:/project/tianchi/data/"

cropper = NoduleCropper(dataPath)
cropper.resampleAndCreateGroundTruth()
