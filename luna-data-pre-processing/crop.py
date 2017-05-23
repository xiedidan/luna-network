# -*- coding:utf-8 -*-

from tqdm import tqdm

from NoduleCropper import NoduleCropper
from NoduleSerializer import NoduleSerializer

dataPath = "d:/project/tianchi/data/"

cropper = NoduleCropper(dataPath = dataPath, cropSize = 128)
serializer = NoduleSerializer(dataPath)

mhdNodules = cropper.cropAllNoduleForMhd()
for fileNodules in tqdm(mhdNodules):
    for idx, nodule in enumerate(fileNodules["nodules"]):
        serializer.writeToNpy("npy/nodules/", fileNodules["seriesuid"] + "-" + str(idx) + ".npy", nodule)
    for idx, groundTruth in enumerate(fileNodules["groundTruths"]):
        serializer.writeToNpy("npy/groundTruths/", fileNodules["seriesuid"] + "-" + str(idx) + ".npy", groundTruth)
