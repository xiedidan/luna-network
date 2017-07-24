import sys
sys.path.append("../luna-data-pre-processing")
from NoduleCropper import NoduleCropper
from NoduleSerializer import NoduleSerializer
from segment import Segment

from tqdm import tqdm

dataPath = "d:/project/tianchi/data/"
cropSizes = {"train": 128, "test": 64, "deploy": 64}

# TODO : prepare for training
trainCropper = NoduleCropper(dataPath = dataPath, phase = "train", cropSize = cropSizes["train"])
trainSerializer = NoduleSerializer(dataPath = dataPath, phase = "train")
trainSegment = Segment(dataPath = dataPath, phase = "train")
# TODO : resample and create groundTruth

trainCropper.resampleAndCreateGroundTruth()
trainSegment.segmentAllFiles()

# TODO : segmentation

# TODO : crop

# TODO : prepare for testing

# TODO : prepare for deployment
