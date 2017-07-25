import sys
sys.path.append("../luna-data-pre-processing")
from NoduleCropper import NoduleCropper
from segment import Segment

from tqdm import tqdm

dataPath = "d:/project/tianchi/data/"
cropSizes = {"train": 128, "test": 64, "deploy": 64}

# prepare for training
trainCropper = NoduleCropper(dataPath = dataPath, phase = "train", cropSize = cropSizes["train"])
trainSegment = Segment(dataPath = dataPath, phase = "train")

trainCropper.resampleAndCreateGroundTruth()
trainSegment.segmentAllFiles()
trainCropper.cropAllNoduleOffline()

# prepare for testing
testCropper = NoduleCropper(dataPath = dataPath, phase = "test", cropSize = cropSizes["test"])
testSegment = Segment(dataPath = dataPath, phase = "test")

testCropper.resampleAndCreateGroundTruth()
testSegment.segmentAllFiles()
testCropper.cropAllNoduleOffline()

# prepare for deployment
deployCropper = NoduleCropper(dataPath = dataPath, phase = "deploy", cropSize = cropSizes["deploy"])
deploySegment = Segment(dataPath = dataPath, phase = "deploy")

deployCropper.resampleAllFiles()
deploySegment.segmentAllFiles()
