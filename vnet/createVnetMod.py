# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")
from CaffeSolver import CaffeSolver

import os
import numpy as np

import caffe

from caffe import layers as L
from caffe import params as P

class ResNetCreator(object):
    # constructor
    def __init__(self, dataLayer="VnetDataLayer", bottleneck = True, stages = [3, 4, 6, 3]):
        self.dataLayer = dataLayer
        self.bottleneck = bottleneck
        self.stages = stages
        if self.bottleneck == True:
            self.resnetString = "n.res(n)_bn1, n.res(n)_relu1, n.res(n)_conv1, \
                    n.res(n)_bn2, n.res(n)_relu2, n.res(n)_conv2, \
                    n.res(n)_bn3, n.res(n)_relu3, n.res(n)_conv3, n.res(n)_add = \
                    self.resnetBottleneckBlock((bottom), numOfOutput=(output))"
        else:
            self.resnetString = "n.res(n)_bn1, n.res(n)_relu1, n.res(n)_conv1, \
                    n.res(n)_bn2, n.res(n)_relu2, n.res(n)_conv2, n.res(n)_add = \
                    self.resnetBlock((bottom), numOfOutput=(output))"

    # interface
    def write(self, dataPath = "d:/project/tianchi/data/", workPath = "resnet_idmap_v1/", batchSize = 2):
        if not os.path.isdir(workPath):
            os.makedirs(workPath)

        # solver.prototxt
        solver = CaffeSolver(subPath=workPath, debug=False)
        solver.sp["display"] = '10'  # WARNING: all params must be str!
        solver.write("{0}solver.prototxt".format(workPath))

        # train.prototxt
        with open("{0}train.prototxt".format(workPath), "w") as file:
            dataLayerParams = dict(data_path=dataPath, net_path=workPath, iter_count=100000, batch_size=batchSize, phase="train", queue_size=30, vol_size=64, shift_ratio=0.9, rotate_ratio=0.9, histogram_shift_ratio=0.9)
            train = self.create(phase="train", dataLayerParams=dataLayerParams)
            file.write("{0}".format(train))

        # test.prototxt
        with open("{0}test.prototxt".format(workPath), "w") as file:
            dataLayerParams = dict(data_path=dataPath, net_path=workPath, iter_count=100000, batch_size=batchSize, phase="test", queue_size=30, vol_size=64, shift_ratio=0.0, rotate_ratio=0.0, histogram_shift_ratio=0.0)
            test = self.create(phase="test", dataLayerParams=dataLayerParams)
            file.write("{0}".format(test))

        # deploy.prototxt
        with open("{0}deploy.prototxt".format(workPath), "w") as file:
            dataLayerParams = dict(data_path=dataPath, net_path=workPath, iter_count=100000, batch_size=batchSize, phase="deploy", queue_size=30, vol_size=64, shift_ratio=0.0, rotate_ratio=0.0, histogram_shift_ratio=0.0)
            deploy = self.create(phase="deploy", dataLayerParams=dataLayerParams)
            file.write("{0}".format(deploy))

    # helper
    def create(self, dataLayerParams,  phase = "train"):
        n = caffe.NetSpec()

        n.data, n.label = L.Python(module="VDataLayer", layer=self.dataLayer, ntop=2, param_str=str(dataLayerParams))

        n.input_conv = L.Convolution(n.data, num_output=16, kernel_size=1, stride=1, pad=1, bias_term=False,
                                     param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type="xavier"))
        n.input_relu = L.ReLU(n.input_conv, in_place=False)

        for i in range(len(self.stages)):
            for j in range(self.stages[i]):
                stageString = self.resnetString

                bottomString = 'n.input_relu'
                if (i != 0) or (j != 0):
                    bottomString = 'n.res{}_add'.format(str(sum(self.stages[:i]) + j))

                exec(stageString.replace('(bottom)', bottomString).
                      replace('(output)', str(2 ** i * 64)).
                      replace('(n)', str(sum(self.stages[:i]) + j + 1)))

        exec('n.conv_output = L.Convolution(n.res{}_add, num_output=2, kernel_size=1, stride=1, pad=1, bias_term=False, param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type="xavier"))'.format(
            str(sum(self.stages))))

        # reshape result and label
        n.flat_output = L.Reshape(n.conv_output, reshape_param={"shape": {"dim": [0, 2, -1]}})
        n.flat_label = L.Reshape(n.label, reshape_param={"shape": {"dim": [0, 1, -1]}})

        if phase == "train":
            n.softmax_out = L.Softmax(n.flat_output)
            n.loss = L.DiceLoss(n.softmax_out, n.flat_label)
        elif phase == "test":
            n.softmax_out = L.Softmax(n.flat_output)
            n.accu = L.DiceLoss(n.softmax_out, n.flat_label)
        else: # deploy
            n.softmax_out = L.Softmax(n.flat_output)
            n.result = L.ArgMax(n.softmax_out, argmax_param=dict(axis=1))

        return n.to_proto()

    # bottleneck residual block
    def resnetBottleneckBlock(self, bottom, numOfOutput):
        bn1 = L.BatchNorm(bottom, use_global_stats=False, in_place=False)
        relu1 = L.ReLU(bn1, in_place=False)
        conv1 = L.Convolution(relu1, num_output=int(round(numOfOutput / 4)), kernel_size=1, stride=1, pad=0, bias_term=False,
                              param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type="xavier"))

        bn2 = L.BatchNorm(conv1, use_global_stats=False, in_place=False)
        relu2 = L.ReLU(bn2, in_place=False)
        conv2 = L.Convolution(relu2, num_output=int(round(numOfOutput / 4)), kernel_size=3, stride=1, pad=1, bias_term=False,
                              param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type="xavier"))

        bn3 = L.BatchNorm(conv2, use_global_stats=False, in_place=False)
        relu3 = L.ReLU(bn3, in_place=False)
        conv3 = L.Convolution(relu3, num_output=numOfOutput, kernel_size=1, stride=1, pad=0, bias_term=False,
                              param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type="xavier"))

        add = L.Eltwise(bottom, conv3, eltwise_param=dict(operation=1))

        return bn1, relu1, conv1, bn2, relu2, conv2, bn3, relu3, conv3, add

    # common residual block
    def resnetBlock(self, bottom, numOfOutput):
        bn1 = L.BatchNorm(bottom, use_global_stats=False, in_place=False)
        relu1 = L.ReLU(bn1, in_place=False)
        conv1 = L.Convolution(relu1, num_output=numOfOutput, kernel_size=3, stride=1, pad=1, bias_term=False,
                              param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type="xavier"))

        bn2 = L.BatchNorm(conv1, use_global_stats=False, in_place=False)
        relu2 = L.ReLU(bn2, in_place=False)
        conv2 = L.Convolution(relu2, num_output=numOfOutput, kernel_size=3, stride=1, pad=1, bias_term=False,
                              param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type="xavier"))

        add = L.Eltwise(bottom, conv2, eltwise_param=dict(operation=1))

        return bn1, relu1, conv1, bn2, relu2, conv2, add

if __name__ == "__main__":
    # update parameters here, to replace the default configs
    creator = ResNetCreator(dataLayer="VDataLayer", bottleneck = True, stages = [3, 4, 6, 3], classCount = 2)
    creator.write(dataPath = "c:/project/tianchi/data/", workPath = "v4/", batchSize = 2)
