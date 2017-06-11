# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")
from CaffeSolver import CaffeSolver

import os

import caffe

from caffe import layers as L
from caffe import params as P

class ResNetCreator(object):
    # constructor
    def __init__(self, dataLayer="NpyDataLayer", bottleneck = True, stages = [3, 4, 6, 3], classCount = 2):
        self.dataLayer = dataLayer
        self.bottleneck = bottleneck
        self.stages = stages
        self.classCount = classCount
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
    def write(self, dataPath = "d:/project/tianchi/data/", workPath = "resnet_v1/", batchSize = 2):
        if not os.path.isdir(workPath):
            os.makedirs(workPath)

        # solver.prototxt
        solver = CaffeSolver(subPath=workPath, debug=False)
        solver.sp["display"] = '10'  # WARNING: all params must be str!
        solver.write("{0}solver.prototxt".format(workPath))

        # train.prototxt
        with open("{0}train.prototxt".format(workPath), "w") as file:
            dataLayerParams = dict(data_path=dataPath, net_path=workPath, iter_count=100000, batch_size=batchSize,
                                   phase="train", queue_size=30, vol_size=64, shift_ratio=0.5, rotate_ratio=0.5)
            train = self.create(phase="train", dataLayerParams=dataLayerParams)
            file.write(train)

        # test.prototxt
        with open("{0}test.prototxt".format(workPath), "w") as file:
            dataLayerParams = dict(data_path=dataPath, net_path=workPath, iter_count=100000, batch_size=batchSize,
                                   phase="test", queue_size=30, vol_size=64, shift_ratio=0.5, rotate_ratio=0.5)
            test = self.create(phase="test", dataLayerParams=dataLayerParams)
            file.write(test)

        # deploy.prototxt
        with open("{0}deploy.prototxt".format(workPath), "w") as file:
            dataLayerParams = dict(data_path=dataPath, net_path=workPath, iter_count=100000, batch_size=batchSize,
                                   phase="deploy", queue_size=30, vol_size=64, shift_ratio=0.5, rotate_ratio=0.5)
            deploy = self.create(phase="deploy", dataLayerParams=dataLayerParams)
            file.write(deploy)

    # helper
    def create(self, dataLayerParams,  phase = "train"):
        n = caffe.NetSpec()

        n.data, n.label = L.Python(module="NpyDataLayer", layer=self.dataLayer, ntop=2, param_str=str(dataLayerParams))

        n.input_conv = L.Convolution(n.data, num_output=16, kernel_size=1, stride=1, pad=1, bias_term=False,
                                     param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type="xavier"))
        n.input_relu = L.ReLU(n.input_conv, in_place=True)

        for i in len(self.stages):
            for j in self.stages[i]:
                stageString = self.resnetString

                bottomString = 'n.input_relu'
                if (i != 0) and (j != 0):
                    bottomString = 'n.res{}_add'.format(str(sum(self.stages[:i]) + j))

                exec(stageString.replace('(bottom)', bottomString).
                      replace('(numOfOutput)', str(2 ** i * 64)).
                      replace('(n)', str(sum(self.stages[:i]) + j + 1)))

        exec('n.pool_ave = L.Pooling(n.res{}_add, pool=P.Pooling.AVE, global_pooling=True)'.format(
            str(sum(self.stages))))
        n.classifier = L.InnerProduct(n.pool_ave, num_output=self.classCount,
                                      param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                                      weight_filler=dict(type='xavier'),
                                      bias_filler=dict(type='constant', value=0))

        if phase == "train":
            n.loss = L.SoftmaxWithLoss(n.classifier, n.label)
        elif phase == "test":
            n.accuracy_top1 = L.Accuracy(n.classifier, n.label, include=dict(phase=1))
            n.accuracy_top5 = L.Accuracy(n.classifier, n.label, include=dict(phase=1), accuracy_param=dict(top_k=5))
        else: # deploy
            n.result = L.Softmax(n.classifier, n.label)

        return n.to_proto()

    # bottleneck residual block
    def resnetBottleneckBlock(self, bottom, numOfOutput):
        bn1 = L.BatchNorm(bottom, use_global_stats=False, in_place=True)
        relu1 = L.ReLU(bn1, in_place=True)
        conv1 = L.Convolution(relu1, num_output=numOfOutput / 4, kernel_size=1, stride=1, pad=0, bias_term=False,
                              param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type="xavier"))

        bn2 = L.BatchNorm(conv1, use_global_stats=False, in_place=True)
        relu2 = L.ReLU(bn2, in_place=True)
        conv2 = L.Convolution(relu2, num_output=numOfOutput / 4, kernel_size=3, stride=1, pad=1, bias_term=False,
                              param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type="xavier"))

        bn3 = L.BatchNorm(conv2, use_global_stats=False, in_place=True)
        relu3 = L.ReLU(bn3, in_place=True)
        conv3 = L.Convolution(relu3, num_output=numOfOutput, kernel_size=1, stride=1, pad=0, bias_term=False,
                              param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type="xavier"))

        add = L.Eltwise(bottom, conv3, eltwise_param=dict(operation=1))

        return bn1, relu1, conv1, bn2, relu2, conv2, bn3, relu3, conv3, add

    # common residual block
    def resnetBlock(self, bottom, numOfOutput):
        bn1 = L.BatchNorm(bottom, use_global_stats=False, in_place=True)
        relu1 = L.ReLU(bn1, in_place=True)
        conv1 = L.Convolution(relu1, num_output=numOfOutput, kernel_size=3, stride=1, pad=1, bias_term=False,
                              param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type="xavier"))

        bn2 = L.BatchNorm(conv1, use_global_stats=False, in_place=True)
        relu2 = L.ReLU(bn2, in_place=True)
        conv2 = L.Convolution(relu2, num_output=numOfOutput, kernel_size=3, stride=1, pad=1, bias_term=False,
                              param=[dict(lr_mult=1, decay_mult=1)], weight_filler=dict(type="xavier"))

        add = L.Eltwise(bottom, conv2, eltwise_param=dict(operation=1))

        return bn1, relu1, conv1, bn2, relu2, conv2, add

if __name__ == "__main__":
    creator = ResNetCreator()
    creator.write()
