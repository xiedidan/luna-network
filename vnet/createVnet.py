# -*- coding:utf-8 -*-

import sys
sys.path.append("../luna-data-pre-processing")
from CaffeSolver import CaffeSolver

import os

import caffe
from caffe import layers as L, params as P

def resUnit2(bottom, kernelSize=3, numberOfOutput=16, stride=1, pad=1):
    bn1 = L.BatchNorm(bottom)
    relu1 = L.PReLU(bn1)
    conv1 = L.Convolution(relu1, convolution_param={"engine": 2, "kernel_size": kernelSize, "stride": stride, "num_output": numberOfOutput, "pad": pad, "group": 1})

    bn2 = L.BatchNorm(conv1)
    relu2 = L.PReLU(bn2)
    conv2 = L.Convolution(relu2, convolution_param={"engine": 2, "kernel_size": kernelSize, "stride": stride, "num_output": numberOfOutput, "pad": pad, "group": 1})

    add = L.Eltwise(bottom, conv2)

    return bn1, relu1, conv1, bn2, relu2, conv2, add


def resUnit3(bottom, kernelSize=3, numberOfOutput=16, stride=1, pad=1):
    bn1 = L.BatchNorm(bottom)
    relu1 = L.PReLU(bn1)
    conv1 = L.Convolution(relu1, convolution_param={"engine": 2, "kernel_size": kernelSize, "stride": stride, "num_output": numberOfOutput, "pad": pad, "group": 1})

    bn2 = L.BatchNorm(conv1)
    relu2 = L.PReLU(bn2)
    conv2 = L.Convolution(relu2, convolution_param={"engine": 2, "kernel_size": kernelSize, "stride": stride, "num_output": numberOfOutput, "pad": pad, "group": 1})

    bn3 = L.BatchNorm(conv2)
    relu3 = L.PReLU(bn3)
    conv3 = L.Convolution(relu3, convolution_param={"engine": 2, "kernel_size": kernelSize, "stride": stride, "num_output": numberOfOutput, "pad": pad, "group": 1})

    add = L.Eltwise(bottom, conv3)

    return bn1, relu1, conv1, bn2, relu2, conv2, bn3, relu3, conv3, add


def vnet(phase, dataLayer, dataLayerParams):
    net = caffe.NetSpec()

    net.data, net.label = L.Python(module="NpyDataLayer", layer=dataLayer, ntop=2, param_str=str(dataLayerParams))
    # net.conv_input = L.Convolution(net.data, convolution_param={"engine": 2, "kernel_size": 1, "stride": 1, "num_output": 16, "pad": 0})
    net.split_input1, net.split_input2, net.split_input3, net.split_input4, net.split_input5, net.split_input6, net.split_input7, net.split_input8, net.split_input9, net.split_input10, net.split_input11, net.split_input12, net.split_input13, net.split_input14, net.split_input15, net.split_input16 = L.Split(net.data, ntop=16)
    net.concat_input = L.Concat(net.split_input1, net.split_input2, net.split_input3, net.split_input4, net.split_input5, net.split_input6, net.split_input7, net.split_input8, net.split_input9, net.split_input10, net.split_input11, net.split_input12, net.split_input13, net.split_input14, net.split_input15, net.split_input16)

    # left block 1 - 16 channel, size 64
    net.left1_bn1, net.left1_relu1, net.left1_conv1, net.left1_bn2, net.left1_relu2, net.left1_conv2, net.left1_add = resUnit2(
        net.concat_input)
    net.pooling1 = L.Convolution(net.left1_add, convolution_param={"engine": 2, "kernel_size": 2, "stride": 2, "num_output": 32, "pad": 0})

    # left block 2 - 32 channel, size 32
    net.left2_bn1, net.left2_relu1, net.left2_conv1, net.left2_bn2, net.left2_relu2, net.left2_conv2, net.left2_bn3, net.left2_relu3, net.left2_conv3, net.left2_add = resUnit3(
        net.pooling1, numberOfOutput=32)
    net.pooling2 = L.Convolution(net.left2_add, convolution_param={"engine": 2, "kernel_size": 2, "stride": 2, "num_output": 64, "pad": 0})

    # left block 3 - 64 channel, size 16
    net.left3_bn1, net.left3_relu1, net.left3_conv1, net.left3_bn2, net.left3_relu2, net.left3_conv2, net.left3_bn3, net.left3_relu3, net.left3_conv3, net.left3_add = resUnit3(
        net.pooling2, numberOfOutput=64)
    net.pooling3 = L.Convolution(net.left3_add, convolution_param={"engine": 2, "kernel_size": 2, "stride": 2, "num_output": 128, "pad": 0})

    # vally block - 128 channel, size 8
    net.vally_bn1, net.vally_relu1, net.vally_conv1, net.vally_bn2, net.vally_relu2, net.vally_conv2, net.vally_bn3, net.vally_relu3, net.vally_conv3, net.vally_add = resUnit3(
        net.pooling3, numberOfOutput=128)
    net.depooling1 = L.Deconvolution(net.vally_add, convolution_param={"engine": 2, "kernel_size": 2, "stride": 2, "num_output": 64, "pad": 0})
    
    # right block 1 - 128 channel, size 16
    net.concat1 = L.Concat(net.left3_add, net.depooling1)
    net.right1_bn1, net.right1_relu1, net.right1_conv1, net.right1_bn2, net.right1_relu2, net.right1_conv2, net.right1_bn3, net.right1_relu3, net.right1_conv3, net.right1_add = resUnit3(
        net.concat1, numberOfOutput=128)
    net.depooling2 = L.Deconvolution(net.right1_add, convolution_param={"engine": 2, "kernel_size": 2, "stride": 2, "num_output": 32, "pad": 0})

    # right block 2 - 32 channel, size 32
    net.concat2 = L.Concat(net.left2_add, net.depooling2)
    net.right2_bn1, net.right2_relu1, net.right2_conv1, net.right2_bn2, net.right2_relu2, net.right2_conv2, net.right2_bn3, net.right2_relu3, net.right2_conv3, net.right2_add = resUnit3(
        net.concat2, numberOfOutput=64)
    net.depooling3 = L.Deconvolution(net.right2_add, convolution_param={"engine": 2, "kernel_size": 2, "stride": 2, "num_output": 16, "pad": 0})

    # right block 3 - 16 channel, size 64
    net.concat3 = L.Concat(net.left1_add, net.depooling3)
    net.right3_bn1, net.right3_relu1, net.right3_conv1, net.right3_bn2, net.right3_relu2, net.right3_conv2, net.right3_add = resUnit2(
        net.concat3, numberOfOutput=32)

    # output - 2 channel, size 64
    net.conv_output = L.Convolution(net.right3_add, convolution_param={"engine": 2, "kernel_size": 1, "stride": 1, "num_output": 2, "pad": 0})

    # reshape result and label
    net.flat_output = L.Reshape(net.conv_output, reshape_param={"shape": {"dim": [0, 2, -1]}})
    net.flat_label = L.Reshape(net.label, reshape_param={"shape": {"dim": [0, 1, -1]}})

    # softmax result
    net.softmax_output = L.Softmax(net.flat_output)

    if phase == "train":
        net.dice_loss = L.DiceLoss(net.softmax_output, net.flat_label)
    elif phase == "test":
        net.accuracy = L.Accuracy(net.softmax_output, net.flat_label)
    elif phase == "deploy":
        net.output = L.Argmax(net.softmax_output, argmax_param={"axis": 1})

    return str(net.to_proto())

def writeVnet(dataPath="d:/project/tianchi/data/", workPath="./vnet/", batchSize=4):
    if not os.path.isdir(workPath):
        os.makedirs(workPath)

    # solver.prototxt
    solver = CaffeSolver(subPath=workPath, debug=False)
    solver.sp["display"] = '10' # WARNING: all params must be str!
    solver.sp["base_lr"] = '0.1'
    solver.sp['weight_decay'] = '0.05'
    solver.write("{0}solver.prototxt".format(workPath))

    # train.prototxt
    with open("{0}train.prototxt".format(workPath), "w") as file:
        dataLayerParams = dict(data_path=dataPath, net_path=workPath, iter_count=100000, batch_size=batchSize, phase="train", queue_size=30, vol_size=64, shift_ratio=0.5, rotate_ratio=0.5)
        train = vnet(phase="train", dataLayer="NpyDataLayer", dataLayerParams=dataLayerParams)
        file.write(train)

    # test.prototxt
    with open("{0}test.prototxt".format(workPath), "w") as file:
        dataLayerParams = dict(data_path=dataPath, net_path=workPath, iter_count=100000, batch_size=batchSize, phase="test", queue_size=30, vol_size=64, shift_ratio=0.5, rotate_ratio=0.5)
        test = vnet(phase="test", dataLayer="NpyDataLayer", dataLayerParams=dataLayerParams)
        file.write(test)

    # deploy.prototxt
    with open("{0}deploy.prototxt".format(workPath), "w") as file:
        dataLayerParams = dict(data_path=dataPath, net_path=workPath, iter_count=100000, batch_size=batchSize, phase="deploy", queue_size=30, vol_size=64, shift_ratio=0.5, rotate_ratio=0.5)
        deploy = vnet(phase="deploy", dataLayer="NpyDataLayer", dataLayerParams=dataLayerParams)
        file.write(deploy)

if __name__ == "__main__":
    writeVnet(workPath="v3/", batchSize=2)
