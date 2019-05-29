# Class Balance Loss

## Abstract

we exploit the recently proposed class-balanced loss to re-weight instances, which reformulates the softmax loss by introducing a weight that is inversely proportional to the number of instances in each class.

## Uses

The codes are based on [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) for Caffe Toolkits.

The file ```roi_weighted_layer.py``` is modified from the original input layer in py-faster-rcnn by calculating the weights for each category.

The files ```weighted_softmax_loss_layer.cpp``` and ```weighted_softmax_loss_layer.cu``` are the cuda and cpp implementations for imbalance loss for softmax loss.
