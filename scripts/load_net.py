#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, 'fast-rcnn/caffe-fast-rcnn/build/install/python')
sys.path.insert(0, 'models')
import caffe
from VGG import VGG
import cPickle as pickle

param_dir = 'fast-rcnn/data/fast_rcnn_models'
param_fn = '%s/vgg16_fast_rcnn_iter_40000.caffemodel' % param_dir
model_dir = 'fast-rcnn/models/VGG16'
model_fn = '%s/test.prototxt' % model_dir

vgg = VGG()
net = caffe.Net(model_fn, param_fn, caffe.TEST)
for name, param in net.params.iteritems():
    layer = getattr(vgg, name)

    print name, param[0].data.shape, param[1].data.shape,
    print layer.W.shape, layer.b.shape

    layer.W = param[0].data
    layer.b = param[1].data
    setattr(vgg, name, layer)

pickle.dump(vgg, open('models/VGG.chainermodel', 'wb'), -1)
