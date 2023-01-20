# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
TensorFlow, Keras and TFLite versions of YOLOv5
Authored by https://github.com/zldrobit in PR https://github.com/ultralytics/yolov5/pull/1127

Usage:
    $ python models/tf.py --weights yolov5s.pt

Export:
    $ python path/to/export.py --weights yolov5s.pt --include saved_model pb tflite tfjs
"""

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from tensorflow import keras

# Add TFSPPF layer
from models.common import Conv, Bottleneck, SPP, SPPF, DWConv, Focus, BottleneckCSP, Concat, autopad, C3, ReOrg, DownC, SPPCSPC, MP, SP
from models.experimental import MixConv2d, attempt_load
from models.yolo import Detect, Segment
from utils.general import make_divisible, print_args, set_logging
from utils.activations import SiLU

LOGGER = logging.getLogger(__name__)


class TFConv(keras.layers.Layer):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        # ch_in, ch_out, weights, kernel, stride, padding, groups
        super(TFConv, self).__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        assert isinstance(k, int), "Convolution with multiple kernels are not allowed."
        # TensorFlow convolution padding is inconsistent with PyTorch (e.g. k=3 s=2 'SAME' padding)
        # see https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch

        conv = keras.layers.Conv2D(
            c2, k, s, 'SAME' if s == 1 else 'VALID', use_bias=False if hasattr(w, 'bn') else True,
            kernel_initializer=keras.initializers.Constant(w.conv.weight.permute(2, 3, 1, 0).detach().numpy()),
            bias_initializer='zeros' if hasattr(w, 'bn') else keras.initializers.Constant(w.conv.bias.detach().numpy())
        )
        self.conv = conv if s == 1 else keras.Sequential([TFPad(autopad(k, p)), conv])
        self.bn = TFBN(w.bn) if hasattr(w, 'bn') else tf.identity

        # YOLOv5 activations
        if isinstance(w.act, nn.LeakyReLU):
            self.act = (lambda x: keras.activations.relu(x, alpha=0.1)) if act else tf.identity
        elif isinstance(w.act, nn.Hardswish):
            self.act = (lambda x: x * tf.nn.relu6(x + 3) * 0.166666667) if act else tf.identity
        elif isinstance(w.act, (nn.SiLU, SiLU)):
            self.act = (lambda x: keras.activations.swish(x)) if act else tf.identity
        else:
            raise Exception(f'no matching TensorFlow activation found for {w.act}')

    def call(self, inputs):
        return self.act(self.bn(self.conv(inputs)))


class TFConv2d(keras.layers.Layer):
    # Substitution for PyTorch nn.Conv2D
    def __init__(self, c1, c2, k, s=1, g=1, bias=True, w=None):
        super(TFConv2d, self).__init__()
        assert g == 1, "TF v2.2 Conv2D does not support 'groups' argument"
        self.conv = keras.layers.Conv2D(
            c2, k, s, 'VALID', use_bias=bias,
            kernel_initializer=keras.initializers.Constant(w.weight.permute(2, 3, 1, 0).detach().numpy()),
            bias_initializer=keras.initializers.Constant(w.bias.detach().numpy()) if bias else None, )

    def call(self, inputs):
        return self.conv(inputs)


## YOLOV7
class TFMP(keras.layers.Layer):
    def __init__(self, k=2, w=None):
        super(TFMP, self).__init__()
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=k)

    def call(self, inputs):
        return self.m(inputs)


class TFSP(keras.layers.Layer):
    def __init__(self, k=3, s=1, w=None):
        super(TFSP, self).__init__()
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=s, padding='SAME')

    def call(self, inputs):
        return self.m(inputs)


class TFReOrg(keras.layers.Layer):
    def __init__(self, dimension=1, w=None):
        super(TFReOrg, self).__init__()

    def call(self, inputs):  # x(b,w,h,c) -> y(b,w/2,h/2,4c)
        return tf.concat([inputs[:, ::2, ::2, :],
                          inputs[:, 1::2, ::2, :],
                          inputs[:, ::2, 1::2, :],
                          inputs[:, 1::2, 1::2, :]], 3)


class TFDownC(keras.layers.Layer):
    def __init__(self, c1, c2, n=1, k=2, w=None):
        super(TFDownC, self).__init__()
        c_ = int(c1)
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_, c2 // 2, 3, k, w=w.cv2)
        self.cv3 = TFConv(c1, c2 // 2, 1, 1, w=w.cv3)
        self.mp = keras.layers.MaxPool2D(pool_size=k, strides=k)

    def call(self, inputs):
        return tf.concat([self.cv2(self.cv1(inputs)), self.cv3(self.mp(inputs))], axis=3)


class TFSPPCSPC(keras.layers.Layer):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13), w=None):
        super(TFSPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(c_, c_, 3, 1, w=w.cv3)
        self.cv4 = TFConv(c_, c_, 1, 1, w=w.cv4)
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding='SAME') for x in k]
        self.cv5 = TFConv(4 * c_, c_, 1, 1, w=w.cv5)
        self.cv6 = TFConv(c_, c_, 3, 1, w=w.cv6)
        self.cv7 = TFConv(2 * c_, c2, 1, 1, w=w.cv7)

    def call(self, inputs):
        x1 = self.cv4(self.cv3(self.cv1(inputs)))
        y1 = self.cv6(self.cv5(tf.concat([x1] + [m(x1) for m in self.m], axis=3)))
        y2 = self.cv2(inputs)
        return self.cv7(tf.concat([y1, y2], axis=3))


## YOLOV5
class TFBN(keras.layers.Layer):
    # TensorFlow BatchNormalization wrapper
    def __init__(self, w=None):
        super(TFBN, self).__init__()
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(w.bias.detach().numpy()),
            gamma_initializer=keras.initializers.Constant(w.weight.detach().numpy()),
            moving_mean_initializer=keras.initializers.Constant(w.running_mean.detach().numpy()),
            moving_variance_initializer=keras.initializers.Constant(w.running_var.detach().numpy()),
            epsilon=w.eps)

    def call(self, inputs):
        return self.bn(inputs)


class TFPad(keras.layers.Layer):
    def __init__(self, pad):
        super(TFPad, self).__init__()
        self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])

    def call(self, inputs):
        return tf.pad(inputs, self.pad, mode='constant', constant_values=0)


class TFFocus(keras.layers.Layer):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, w=None):
        # ch_in, ch_out, kernel, stride, padding, groups
        super(TFFocus, self).__init__()
        self.conv = TFConv(c1 * 4, c2, k, s, p, g, act, w.conv)

    def call(self, inputs):  # x(b,w,h,c) -> y(b,w/2,h/2,4c)
        # inputs = inputs / 255.  # normalize 0-255 to 0-1
        return self.conv(tf.concat([inputs[:, ::2, ::2, :],
                                    inputs[:, 1::2, ::2, :],
                                    inputs[:, ::2, 1::2, :],
                                    inputs[:, 1::2, 1::2, :]], 3))


class TFBottleneck(keras.layers.Layer):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, w=None):  # ch_in, ch_out, shortcut, groups, expansion
        super(TFBottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_, c2, 3, 1, g=g, w=w.cv2)
        self.add = shortcut and c1 == c2

    def call(self, inputs):
        return inputs + self.cv2(self.cv1(inputs)) if self.add else self.cv2(self.cv1(inputs))


class TFBottleneckCSP(keras.layers.Layer):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(TFBottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv2d(c1, c_, 1, 1, bias=False, w=w.cv2)
        self.cv3 = TFConv2d(c_, c_, 1, 1, bias=False, w=w.cv3)
        self.cv4 = TFConv(2 * c_, c2, 1, 1, w=w.cv4)
        self.bn = TFBN(w.bn)
        self.act = lambda x: keras.activations.relu(x, alpha=0.1)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        y1 = self.cv3(self.m(self.cv1(inputs)))
        y2 = self.cv2(inputs)
        return self.cv4(self.act(self.bn(tf.concat((y1, y2), axis=3))))


class TFC3(keras.layers.Layer):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, w=None):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super(TFC3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c1, c_, 1, 1, w=w.cv2)
        self.cv3 = TFConv(2 * c_, c2, 1, 1, w=w.cv3)
        self.m = keras.Sequential([TFBottleneck(c_, c_, shortcut, g, e=1.0, w=w.m[j]) for j in range(n)])

    def call(self, inputs):
        return self.cv3(tf.concat((self.m(self.cv1(inputs)), self.cv2(inputs)), axis=3))


class TFSPP(keras.layers.Layer):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), w=None):
        super(TFSPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * (len(k) + 1), c2, 1, 1, w=w.cv2)
        self.m = [keras.layers.MaxPool2D(pool_size=x, strides=1, padding='SAME') for x in k]

    def call(self, inputs):
        x = self.cv1(inputs)
        return self.cv2(tf.concat([x] + [m(x) for m in self.m], 3))


class TFSPPF(keras.layers.Layer):
    # Spatial pyramid pooling-Fast layer
    def __init__(self, c1, c2, k=5, w=None):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = TFConv(c1, c_, 1, 1, w=w.cv1)
        self.cv2 = TFConv(c_ * 4, c2, 1, 1, w=w.cv2)
        self.m = keras.layers.MaxPool2D(pool_size=k, strides=1, padding='SAME')

    def call(self, inputs):
        x = self.cv1(inputs)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(tf.concat([x, y1, y2, self.m(y2)], 3))


class TFDetect(keras.layers.Layer):
    def __init__(self, nc=80, anchors=(), ch=(), imgsz=(640, 640), w=None):  # detection layer
        super(TFDetect, self).__init__()
        self.stride = tf.convert_to_tensor(w.stride.detach().numpy(), dtype=tf.float32)
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [tf.zeros(1)] * self.nl  # init grid
        self.anchors = tf.convert_to_tensor(w.anchors.detach().numpy(), dtype=tf.float32)
        self.anchor_grid = tf.reshape(self.anchors * tf.reshape(self.stride, [self.nl, 1, 1]),
                                      [self.nl, 1, -1, 1, 2])
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]
        self.training = False  # set to False after building model
        self.imgsz = imgsz
        for i in range(self.nl):
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            self.grid[i] = self._make_grid(nx, ny)

    def call(self, inputs):
        z = []  # inference output
        x = []
        for i in range(self.nl):
            x.append(self.m[i](inputs[i]))
            # x(bs,20,20,255) to x(bs,3,20x20,85)
            ny, nx = self.imgsz[0] // self.stride[i], self.imgsz[1] // self.stride[i]
            x[i] = tf.transpose(tf.reshape(x[i], [-1, ny * nx, self.na, self.no]), [0, 2, 1, 3])
            if isinstance(self, TFSegment):
                xy = (tf.sigmoid(x[i][..., 0:2]) * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                wh = (tf.sigmoid(x[i][..., 2:4]) * 2) ** 2 * self.anchor_grid[i]
                # Normalize xywh to 0-1 to reduce calibration error
                xy /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                wh /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                y = tf.concat([xy, wh, tf.sigmoid(x[i][..., 4:self.nc+5]), x[i][..., self.nc+5:]], -1)
            else:
                y = tf.sigmoid(x[i])
                xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                # Normalize xywh to 0-1 to reduce calibration error
                xy /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                wh /= tf.constant([[self.imgsz[1], self.imgsz[0]]], dtype=tf.float32)
                y = tf.concat([xy, wh, y[..., 4:]], -1)
            z.append(tf.reshape(y, [-1, 3 * ny * nx, self.no]))

        return tf.concat(z, 1), x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        # yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        # return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, ny * nx, 2]), dtype=tf.float32)


class TFSegment(TFDetect):
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), imgsz=(640, 640), w=None):
        super(TFSegment, self).__init__(nc, anchors, ch, imgsz, w)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = [TFConv2d(x, self.no * self.na, 1, w=w.m[i]) for i, x in enumerate(ch)]  # output conv

        self.proto = TFProto(ch[0], self.npr, self.nm, w=w.proto)  # protos
        self.detect = TFDetect.call

    def call(self, inputs):
        p = self.proto(inputs[0])
        x = self.detect(self, inputs)
        return x[0], p


class TFProto(keras.layers.Layer):
    def __init__(self, c1, c_=256, c2=32, w=None): # ch_in, number of protos, number of masks
        super(TFProto, self).__init__()
        self.cv1 = TFConv(c1, c_, k=3, w=w.cv1)
        self.upsample = TFUpsample(None, scale_factor=2, mode='nearest')
        self.cv2 = TFConv(c_, c_, k=3, w=w.cv2)
        self.cv3 = TFConv(c_, c2, k=1, w=w.cv3)

    def call(self, inputs):
        return self.cv3(self.cv2(self.upsample(self.cv1(inputs))))


class TFUpsample(keras.layers.Layer):
    def __init__(self, size, scale_factor, mode, w=None):  # warning: all arguments needed including 'w'
        super(TFUpsample, self).__init__()
        assert scale_factor == 2, "scale_factor must be 2"
        self.upsample = tf.keras.layers.UpSampling2D(size=(scale_factor, scale_factor), interpolation=mode)#lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2), method=mode)
        # self.upsample = keras.layers.UpSampling2D(size=scale_factor, interpolation=mode)
        # with default arguments: align_corners=False, half_pixel_centers=False
        # self.upsample = lambda x: tf.raw_ops.ResizeNearestNeighbor(images=x,
        #                                                            size=(x.shape[1] * 2, x.shape[2] * 2))

    def call(self, inputs):
        return self.upsample(inputs)


class TFConcat(keras.layers.Layer):
    def __init__(self, dimension=1, w=None):
        super(TFConcat, self).__init__()
        assert dimension == 1, "convert only NCHW to NHWC concat"
        self.d = 3

    def call(self, inputs):
        return tf.concat(inputs, self.d)


def parse_model(d, ch, model, imgsz):  # model_dict, input_channels(3)
    LOGGER.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m_str = m
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, BottleneckCSP, C3, DownC, SPPCSPC]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m in {Detect, Segment}:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
            args.append(imgsz)
        else:
            c2 = ch[f]

        tf_m = eval('TF' + m_str.replace('nn.', ''))
        m_ = keras.Sequential([tf_m(*args, w=model.model[i][j]) for j in range(n)]) if n > 1 \
            else tf_m(*args, w=model.model[i])  # module

        if m in {Detect, Segment}:
            torch_m_ = m(*args[:-1])  # module
        else:
            torch_m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module

        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in torch_m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return keras.Sequential(layers), sorted(save), nc