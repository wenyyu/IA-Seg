#! /usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import functools

import numpy as np

from configs.train_config import cfg

import time

def conv_downsample(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
    return layers

class CNN_PP(nn.Module):
    def __init__(self, in_channels=3):
        super(CNN_PP, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256,256),mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *conv_downsample(16, 32, normalization=True),
            *conv_downsample(32, 64, normalization=True),
            *conv_downsample(64, 128, normalization=True),
            *conv_downsample(128, 128),
            #*discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, cfg.num_filter_parameters, 8, padding=0),
        )

    def forward(self, img_input):
        self.Pr = self.model(img_input)
        self.filtered_image_batch = img_input
        filters = cfg.filters
        filters = [x(img_input, cfg) for x in filters]
        self.filter_parameters = []
        self.filtered_images = []

        for j, filter in enumerate(filters):
            # with tf.variable_scope('filter_%d' % j):
            # print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.',
            #       filter.get_short_name())
            # print('      filter_features:', self.Pr.shape)

            self.filtered_image_batch, filter_parameter = filter.apply(
                self.filtered_image_batch, self.Pr)
            self.filter_parameters.append(filter_parameter)
            self.filtered_images.append(self.filtered_image_batch)

            # print('      output:', self.filtered_image_batch.shape)
        return self.filtered_image_batch, self.filtered_images, self.Pr, self.filter_parameters



def DIP():
    model = CNN_PP()
    return model



