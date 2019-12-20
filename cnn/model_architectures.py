import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNM2048(nn.Module):
    def __init__(self, input_shape, dropout_rate=0.9, is_spatial=True):
        super(CNNM2048, self).__init__()
        self.in_channels = input_shape[1]
        self.dropout_rate = dropout_rate
        self.is_spatial = is_spatial

        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()

        # build the network
        self.build_module()

    def build_module(self):
        self.layer_dict['conv_1'] = nn.Conv2d(self.in_channels, 96, kernel_size = 7, stride = 2, padding = 0, bias=True)
        self.layer_dict['conv_2'] = nn.Conv2d(96 , 256, kernel_size=5, stride=2, padding=1, bias=True)
        self.layer_dict['conv_3'] = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer_dict['conv_4'] = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer_dict['conv_5'] = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)

        in_features = 12800 # = 512 * 5 * 5 # num_channels * height * width, from previous layer, flattened
        self.layer_dict['fc_1'] = nn.Linear(in_features=in_features, out_features=4096, bias=True)
        self.layer_dict['fc_2'] = nn.Linear(in_features=4096, out_features=2048, bias=True)
        self.layer_dict['logits'] = nn.Linear(in_features=2048, out_features=2, bias=True)

        # # focus training on conv layers
        # self.layer_dict['logits'] = nn.Linear(in_features=in_features, out_features=2, bias=True)

    def forward(self, x):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        # print(x.shape)
        x = F.relu(self.layer_dict['conv_1'](x))
        x = F.local_response_norm(x, size=5, alpha=0.0001, beta=0.75, k=2.0)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # print(x.shape)
        x = F.relu(self.layer_dict['conv_2'](x))
        if self.is_spatial:
            x = F.local_response_norm(x, size=5, alpha=0.0001, beta=0.75, k=2.0)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = F.relu(self.layer_dict['conv_3'](x))

        x = F.relu(self.layer_dict['conv_4'](x))

        x = F.relu(self.layer_dict['conv_5'](x))
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # print(x.shape)


        x = x.view(x.shape[0], -1)  # flatten x from (b, c, h, w) to (b, c*h*w)
        # print(x.shape)
        x = F.relu(self.layer_dict['fc_1'](x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # print(x.shape)
        x = F.relu(self.layer_dict['fc_2'](x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # print(x.shape)
        x = self.layer_dict['logits'](x)
        # print(x.shape)

        # # focus training on conv layers
        # x = x.view(x.shape[0], -1)  # flatten x from (b, c, h, w) to (b, c*h*w)
        # # print(x.shape)
        # x = self.layer_dict['logits'](x)
        # # print(x.shape)

        return x

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        ReLU, local_response_norm, max_pool, and dropout do not have parameters which would be required to set back.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

class TwoStreamNetwork(nn.Module):
    def __init__(self, input_shape, dropout_rate):
        """
        Initializes a convolutional network module object as given by Simonyan et al. 2014
        """
        super(TwoStreamNetwork, self).__init__()
        # set up class attributes useful in building the network and inference

        self.spatial_shape = (input_shape[0], 1, input_shape[2], input_shape[3])
        self.temporal_shape = input_shape
        self.dropout_rate = dropout_rate

        # build the network
        self.build_module()

    def build_module(self):
        self.spatial_stream = CNNM2048(self.spatial_shape, dropout_rate=self.dropout_rate, is_spatial=True)
        self.temporal_stream = CNNM2048(self.temporal_shape, dropout_rate=self.dropout_rate, is_spatial=False) # !


    def forward(self, x_spatial, x_temporal):
        """
        Forward propages the network given an input batch
        :param x: Inputs x_spatial (b, 1, h, w) and x_temporal (b, c, h, w) (where c = 90)
        :return: preds spatial (b, 2) and temporal (b, 2)

        """
        # x_spatial, x_temporal = x[0], x[1]
        x_spatial = self.spatial_stream.forward(x_spatial)
        x_temporal = self.temporal_stream.forward(x_temporal)
        return x_spatial, x_temporal


    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        self.spatial_stream.reset_parameters()
        self.temporal_stream.reset_parameters()

