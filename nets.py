# -*- coding: utf-8 -*-
# @Time   : 19-4-25 上午10:59
# @Author : TerYang
# @contact : adau22@163.com ============================
# My github:https://github.com/TerYang/              ===
# Copyright: MIT License                             ===
# Good good study,day day up!!                       ===
# ======================================================
import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
# from dataloader import dataloader
# from readDataToGAN import *

class generator(nn.Module):
    """
       convtranspose2d
        #   1.  s =1
        #     o' = i' - 2p + k - 1
        #   2.  s >1
        # o = (i-1)*s +k-2p+n
        # n =  output_padding,p=padding,i=input dims,s=stride,k=kernel
    """
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

        self.fc = nn.Sequential(
            nn.Linear(256, 3*4*512),
            nn.ReLU(),
        )
    def forward(self, input):
        x = self.fc(input)
        # x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = x.view(-1, 512, 4, 3)
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    # input 64*48
    def __init__(self, input_dim=1, output_dim=1, input_size=64):
        super(discriminator,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            # nn.Conv2d(self.input_dim, 8, (2,1), (2,1)),#torch.Size([64, 8, 32, 48])
            nn.Conv2d(self.input_dim, 64, (4,5), (2,1),(1,2)),#torch.Size([64, 8, 32, 48])
            nn.ReLU(),
            # nn.Conv2d(8, 16, (2, 1), (2, 1)),
            nn.Conv2d(64, 128, (4, 5), (2, 1), (1, 2)),  # torch.Size([64, 8, 32, 48])

            # nn.Conv2d(8, 16, (4,1), (2,1),(1,0)),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16*(16 * 3), 1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * 16*(16 * 3))
        x = self.fc(x)
        return x

# t = torch.ones((64,1,64,48))
# print(t.size())
# g = discriminator(1,1,64)
# l = g(t)
# print(l.size())

# qq = torch.randn((64,256))
# g = generator()
# l = g(qq)
# print(l.size())
