import sys
import torch
import re
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import util.util as util


class ProjectionDiscriminator(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ndf
        input_nc = 3
        label_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        self.enc1 = nn.Sequential(norm_layer(nn.Conv2d(input_nc, nf, kernel_size=3, stride=2, padding=1)), nn.LeakyReLU(0.2, True))
        mult = 1
        for i in range(2, 6):
            self.add_module('enc' + str(i), nn.Sequential(norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                                      kernel_size=3, stride=2, padding=1)), nn.LeakyReLU(0.2, True)))

            mult *= 2
            mult = max(mult, nf*8)
            self.add_module('lat' + str(i), nn.Sequential(norm_layer(nn.Conv2d(opt.ngf * mult, nf*4,
                                                      kernel_size=3, stride=2, padding=1)), nn.LeakyReLU(0.2, True)))

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        for i in range(2, 5):
            self.add_module('final' + str(2),
                            nn.Sequential(norm_layer(nn.Conv2d(nf*4, nf*2, kernel_size=3, padding=1), opt), nn.LeakyReLU(0.2, True)))

        # shared True/False layer
        self.tf = nn.Conv2d(nf*2, 1, kernel_size=1)
        self.seg = nn.Conv2d(nf*2, nf*2, kernel_size=1) # do not need softmax
        self.embedding = nn.Conv2d(label_nc, nf*2, kernel_size=1)

    def forward(self, fake_and_real_img, segmap):
        feat11 = self.enc1(fake_and_real_img)
        feat12 = self.enc2(feat11)
        feat13 = self.enc3(feat12)
        feat14 = self.enc4(feat13)
        feat15 = self.enc5(feat14)
        feat25 = self.lat5(feat15)
        feat24 = self.up(feat25) + self.lat4(feat14)
        feat23 = self.up(feat24) + self.lat3(feat13)
        feat22 = self.up(feat23) + self.lat2(feat12)
        feat32 = self.final2(feat22)
        feat33 = self.final3(feat23)
        feat34 = self.final4(feat24)

        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)
        
        seg2 = self.seg(feat32)
        seg3 = self.seg(feat33)
        seg4 = self.seg(feat34)

        if self.opt.gan_matching_feats == 'basic':
            feats = [feat12, feat13, feat14, feat15]
        elif self.opt.gan_matching_feats == 'more':
            feats = [feat12, feat13, feat14, feat15, feat25, feat24, feat23, feat22]
        else:
            feats = [feat12, feat13, feat14, feat15, feat25, feat24, feat23, feat22, feat32, feat33, feat34]

        # calculate segmentation loss
        # segmentation map embedding
        segemb = self.embedding(segmap)
        # downsample
        segemb = F.avg_pool2d(segemb, kernel_size=2, stride=2)
        segemb2 = F.avg_pool2d(segemb, kernel_size=2, stride=2)
        segemb3 = F.avg_pool2d(segemb2, kernel_size=2, stride=2)
        segemb4 = F.avg_pool2d(segemb3, kernel_size=2, stride=2)

        # product
        pred2 += torch.mul(segemb2, seg2).sum(dim=1, keepdim=True)
        pred3 += torch.mul(segemb3, seg3).sum(dim=1, keepdim=True)
        pred4 += torch.mul(segemb4, seg4).sum(dim=1, keepdim=True)

        results = [pred2, pred3, pred4]

        return [feats, results]
