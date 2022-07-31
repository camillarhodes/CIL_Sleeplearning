import torch
from torch import nn
from torch.nn import functional as F

from asppaux import ASPP, SeparableConv2d

from torch import nn, optim, argmax, concat, no_grad
from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.base import modules as md


try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None

import GPUtil


class Conv2dDilatedReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
        dilation = 1
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dDilatedReLU, self).__init__(conv, bn, relu)


class ResDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = SeparableConv2d(#Conv2dDilatedReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )
        self.conv2 = SeparableConv2d(#Conv2dDilatedReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        #self.skip =  nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1)
        #self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
    def forward(self, x, skip=None):

        x = F.interpolate(x, scale_factor=2, mode="nearest") #self.upsample(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            #res = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)

        #x = F.interpolate(x, scale_factor=2, mode="nearest") #self.upsample(x)
        # if skip is not None:
            # return x+res
        return x




class DeepLabV3Plus3Decoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        out_channels=256,
        atrous_rates=(12, 24, 36),
        output_stride=16,
    ):
        super().__init__()
        if output_stride not in {8, 16}:
            raise ValueError("Output stride should be 8 or 16, got {}.".format(output_stride))

        self.out_channels = out_channels
        self.output_stride = output_stride

        self.aspp = nn.Sequential(
            ASPP(encoder_channels[-1], out_channels, atrous_rates, separable=True),
            SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 2 if output_stride == 8 else 4
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[-4]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(highres_in_channels, highres_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.unetblock1 = ResDecoderBlock(in_channels=128, skip_channels=encoder_channels[-5], out_channels=64, use_batchnorm=True)
        #self.unetblock2 = ResDecoderBlock(in_channels=128, skip_channels=0, out_channels=64, use_batchnorm=True)

        #self.unetblock1 = ResDecoderBlock(in_channels=64, skip_channels=encoder_channels[-5], out_channels=32)
        #self.upsample = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        # self.unetblock2 = nn.Sequential(
            # nn.Conv2d(32, 16, kernel_size=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU())   #ResDecoderBlock(in_channels=64, skip_channels=0, out_channels=16)

    def forward(self, *features):
    
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        
        #additions
        unetmap1 = self.unetblock1(fused_features, features[-5])
        #unetmap2 = self.unetblock2(unetmap1, None)#F.interpolate(unetmap1, scale_factor=2, mode="nearest")

       
        return unetmap1 #fused_features

