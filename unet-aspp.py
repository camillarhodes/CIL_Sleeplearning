from torch import nn, cat
from torch.nn import functional as F
from typing import Optional, Union, List
from asppaux import ASPP, SeparableConv2d

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)



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
        self.conv1 = SeparableConv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )
        self.conv2 = SeparableConv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )
        
    def forward(self, x, skip=None):

        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            x = cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x




class UnetASPPDecoder(nn.Module):
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
        highres_out_channels = 48
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

    def forward(self, *features):
    
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[-4])
        concat_features = cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        
        #additions
        unetmap1 = self.unetblock1(fused_features, features[-5])
       
        return unetmap1


class UnetASPP(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (6,12,18)
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 2,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        if encoder_output_stride not in [8, 16]:
            raise ValueError("Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride))

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )

        self.decoder = UnetASPPDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
