import cv2
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.unet import Unet
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from sklearn.metrics import f1_score
from torch import argmax, concat, nn, no_grad, optim
from torchgeometry.losses.dice import dice_loss as dice

from asppaux import ASPP
from pytorch_hed_fork.run import Network as HED_model


class SegmentationModel(pl.LightningModule):
    """Wrapper class for segmentation models to generalize training and evaluating.

    Paramaters:
        seg_model : str
            The name of the model. Valid values are "unet", "unet++",
            "unet_scse", "unet++_scse", "unet_big", "hed_unet", "deeplabv3plus",
            "edgemap_fused_unet", "aspp".
        pretrained_weights : str, optional
            The name of the pretrained weights set to use.
        lr : float
            The learning rate.
    """

    def __init__(self, seg_model, pretrained_weights="imagenet", lr=1e-4):
        super().__init__()
        self.seg_model = get_seg_model(seg_model, pretrained_weights)
        self.lr = lr

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        img, mask = batch
        labels = mask[:, 0, :, :].int().long()

        pred_mask = self.seg_model(img)

        dice_loss = dice(pred_mask, labels)
        loss = dice_loss
        self.log("train_dice_loss", dice_loss, prog_bar=True)
        train_f1_score = f1_score(
            pred_mask.argmax(dim=1).reshape(-1).cpu(), labels.reshape(-1).cpu()
        )
        self.log("train_f1_score", train_f1_score, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        img, mask = batch
        labels = mask[:, 0, :, :].int().long()

        pred_mask = self.seg_model(img)

        if optimizer_idx == 0:
            dice_loss = dice(pred_mask, labels)
            loss = dice_loss
            self.log("val_dice_loss", dice_loss, prog_bar=True)
            val_f1_score = f1_score(
                pred_mask.argmax(dim=1).reshape(-1).cpu(), labels.reshape(-1).cpu()
            )
            self.log("val_f1_score", val_f1_score, prog_bar=True)

        return loss

    def forward(self, x):
        return self.seg_model(x)

    def predict_full_mask(self, x):
        """Generates the full mask.

        Because the models are trained on cropped images, we generate the full mask by combining multiple crops.
        """
        if not torch.is_tensor(x):
            x = torch.Tensor(x, device=self.device)

        assert len(x.shape) == 4 and x.size(1) == 3

        img_size = x.size(2)

        if img_size == 400:
            crop_size = 256
        elif img_size == 800:
            crop_size = 512
        else:
            raise ValueError("Unsupported input image size")

        second_crop_start_idx = img_size - crop_size
        crop_overlap = crop_size - second_crop_start_idx

        pred_mask_1 = self.predict(x[:, :, :crop_size, :crop_size])
        pred_mask_2 = self.predict(x[:, :, -crop_size:, :crop_size])
        pred_mask_3 = self.predict(x[:, :, :crop_size, -crop_size:])
        pred_mask_4 = self.predict(x[:, :, -crop_size:, -crop_size:])

        pred_mask = torch.zeros((x.size(0), 1, img_size, img_size))

        pred_mask[:, :, :crop_size, :crop_size] = pred_mask_1
        pred_mask[:, :, -crop_size:, :crop_size] = pred_mask_2
        pred_mask[:, :, :crop_size, -crop_size:] = pred_mask_3
        pred_mask[:, :, -crop_size:, -crop_size:] = pred_mask_4

        pred_mask[
            :, :, :second_crop_start_idx, second_crop_start_idx:crop_size
        ] = 0.5 * (
            pred_mask_1[:, :, :second_crop_start_idx, second_crop_start_idx:crop_size]
            + pred_mask_3[:, :, :second_crop_start_idx, :crop_overlap]
        )
        pred_mask[
            :, :, second_crop_start_idx:crop_size, :second_crop_start_idx
        ] = 0.5 * (
            pred_mask_1[:, :, second_crop_start_idx:crop_size, :second_crop_start_idx]
            + pred_mask_2[:, :, :crop_overlap, :second_crop_start_idx]
        )
        pred_mask[:, :, crop_size:, second_crop_start_idx:crop_size] = 0.5 * (
            pred_mask_2[:, :, crop_overlap:crop_size, second_crop_start_idx:crop_size]
            + pred_mask_4[:, :, crop_overlap:crop_size, :crop_overlap]
        )
        pred_mask[:, :, second_crop_start_idx:crop_size, crop_size:] = 0.5 * (
            pred_mask_3[:, :, second_crop_start_idx:crop_size, crop_overlap:crop_size]
            + pred_mask_4[:, :, :crop_overlap, crop_overlap:crop_size]
        )

        pred_mask[
            :, :, second_crop_start_idx:crop_size, second_crop_start_idx:crop_size
        ] = 0.25 * (
            pred_mask_1[
                :, :, second_crop_start_idx:crop_size, second_crop_start_idx:crop_size
            ]
            + pred_mask_2[:, :, :crop_overlap, second_crop_start_idx:crop_size]
            + pred_mask_3[:, :, second_crop_start_idx:crop_size, :crop_overlap]
            + pred_mask_4[:, :, :crop_overlap, :crop_overlap]
        )

        if img_size == 800:
            pred_mask_np = pred_mask.cpu().numpy()[0]
            pred_mask_np_resized = cv2.resize(
                255 * pred_mask_np.transpose(1, 2, 0),
                (400, 400),
                interpolation=cv2.INTER_AREA,
            )
            pred_mask = torch.Tensor(pred_mask_np_resized / 255, device=x.device)[
                None, None, :, :
            ]

        return pred_mask

    @no_grad()
    def predict(self, x):
        if self.training:
            self.eval()

        pred_mask = self.seg_model(x).softmax(dim=1)[:, 1:2, :, :]
        return pred_mask

    def configure_optimizers(self):
        return (optim.Adam(self.seg_model.parameters(), lr=self.lr),)


class EdgemapFusedUnet(Unet):
    def __init__(
        self,
        encoder_depth=5,
        decoder_use_batchnorm=True,
        decoder_attention_type=None,
        decoder_channels=(256, 128, 64, 32, 16),
        *args,
        **kwargs
    ):

        if "encoder_depth" not in kwargs:
            kwargs["encoder_depth"] = encoder_depth
        if "decoder_channels" not in kwargs:
            kwargs["decoder_channels"] = decoder_channels

        # import pdb; pdb.set_trace()
        super().__init__(*args, **kwargs)

        # reflect the two encoders
        encoder_out_channels = [x * 2 for x in self.encoder.out_channels]
        encoder_out_channels[0] = 4  # single channel edgemap and the RGB

        self.decoder = UnetDecoder(
            encoder_channels=encoder_out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if kwargs["encoder_name"].startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.edgemap_encoder = get_encoder(
            kwargs["encoder_name"],
            in_channels=1,
            # depth=kwargs['encoder_depth'],
            weights=kwargs["encoder_weights"],
            # weights=None,
        )
        self.hed_model = HED_model().eval()
        for param in self.hed_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features = self.encoder(x)

        edgemap = self.hed_model(x)
        edgemap_features = self.edgemap_encoder(edgemap)

        combined_features = [
            concat([feature, edgemap_feature], dim=1)
            for feature, edgemap_feature in zip(features, edgemap_features)
        ]

        decoder_output = self.decoder(*combined_features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks


def get_seg_model(model_name, encoder_weights="imagenet"):
    """Returns the specified segmentation model.

    Parameters:
        model_name : str
            The name of the model. Valid values are "unet", "unet++",
            "unet_scse", "unet++_scse", "unet_big", "hed_unet", "deeplabv3plus",
            "edgemap_fused_unet", "aspp".
        encoder_weights : str, optional
            The name of the pretrained weights set to use.
    """
    if model_name == "unet":
        return smp.Unet(
            # encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name="vgg19",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,
        )
    if model_name == "unet++":
        return smp.UnetPlusPlus(
            # encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name="vgg19",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,
        )
    if model_name == "unet_scse":
        return smp.Unet(
            # encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name="vgg19",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,
            decoder_attention_type="scse",
        )
    if model_name == "unet++_scse":
        return smp.UnetPlusPlus(
            # encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name="vgg19",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,
            decoder_attention_type="scse",
        )
    if model_name == "unet_big":
        return smp.Unet(
            # encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name="vgg19",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            decoder_channels=tuple(x * 2 for x in (256, 128, 64, 32, 16)),
            decoder_attention_type="scse",
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,
        )
    if model_name == "hed_unet":
        return smp.Unet(
            # encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name="vgg19",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=4,
        )
    if model_name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            # encoder_name='vgg19',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name="resnet101",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,
        )
    if model_name == "edgemap_fused_unet":
        return EdgemapFusedUnet(
            encoder_name="vgg19",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            # encoder_name='mobilenet_v2',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,
        )
    if model_name == "aspp":
        return ASPP(in_channels=3, out_channels=1, atrous_rates=4)
    raise NotImplementedError("Unsupported model")
