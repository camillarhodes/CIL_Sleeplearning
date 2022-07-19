import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch import argmax, nn, optim
from torchgeometry.losses.dice import dice_loss as dice
from sklearn.metrics import f1_score

from drn import get_drnseg_model


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, lr=1e-5):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        img, mask = batch
        labels = mask[:, 0, :, :].int().long()

        pred_mask = self.model(img)

        # bce = nn.BCEWithLogitsLoss()
        # bce_loss = bce(pred_mask, mask)
        # return bce_loss

        dice_loss = dice(pred_mask, labels)
        return dice_loss

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        labels = mask[:, 0, :, :].int().long()

        pred_mask = self.model(img)

        # bce = nn.BCEWithLogitsLoss()
        # bce_loss = bce(pred_mask, mask)
        # self.log('val_bce_loss', bce_loss, prog_bar=True)

        dice_loss = dice(pred_mask, labels)
        self.log('val_dice_loss', dice_loss, prog_bar=True)

        return dice_loss

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        pred_mask = self.forward(x)
        pred_mask = argmax(pred_mask, dim=1)
        return pred_mask

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)


baseline_model = smp.Unet(
    encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # use `imagenet` pre-trained weights for encoder initialization
    encoder_weights='imagenet',
    # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    in_channels=3,
    classes=2
)


def create_baseline_unet_model(is_unet_plusplus=False, encoder_depth=5, encoder_name='resnet34', encoder_weights='imagenet', use_attention=False, is_greyscale=False):
    Unet = smp.UnetPlusPlus if is_unet_plusplus else smp.Unet
    return SegmentationModel(Unet(
        encoder_name=encoder_name,
        encoder_depth=encoder_depth,
        encoder_weights=encoder_weights,
        decoder_attention_type='scse' if use_attention else None,
        in_channels=1 if is_greyscale else 3,
        classes=2,
    ))


def get_pl_model(model_name):
    if model_name == 'baseline':
        return SegmentationModel(baseline_model)
    if model_name == 'drn_seg':
        return SegmentationModel(get_drnseg_model())
    raise NotImplementedError('Unsupported model')
