import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torch import nn, optim


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, lr=1e-5):
        super().__init__()
        self.model=model
        self.lr = lr
        
    def training_step(self, batch, batch_idx):
        img, mask = batch
        pred_mask = self.model(img)
        bce = nn.BCEWithLogitsLoss()
        bce_loss = bce(pred_mask, mask)
        return bce_loss
    
    def validation_step(self, batch, batch_idx):
        img, mask = batch
        pred_mask = self.model(img)
        bce = nn.BCEWithLogitsLoss()
        bce_loss = bce(pred_mask, mask)
        self.log('val_bce_loss', bce_loss, prog_bar=True)

        return bce_loss
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)
        
        
baseline_model = smp.Unet(
    encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
)

def get_pl_model(model_name):
    if model_name == 'baseline':
        return SegmentationModel(baseline_model)
    raise NotImplementedError('Unsupported model')