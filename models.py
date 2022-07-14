import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torch import nn, optim, argmax, concat
from torchgeometry.losses.dice import dice_loss as dice
from sklearn.metrics import f1_score
from pytorch_hed.run import estimate

class EdgemapFusedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, img):
        edgemap = estimate(img)
        fused = concat([img,edgemap],dim=1)
        return self.model(fused)

    
class SegmentationModel(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
        super().__init__()
        self.model=model
        self.lr = lr
        
    def training_step(self, batch, batch_idx):
        img, mask = batch
        labels = mask[:,0,:,:].int().long()
        
        
        
        pred_mask = self.model(img)
        
        # bce = nn.BCEWithLogitsLoss()
        # bce_loss = bce(pred_mask, mask)
        # return bce_loss
        
        dice_loss = dice(pred_mask, labels)
        self.log('train_dice_loss', dice_loss, prog_bar=True)
        train_f1_score = f1_score(pred_mask.argmax(dim=1).reshape(-1), labels.reshape(-1))
        self.log('train_f1_score', train_f1_score, prog_bar=True)
        return dice_loss
    
    def validation_step(self, batch, batch_idx):
        img, mask = batch
        labels = mask[:,0,:,:].int().long()
        
        pred_mask = self.model(img)
        
        # bce = nn.BCEWithLogitsLoss()
        # bce_loss = bce(pred_mask, mask)
        # self.log('val_bce_loss', bce_loss, prog_bar=True)
        
        dice_loss = dice(pred_mask, labels)
        
        self.log('val_dice_loss', dice_loss, prog_bar=True)
        val_f1_score = f1_score(pred_mask.argmax(dim=1).reshape(-1), labels.reshape(-1))
        self.log('val_f1_score', val_f1_score, prog_bar=True)

        return dice_loss
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        pred_mask = self.forward(x)
        pred_mask = argmax(pred_mask, dim=1)
        return pred_mask
    
    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)
        
        
unet = smp.Unet(
    encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,
)

unet_4_channels = smp.Unet(
    encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,
)

edgemap_fused_unet = EdgemapFusedModel(unet_4_channels)

deeplabv3plus = smp.DeepLabV3Plus(
    encoder_name='resnet50',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,
)

def get_pl_model(model_name):
    if model_name == 'unet':
        return SegmentationModel(unet)
    if model_name == 'deeplabv3plus':
        return SegmentationModel(deeplabv3plus)
    if model_name == 'edgemap_fused_unet':
        return SegmentationModel(edgemap_fused_unet)
    raise NotImplementedError('Unsupported model')