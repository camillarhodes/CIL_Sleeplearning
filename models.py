import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from torch import nn, optim, argmax
from torchgeometry.losses.dice import dice_loss as dice
from sklearn.metrics import f1_score

    
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
        
        
baseline_model = smp.Unet(
    encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,
    # activation='softmax'
)

def get_pl_model(model_name):
    if model_name == 'baseline':
        return SegmentationModel(baseline_model)
    raise NotImplementedError('Unsupported model')