import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.unet import Unet
import pytorch_lightning as pl
from torch import nn, optim, argmax, concat, no_grad
import torch
from torchgeometry.losses.dice import dice_loss as dice
from sklearn.metrics import f1_score
from pytorch_hed_fork.run import Network as HED_model

    
class Discriminator(nn.Module):
        def __init__(self, lr=2e-5):
            super().__init__()
            self.lr = lr
            
            conv_layers = [
                nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2),
                nn.InstanceNorm2d(16),
                nn.Conv2d(64, 256, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2),
                nn.InstanceNorm2d(32),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2),
                nn.InstanceNorm2d(64),
                nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2),
                nn.InstanceNorm2d(64),
                nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2),
                nn.InstanceNorm2d(64),
                nn.Conv2d(256, 64, kernel_size=3, padding=1, stride=2),
            ]
            fc_layers = [
                    nn.Linear(1024, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
            ]
            self.conv_layers = nn.Sequential(*conv_layers)
            self.fc_layers = nn.Sequential(*fc_layers)
            
        def forward(self, mask):
            # d_input_mask = torch.softmax(mask,dim=1)[:,1:2,:,:] if mask.size(1) == 2 else mask/2
            d_input_mask = mask
            d_conv_output = self.conv_layers(d_input_mask)
            return self.fc_layers(d_conv_output.view(d_conv_output.size(0), -1))
            # return self.model(mask)
     
    
class SegmentationModel(pl.LightningModule):
    def __init__(self, seg_model, lr=1e-4, discriminate=False):
        super().__init__()
        self.seg_model=get_seg_model(seg_model)
        self.lr = lr
        self.discriminate = discriminate
        if self.discriminate:
            self.discriminator = Discriminator()
            
        
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        img, mask = batch
        labels = mask[:,0,:,:].int().long()
        
        # canny = canny[:,0,:,:,].int().long()
        
        
        pred_mask = self.seg_model(img)
        
        # pred_canny = pred[:,2:4,:,:]
        
        # pred_mask_softmaxed = torch.softmax(pred_mask,dim=1)[:,1:2,:,:] 
        
        if optimizer_idx == 0:
            dice_loss = dice(pred_mask, labels)
            loss = dice_loss
            self.log('train_dice_loss', dice_loss, prog_bar=True)
            train_f1_score = f1_score(pred_mask.argmax(dim=1).reshape(-1).cpu(), labels.reshape(-1).cpu())
            self.log('train_f1_score', train_f1_score, prog_bar=True)
            
            # dice_loss_canny = dice(pred_canny, canny)
            # self.log('train_dice_loss_canny', dice_loss, prog_bar=False)
            
            # train_weighted_mse_loss = (labels * ((pred_mask_softmaxed - 1)**2)).mean()
            # self.log('train_weighted_mse_loss', train_weighted_mse_loss, prog_bar=False)
            
            if self.discriminate:
                train_generator_loss = torch.log(1 - self.discriminator(pred_mask_softmaxed)).mean()
                loss += train_generator_loss*0.15
                self.log('train_generator_loss', train_generator_loss, prog_bar=False)
                
            # loss += 0 * train_weighted_mse_loss
            return loss
        
        if optimizer_idx == 1:
            train_discriminator_loss = (-torch.log(self.discriminator(mask)) - torch.log(1 - self.discriminator(pred_mask_softmaxed))).mean()
            self.log('train_discriminator_loss', train_discriminator_loss, prog_bar=False)
            return train_discriminator_loss
    
    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        img, mask = batch
        labels = mask[:,0,:,:].int().long()
        # canny = canny[:,0,:,:,].int().long()
        
        pred_mask = self.seg_model(img)
        
        
        # pred_mask_softmaxed = torch.softmax(pred_mask,dim=1)[:,1:2,:,:] 
        
        if optimizer_idx == 0:
            dice_loss = dice(pred_mask, labels)
            loss = dice_loss
            self.log('val_dice_loss', dice_loss, prog_bar=True)
            val_f1_score = f1_score(pred_mask.argmax(dim=1).reshape(-1).cpu(), labels.reshape(-1).cpu())
            self.log('val_f1_score', val_f1_score, prog_bar=True)
            
            
            # dice_loss_canny = dice(pred_canny, canny)
            # self.log('val_dice_loss_canny', dice_loss, prog_bar=False)
            
            # val_weighted_mse_loss = (labels * ((pred_mask_softmaxed - 1)**2)).mean()
            # self.log('val_weighted_mse_loss', val_weighted_mse_loss, prog_bar=False)
            
            # if self.discriminate:
            #     val_generator_loss = torch.log(1 - self.discriminator(pred_mask_softmaxed)).mean()
            #     loss += val_generator_loss*0.15
            #     self.log('val_generator_loss', val_generator_loss, prog_bar=False)
                
#         if optimizer_idx == 1:
                                        
#             val_discriminator_loss = (-torch.log(self.discriminator(mask)) - torch.log(1 - self.discriminator(softmaxed))).mean()
#             self.log('val_discriminator_loss', val_discriminator_loss, prog_bar=False)
#             return val_discriminator_loss
        
#         loss += 0 * val_weighted_mse_loss
                
        return loss
    
    def forward(self, x):
        return self.seg_model(x)
    
    def predict_full_mask(self, x):
        
        if not torch.is_tensor(x):
            x = torch.Tensor(x, device=self.device)
            
        assert len(x.shape) == 4 and x.size(1) == 3
        
        img_size = x.size(2)
        
        if img_size == 400:
            crop_size = 256 
        elif img_size == 800:
            crop_size = 512
        else:
            raise ValueError('Unsupported img size')
            
        second_crop_start_idx = img_size - crop_size
        crop_overlap = crop_size - second_crop_start_idx
            
        pred_mask_1=self.predict(x[:,:,:crop_size,:crop_size])
        pred_mask_2=self.predict(x[:,:,-crop_size:,:crop_size])
        pred_mask_3=self.predict(x[:,:,:crop_size,-crop_size:])
        pred_mask_4=self.predict(x[:,:,-crop_size:,-crop_size:])
        
        pred_mask=torch.zeros((x.size(0), 1, img_size, img_size))
        
        pred_mask[:,:,:crop_size,:crop_size] = pred_mask_1
        pred_mask[:,:,-crop_size:,:crop_size] = pred_mask_2
        pred_mask[:,:,:crop_size,-crop_size:] = pred_mask_3
        pred_mask[:,:,-crop_size:,-crop_size:] = pred_mask_4
        
        pred_mask[:,:,:second_crop_start_idx,second_crop_start_idx:crop_size] = 0.5*(pred_mask_1[:,:,:second_crop_start_idx,second_crop_start_idx:crop_size] + pred_mask_3[:,:,:second_crop_start_idx,:crop_overlap])
        pred_mask[:,:,second_crop_start_idx:crop_size,:second_crop_start_idx] = 0.5*(pred_mask_1[:,:,second_crop_start_idx:crop_size,:second_crop_start_idx] + pred_mask_2[:,:,:crop_overlap,:second_crop_start_idx])
        pred_mask[:,:,crop_size:,second_crop_start_idx:crop_size] = 0.5*(pred_mask_2[:,:,crop_overlap:crop_size,second_crop_start_idx:crop_size] + pred_mask_4[:,:,crop_overlap:crop_size,:crop_overlap])
        pred_mask[:,:,second_crop_start_idx:crop_size,crop_size:] = 0.5*(pred_mask_3[:,:,second_crop_start_idx:crop_size,crop_overlap:crop_size] + pred_mask_4[:,:,:crop_overlap,crop_overlap:crop_size])
        
        pred_mask[:,:,second_crop_start_idx:crop_size,second_crop_start_idx:crop_size] = 0.25*(pred_mask_1[:,:,second_crop_start_idx:crop_size,second_crop_start_idx:crop_size]+pred_mask_2[:,:,:crop_overlap,second_crop_start_idx:crop_size]+pred_mask_3[:,:,second_crop_start_idx:crop_size,:crop_overlap]+pred_mask_4[:,:,:crop_overlap,:crop_overlap])
        
        return pred_mask
    
    
    @no_grad()
    def predict(self, x):
        if self.training:
            self.eval()
            
        pred_mask = self.seg_model(x).softmax(dim=1)[:,1:2,:,:]
        # pred_mask = argmax(pred_mask, dim=1)
        return pred_mask
    
#     def on_load_checkpoint(self, checkpoint):
#         state_dict = {}
#         for (k,v) in checkpoint['state_dict'].items():
#             if 'model.' in k and 'model.model.' not in k:
#                 state_dict[k.replace('model.','model.model.')] = v
#         checkpoint['state_dict']=state_dict
    
    def configure_optimizers(self):
        if not self.discriminate:
#             params_dicts = []
#             params_dicts.append(
#                 {'params': self.model.encoder.parameters(), 'lr':self.lr}
#             )
#             params_dicts.append(
#                 {'params': self.model.decoder.parameters(), 'lr':self.lr}
#             )
            
            
#             if isinstance(self.model, EdgemapFusedUnet):
#                 print('EdgemapFusedUnet, setting different lrs')
#                 params_dicts.append(
#                     {'params': self.model.edgemap_encoder.parameters(), 'lr':self.lr*10}
#                 )
#                 # params_dicts.append(
# #                     {'params': self.model.hed_model.parameters(), 'lr':self.lr*10}
#                 # )
#             return optim.Adam(params_dicts)
            return optim.Adam(self.seg_model.parameters(), lr=self.lr),
        
        return [
            optim.Adam(self.seg_model.parameters(), lr=self.lr),
            optim.Adam(self.discriminator.parameters(), lr=self.discriminator.lr),
        ]
                
    

            
class EdgemapFusedUnet(Unet):
    
    def __init__(self, encoder_depth = 5, decoder_use_batchnorm = True, decoder_attention_type = None, decoder_channels = (256, 128, 64, 32, 16), *args, **kwargs):
        
        if 'encoder_depth' not in kwargs:
            kwargs['encoder_depth']=encoder_depth
        if 'decoder_channels' not in kwargs:
            kwargs['decoder_channels']=decoder_channels
            
        # import pdb; pdb.set_trace()
        super().__init__(*args, **kwargs)
        
        # reflect the two encoders
        encoder_out_channels = [x*2 for x in self.encoder.out_channels]
        encoder_out_channels[0] = 4 # single channel edgemap and the RGB
        
        self.decoder = UnetDecoder(
            encoder_channels=encoder_out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if kwargs['encoder_name'].startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )
        
        
        self.edgemap_encoder = get_encoder(
            kwargs['encoder_name'],
            in_channels=1,
            # depth=kwargs['encoder_depth'],
            weights=kwargs['encoder_weights'],
            # weights=None,
        )
        self.hed_model = HED_model().eval()
        for param in self.hed_model.parameters():
            param.requires_grad=False
        
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        
        features = self.encoder(x)
        
        edgemap = self.hed_model(x)
        edgemap_features = self.edgemap_encoder(edgemap)
        
        
        combined_features = [concat([feature, edgemap_feature],dim=1) for feature, edgemap_feature in zip(features, edgemap_features)]
        
        decoder_output = self.decoder(*combined_features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks
        
        




# fpn = smp.FPN(
#     encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#     encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
#     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#     classes=2,
# )


# # edgemap_fused_unet = EdgemapFusedModel(unet)



def get_seg_model(model_name):
    if model_name == 'unet':
        return smp.Unet(
            # encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name='vgg19',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,
        )
    if model_name == 'unet_big':
        return smp.Unet(
            # encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name='vgg19',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            decoder_channels = tuple(x*2 for x in (256, 128, 64, 32, 16)),
            decoder_attention_type='scse',
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,
        )
    if model_name == 'hed_unet':
        return smp.Unet(
            # encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name='vgg19',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=4,
        )
    if model_name == 'unet_d':
        return smp.Unet(
            # encoder_name='resnet34',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name='vgg19',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,
        )
    
    if model_name == 'deeplabv3plus':
        return smp.DeepLabV3Plus(
            # encoder_name='vgg19',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_name='resnet101',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,
        )
    if model_name == 'edgemap_fused_unet':
        return EdgemapFusedUnet(
            encoder_name='vgg19',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            # encoder_name='mobilenet_v2',        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights='imagenet',     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,
        )
    raise NotImplementedError('Unsupported model')