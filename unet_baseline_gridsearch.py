from config import args
from pytorch_lightning import Trainer, loggers
from models import create_baseline_unet_model
from augmentations import get_transforms
from datasets import get_train_val_dataloaders


def test_hyperparams(train_dataloader, val_dataloader, max_epochs, is_unet_plusplus, encoder_depth, encoder_name, encoder_weights, use_attention, is_greyscale=False):
    model = create_baseline_unet_model(is_unet_plusplus=is_unet_plusplus,
                                       encoder_depth=encoder_depth,
                                       encoder_name=encoder_name,
                                       encoder_weights=encoder_weights,
                                       use_attention=use_attention,
                                       is_greyscale=is_greyscale)

    model = model.train()
    # args.gpus = None # Remove this line if you actually have gpus

    tb_logger = loggers.TensorBoardLogger(save_dir="logs/")
    trainer = Trainer(gpus=args.gpus,
                      max_epochs=max_epochs,
                      logger=tb_logger,
                      # accelerator=args.accelerator,
                      resume_from_checkpoint=args.checkpoint_path
                      )
    trainer.fit(model, train_dataloader, val_dataloader)

    result = trainer.validate(model, val_dataloader, verbose=False)

    return result[0]['val_dice_loss']


def format_write_string(val_score, max_epochs, is_unet_plusplus, encoder_depth, encoder_name, encoder_weights, use_attention):
    return f"{{val_score: {val_score}, max_epochs: {max_epochs}, is_unet_plusplus: {is_unet_plusplus}, encoder_depth: {encoder_depth}, encoder_name: {encoder_name}, encoder_weights: {encoder_weights}, use_attention: {use_attention} }}\n"


if __name__ == '__main__':
    transform = get_transforms('c')

    train_dataloader, val_dataloader = get_train_val_dataloaders(
        transform=transform, include_massachusetts=False, num_workers=1)

    val_dataloader.dataset.transform = None  # don't augment validation set

    is_unet_plusplus = False  # Use a unet variant with a more complex decoder
    encoder_depth = 5  # Valid values are 3, 4 and 5
    encoder_name = 'resnet34'
    encoder_weights = 'imagenet'
    use_attention = True  # False - use no attention, True - use scse attention
    max_epochs = 1

    for is_unet_plusplus in [False, True]:
        for encoder_depth in [5]:
            for use_attention in [False, True]:
                for max_epochs in [10, 50, 100]:
                    if not is_unet_plusplus:
                        if not use_attention:
                            continue
                        if max_epochs != 100:
                            continue
                    val_dice_loss = test_hyperparams(
                        train_dataloader, val_dataloader, max_epochs, is_unet_plusplus, encoder_depth, encoder_name, encoder_weights, use_attention)
                    output_string = format_write_string(
                        val_dice_loss, max_epochs, is_unet_plusplus, encoder_depth, encoder_name, encoder_weights, use_attention)
                    with open("unet_gridsearch_results.out", 'a') as f:
                        f.write(output_string)
                        f.flush()
