# Multi-Scale Ensemble for Road Semantic Segmentation using Super-Resolution

Our model for semantic road segmentation which takes advantage of super-resolution to obtain a multi-scale ensemble. Done as a project for the "Computational Intelligence Lab" course at ETH Zürich.

[Kaggle Competition](https://www.kaggle.com/competitions/cil-road-segmentation-2022) (team: Sleeplearning)

## Brief description

Our inputs are 400 x 400 road images. We first train VDSR, a deep-learning-based super-resolution method, to obtain 800 x 800 inputs from our original inputs (corresponding training masks are resized using the opencv "resize" function). We then train four networks: {U-Net, DeepLabv3+} x {256, 512}, by training both architectures on 256 x 256 crops of 400 x 400 inputs, and on 512 x 512 crops of 800 x 800 inputs. The final prediction mask is obtained by averaging the predictions of the four networks.

## Setting up dependencies

After cloning the repository, you should pull the necessary git submodules using the command

```console
$ git submodule update --init --recursive
```

The following commands require [anaconda](https://www.anaconda.com). You should run these commands from the root project directory.

First, set up the conda environment:

```console
$ conda env create -n RoadSegSR --file environment.yml
```

Then, activate the environment:

```console
$ conda activate RoadSegSR
```

Once you have activated the environment, you can run the following command to add this environment as a notebook kernel:

```console
$ python -m ipykernel install --user --name=RoadSegSR
```

Now, when using the provided notebooks, you can activate the kernel via: Kernel -> Change Kernel -> RoadSegSR in the dropdown menu.

## Interacting with code & reproducing results

Interacting with our code can be done through two jupyter notebooks:

**train_vdsr.ipynb** contains the code to train the VDSR model, apply it on the training images and store the results. The whole notebook **must** be ran before running other code as it creates the upscaled dataset for the models which are trained on VDSR-upscaled images.

**segmentation_pipeline.ipynb** contains code for training, evaluating and generating the submission files for the models presented in the paper. We provide the pipeline and configurations used to test U-Lab-MS and other models during ablation testing. Running all cells sequentially is enough to generate the Dice loss, F1 score, output masks and the Kaggle submission file, no further input is required. The models are placed in separate sections which names correspond with the model names in Table 1 of our paper. For the non-ensemble methods, we use the train_dice_loss as the training Dice loss and train_f1_score as the training F1 score from the training process. The outputted masks are placed in `./data/test/{model_name}/`, and the Kaggle submission file is placed in `./kaggle_submissions/{model_name}.csv`. For the ensemble methods, we calculate the training Dice loss and training F1 score after training and print it out while running the `perform_ensemble_pipeline` method. The outputted masks are placed in `./data/test/{ensemble_name}/`, and the Kaggle submission file is placed in `./kaggle_submissions/{ensemble_name}.csv`.

<!-- contains code to train the actual segmentation models. To reproduce our final model, use this code in this notebook to train the four models we use for our final ensemble with the given specifications. You can also retrive our model parameters by loading the model with the appopriately labeled code. The notebook also contains code blocks that predict and store the inference masks of the model, as well as code to visualize the models results (run this to run obtain a comparison between the GT, ensemble and individual model masks like the one found in the report). -->

**plot_losses.ipynb** contains the code for generating the plots of the training Dice loss and training F1 score. The data used was manually downloaded from Tensorboard using the appropriate tfevents under `./logs/{model_name or ensemble_name}/`, renamed accordingly, and placed in `./model_loss/data/`. For convenience, we include the downloaded files in our submission. The plots are saved in `./loss_plots/`.

**models.py** provides a common wrapper that is used in **segmentation_pipeline.ipynb** for all models in order to generalize training and evaluating.

**datasets.py** provides code that converts raw input datasets into PyTorch dataloaders with augmentations.

**augmentations.py** provides the various types of augmentations that we used during our experiments.

**unet-aspp.py** contains code for the U-Net-ASPP model.

**asppaux.py** contains auxiliary nn.Module subclasses that are used as components of U-Net-ASPP.

**(WIP)**
