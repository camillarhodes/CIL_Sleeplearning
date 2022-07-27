# Multi-Scale Ensemble for Road Semantic Segmentation using Super-Resolution

Our model for semantic road segmentation which takes advantage of super-resolution to obtain a multi-scale ensemble. Done as a project for the "Computational Intelligence Lab" course at ETH Zürich.

[Kaggle Competition](https://www.kaggle.com/competitions/cil-road-segmentation-2022) (team: Sleeplearning)

## Brief description

Our inputs are 400 x 400 road images. We first train VDSR, a deep learning based super-resolution method, to obtain 800 x 800 inputs from our original inputs (corresponding training masks are resized using the opencv "resize" function). We then train four networks: {Unet-SCSE, DeepLabv3+} x {256, 512}, by training both architectures on 256 x 256 crops of 400 x 400 inputs, and on 512 x 512 crops of 800 x 800 inputs. The final prediction mask is obtained by averaging the predictions of the four networks.

## Setting up dependencies

Command for setting up anaconda environment (do this from the main project directory, as it contains the environment.yml file):

```console
$ conda env create -n RoadSegSR --file environment.yml
```

Command to activate the environment:

```console
$ conda activate RoadSegSR
```

Once you activated the environment, you can run the following command to add this environment as a notebook kernel:

```console
$ python -m ipykernel install --user --name=RoadSegSR
```

Now when using the provided notebooks, you can activate the kernel via: Kernel -> Change Kernel -> RoadSegSR in the dropdown menu.

For installing the dependecies with pip, use the following command instead (do this from the main project directory, as it contains the requirements.txt file)

Unix/MacOS:

```console
$ python3 -m pip install -r requirements.txt
```

Windows:

```console
$ py -m pip install -r requirements.txt
```

## Interacting with code & reproducing results

Interacting with our code can be done through two jupyter notebooks:

**train_vdsr.ipynb** contains code to train the VDSR model, apply it on the training images and store the results. This should be ran first, such that the enlarged training samples are obtained.

**train_segmentation.ipynb** contains code to train the actual segmentation models. To reproduce our final model, use this code in this notebook to train the four models we use for our final ensemble with the given specifications. You can also retrive our model parameters by loading the model with the appopriately labeled code. The notebook also contains code blocks that predict and store the inference masks of the model, as well as code to visualize the models results (run this to run obtain a comparison between the GT, ensemble and individual model masks like the one found in the report).

**(WIP)**
