import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '--seed',
    type=int,
    default=42,
    help='Fix a seed'
)
parser.add_argument(
    '--encoder-name',
    type=str,
    default='resnet34',
    help='Encoder backbone for segmentation model'
)
parser.add_argument(
    '--encoder-weights',
    type=str,
    default='imagenet',
    help='The dataset on which pretrained backbone weights are optimized'
)
parser.add_argument(
    '--preprocess-like-pretraining',
    type=bool,
    default='False',
    help='Whether to apply the preprocessing function from the pretraining phase'
)
parser.add_argument(
    '--gpus',
    type=list,
    default=[0],
    help='GPUs to train on'
)
parser.add_argument(
    '--epochs',
    type=int,
    default=20,
    help='Number of epochs'
)
parser.add_argument(
    '--checkpoint-path',
    type=str,
    default=None,
)
    
args, _ = parser.parse_known_args()
