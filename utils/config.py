"""Graph matching config system."""

import os
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:

cfg = __C

__C.combine_classes = False

# SWU Dataset
__C.SWU = edict()
__C.SWU.ROOT_DIR = "./data/downloaded/SWUIllustration_869_v2"
__C.SWU.CLASSES = [
    "air conditioner",
    "dishwasher",
    "disinfection cabinet",
    "fridge",
    "gas stove",
    "oven",
    "panel",
    "remote",
    "washer",
    "water heater",
    "water purifier",
]


#
# Training options
#

__C.TRAIN = edict()

__C.EVAL = edict()


# Mean and std to normalize images
__C.NORM_MEANS = [0.485, 0.456, 0.406]
__C.NORM_STD = [0.229, 0.224, 0.225]

# random seed used for data loading
__C.RANDOM_SEED = 123
