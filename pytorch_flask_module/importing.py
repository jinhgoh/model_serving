from flask import Flask, request
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import pandas as pd
import numpy as np


from my_pytorch import ML_model

prediction = ML_model(40,20000)
print(prediction)
print('-------------')


