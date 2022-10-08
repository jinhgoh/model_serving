from flask import Flask, request
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import pandas as pd
import numpy as np

local_scaler = pickle.load(open('sc.pickle','rb'))

def ML_model(age, salary):
    
    input_size=2
    output_size=2
    hidden_size=10

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
            self.fc3 = torch.nn.Linear(hidden_size, output_size)


        def forward(self, X):
            X = torch.relu((self.fc1(X)))
            X = torch.relu((self.fc2(X)))
            X = self.fc3(X)

            return F.log_softmax(X,dim=1)

    new_predictor2 = Net()

    new_predictor2.load_state_dict(torch.load('customer_buy_state_dict'))
    prediction = new_predictor2(torch.from_numpy(local_scaler.transform(np.array([[age,salary]]))).float())[:,0]
    return prediction

