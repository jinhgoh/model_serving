from flask import Flask, request
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import pandas as pd
import numpy as np

# if name main 등 함수 안에 넣어서 필요한것들이 다 실행되게 하기 (import * 하는게 아니라)
# 아 from A import a 하면 A를 다 실행시키고가져오는게 아니라 a의 코드만 가져오는것. (실행하거나)


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

'''
if __name__ == "__main__":
    app.run(host= '0.0.0.0',port=8005, debug=True)
'''
