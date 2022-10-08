from flask import Flask, request
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import pandas as pd
import numpy as np

from my_pytorch import ML_model

from flask_restx import Api, Resource, reqparse

app = Flask(__name__)


simple_dict = {"a":1, "b":2}

@app.route('/local_result')
def local_result():
   return 'Hello, world!'


@app.route('/model',methods=['POST'])
def ML_result():
    request_data = request.get_json(force=True)
    age = request_data['age']
    salary = request_data['salary']
    print(age)
    print(salary)

    prediction = ML_model(40,20000)
    
    print("prediction:")
    print(prediction)
    return "The prediction from GCP API is {}".format(prediction)
    #return "1"

@app.route('/hello')
def hello():
   return 'Hello, world!'

if __name__ == "__main__":
    app.run(host= '0.0.0.0',port=8005, debug=True)


# https://flask-restx.readthedocs.io/en/latest/parsing.html
#https://velog.io/@minho/SyntaxError-positioanl-argument-follows-keyword-argument