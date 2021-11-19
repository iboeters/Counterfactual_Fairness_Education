import pandas as pd
import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from collections import defaultdict
from argparse import ArgumentParser
from get_data import get_data, get_train_test_split
from fairness_unawareness import fairness_unawareness
from counterfactual_fairness import counterfactual_fairness
from train_CEVAE import train_CEVAE

# to run:
# python main.py -filename=test -model_name=test -nIter_CEVAE=100 -nIter=100 -evalIter=100 -uDim=5 -rep=1

parser = ArgumentParser()
parser.add_argument('-lr', type=float, default=0.0005)
parser.add_argument('-hDim', type=int, default=50)
parser.add_argument('-uDim', type=int, default=5)
parser.add_argument('-rep', type=int, default=10)
parser.add_argument('-nIter_CEVAE', type=int, default=15000)
parser.add_argument('-nIter', type=int, default=1000)
parser.add_argument('-batchSize', type=int, default=500)
parser.add_argument('-nSamplesU', type=int, default=1)
parser.add_argument('-evalIter', type=int, default=100)
parser.add_argument('-dataset', type=str, default='out1_dum.csv')
parser.add_argument('-device', type=str, default='cpu')
parser.add_argument('-test_size', type=int, default=0.1)
parser.add_argument('-filename', type=str, default='x')
parser.add_argument('-model_name', type=str, default='x')
args = parser.parse_args()

# pd.set_option('max_columns', 999)

# check if cuda is available; if cuda => run on gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# load data -------------------------------------------------------------
data = get_data(args.dataset)

# train causal effect variational autoencoder ---------------------------
train_data, test_data = get_train_test_split(data, args.test_size)
train_CEVAE(args, train_data, test_data)

# train ML models for fair prediction -----------------------------------
train_data, test_data = get_train_test_split(data, args.test_size)
counterfactual_fairness(args, train_data, test_data)
