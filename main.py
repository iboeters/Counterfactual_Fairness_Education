import pandas as pd
import numpy as np
import os
import sys
import miceforest as mf
from import_data import import_data
from impute_data import impute_data
from clean_data import clean_data

# import and clean data
df = import_data()
df = clean_data(df)
print(df.head())

# impute education level parents
df = impute_data(df)
print(df.head())
