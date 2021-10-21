import pandas as pd
import numpy as np
import os
import sys
from import_data import import_data
from impute_data import impute_data
from clean_data import clean_data

# import and clean data
df = import_data()
df = clean_data(df)

# make dataset ready for imputation
df = impute_data(df)
