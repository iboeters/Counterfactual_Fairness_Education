import pandas as pd
import numpy as np
import holoviews as hv
import os
import plotly.graph_objects as go
import plotly.express as pex

# hv.extension('bokeh')
pd.set_option('max_columns', 100)
pd.set_option('max_rows', 100)

# read data
cwd = os.getcwd()
df = pd.read_csv(cwd + "/../output/out1.csv", index_col=0)
print(df.head())

# save dataset for switching levels
df_counts = df[['ADV_FINAL', 'LEVEL3']].copy()
df_counts = df_counts.value_counts()
df_counts = df_counts.to_frame('value').reset_index()
df_counts['LEVEL3'] += 20
df_counts['value_perc'] = df_counts['value'] / len(df) * 100
df_counts.to_csv('../output/sankey_diagram.csv')

# dataset for ethnicity ~ adv_final + level3
df_e = df[['ETHN', 'LEVEL3']].copy()
df_e = df_e.value_counts()
df_e = df_e.to_frame('value').reset_index()
df_e2 = df[['ETHN', 'ADV_FINAL']].copy()
df_e2 = df_e2.value_counts()
df_e2 = df_e2.to_frame('value').reset_index()
df_e.to_csv('../output/df_e.csv')
df_e2.to_csv('../output/df_e2.csv')

df_total = df['ETHN'].value_counts().to_frame('total').reset_index()
df_e2.to_csv('../output/df_total.csv')
