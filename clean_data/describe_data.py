import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
pd.set_option('max_columns', 100)
pd.set_option('max_rows', 20)

cwd = os.getcwd()
# df = pd.read_csv(cwd + "/out1.csv", index_col=0)

# print(df.head())
# print(df.info)

# calculate age


# sankey diagram switching levels
fig = go.Figure(data=[go.Sankey(
    node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = ["A1", "A2", "B1", "B2", "C1", "C2"],
        color = "blue"
    ),
    link = dict(
        source = [0, 1, 0, 2, 3, 3],
        target = [2, 3, 3, 4, 4, 5],
        value = [8, 4, 2, 8, 4, 2]
    ))])

fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)
fig.write_image("/../output/image.png")
