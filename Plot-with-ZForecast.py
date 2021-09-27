import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import os
import time
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, TimeDistributed, Flatten, Concatenate, LSTM, MaxPool2D, Reshape, Conv2DTranspose, BatchNormalization, UpSampling2D, Cropping2D, Conv2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras import backend as K
import pydot as pydot

from PIL import Image
import pandas as pd
import csv
import plotly.graph_objects as go


architecture_ID = "O-R-F-C"

Z_test = np.load('C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/Results Z2V/Turf Valley/Embedding data/Z_test_last-' + architecture_ID + '.npy')
ZForecast_path = 'C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/Results Z2V/Turf Valley/ZForecast/' + architecture_ID + '.npy'
Xhat_path = 'C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/Results Z2V/Turf Valley/Xhat/' + architecture_ID
latent_dim = 3
ZForecast = np.load(ZForecast_path)

# T = np.arange(1, Z.shape[0], 1)
# print(Z.shape)
# loop1end = 2251
# loop2end = 4444

fig2 = go.Figure(data=[go.Scatter3d(
    name='Loop 1',
    x=Z_test[:, 0],
    y=Z_test[:, 1],
    z=Z_test[:, 2],
    mode='markers+lines',
    marker=dict(
        size=2,
        color='black',
        symbol='circle'
    ),
    line=dict(
        color='black',
        width=4
    )
)])

fig2.add_trace(go.Scatter3d(
    name='Loop 2',
    x=ZForecast[:, 0],
    y=ZForecast[:, 1],
    z=ZForecast[:, 2],
    mode='markers+lines',
    marker=dict(
        size=3,
        color='green',
        symbol='circle'
    ),
    line=dict(
        color='green',
        width=4
    )
))


fig2.update_layout(
    width=900,
    height=700,
    scene_xaxis_title_text='z<sub>1</sub>',
    scene_yaxis_title_text="z<sub>2</sub>",
    scene_zaxis_title_text="z<sub>3</sub>",

    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.17,
        font_size=16
    ),
    scene_xaxis=dict(tickfont=dict(size=12),
                     title_font_size=20),
    scene_yaxis=dict(tickfont=dict(size=12),
                     title_font_size=20),
    scene_zaxis=dict(tickfont=dict(size=12),
                     title_font_size=20),
)

fig2.show()
