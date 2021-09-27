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

architecture_ID = "O-R"

#Load from:
Z_test = np.load('C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/Results Z2V/Turf Valley/Embedding data/Z_tr_last-' + architecture_ID + '.npy')

# Save to:
ZForecast_path = 'C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/Results Z2V/Turf Valley/ZForecast/' + architecture_ID + '.npy'
Xhat_path = 'C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/Results Z2V/Turf Valley/Xhat/' + architecture_ID + "/"

#Option 1: load Latent-space LSTM (trained independently)
# model = keras.models.load_model('C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/saved models/Turf Valley Z2V/Latent Space LSTM/last-' + architecture_ID)

# Option 2: Extract LSTM module from trained full model (trained jointly with Encoder)
full_model = keras.models.load_model('C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/saved models/Turf Valley Z2V/DVE/last-' + architecture_ID)  # Full model
detached_LSTM_input = Input((75, 3), name='detached_LSTM_input')
w = full_model.layers[6](detached_LSTM_input)
vhat = full_model.layers[8](w)
model = Model(detached_LSTM_input, vhat, name='joint_LSTM')

model.summary()

N_steps = 300
T_test = len(Z_test)
l = 75
latent_dim=3
batch_size = 1
start_frame_test = l - 1  # first frame inded in the valid range
end_frame_test = T_test - 2  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
samples_test = end_frame_test - start_frame_test + 1

t = start_frame_test+1

ZSegment = Z_test[t - l + 1:t + 1, :]

ZSegment = np.expand_dims(ZSegment, axis=0)  # add batch axis
print(ZSegment.shape)

ZForecast = []
# march forward in the latent space, using the latent-LSTM model to predict the next velocity vector and z, and so on..
step_size = 0.1
for i in range(N_steps):
    vhat_i_plus_1 = model.predict(ZSegment)  # shape (1,3) NOTE: vhat has only been optimized to ALIGN (in orientation) with the true v. consider stepping in small increments. the magnitude of vhat is not clear

    # using full vhat as a step:
    z_i_plus_1 = ZVSegment[:, -1, :] + vhat_i_plus_1  # shape (1,3)

    ZForecast.append(np.squeeze(z_i_plus_1))
    z_i_plus_1 = np.expand_dims(z_i_plus_1, axis=0)  # shape (1, 1,6)
    ZSegment = np.concatenate((ZSegment[:, 1:, :], z_i_plus_1), axis=1)  # crop off first timestep and append the one just computed

ZForecast = np.array(ZForecast)
np.save(ZForecast_path, ZForecast)
print(f"ZForecast: {ZForecast.shape}")

# Now synthesize new frames using the decoder-----------------------------
#decoder is layer 15 for OF, OR, ORF. and 22 for ORFC

# full_model = keras.models.load_model('C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/saved models/Turf Valley Z2V/DVE/last-' + architecture_ID)  # Full model
# full_model.summary()
# detached_decoder_input = Input((3,), name='detached_decoder_input')
# xhat = full_model.layers[10](detached_decoder_input)
# decoder = Model(detached_decoder_input, xhat, name='decoder')
# # decoder.summary()
#
# Xhat = decoder.predict(ZForecast)
# Xhat = np.squeeze(Xhat)
# print(Xhat.shape)
#
# def deprocess_image(img_tensor):
#     img_tensor *= 255
#     img_tensor = np.clip(img_tensor, 0, 255)
#     img_tensor = np.uint8(img_tensor)
#     return img_tensor
#
# for i in range(N_steps):
#     img = deprocess_image(Xhat[i])
#     img = Image.fromarray(img, mode='L')
#     img.save(Xhat_path + 'frame_' + str(i) + '.png')

print('done saving images')

# Plot-----------------------------------------------------------------------

ZForecast = np.concatenate((np.expand_dims(Z_test[t, :], axis=0), ZForecast), axis=0)
print(f"ZForecast: {ZForecast.shape}")

fig2 = go.Figure(data=[go.Scatter3d(
    name='z test',
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
    name='z forecast',
    x=ZForecast[:, 0],
    y=ZForecast[:, 1],
    z=ZForecast[:, 2],
    mode='markers+lines',
    marker=dict(
        size=3,
        color='red',
        symbol='circle'
    ),
    line=dict(
        color='red',
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
