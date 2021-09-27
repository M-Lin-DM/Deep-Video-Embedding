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
import plotly.graph_objects as go

train_dir = 'D:/Datasets/Turf Valley Resort/train/New folder/'
validation_dir = 'D:/Datasets/Turf Valley Resort/validation/New folder/'
test_dir = 'D:/Datasets/Turf Valley Resort/test/New folder/'

def count_files(path):
    Number_Of_Files = 0
    for file in os.walk(path):
        #     print(file)
        for files in file[2]:
            Number_Of_Files = Number_Of_Files + 1
    return Number_Of_Files

architecture_ID = "O-R-F32"
model = keras.models.load_model('C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/saved models/Turf Valley Z2V/DVE/last-' + architecture_ID)
model.summary()

# tf.keras.utils.plot_model(
#     model, to_file='full_model.png', show_shapes=True,
#     show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

model.layers[2] #The wrapped encoder
target_size = (150, 268)
top_frames_shape = (2,) + tuple(target_size) + (1,)

detached_encoder_input = Input(top_frames_shape, name='detached_encoder_input')
embedded_top_frames = model.layers[2](detached_encoder_input)
output=embedded_top_frames[:,0,:]  # we only need the first vector since the prediction generator will be looping over all points

encoder = Model(detached_encoder_input, output, name='encoder')
encoder.summary()

l = 75
T_tr = count_files(train_dir)
T_val = count_files(validation_dir)
T_test = count_files(test_dir)

batch_size = 1
top_frames_shape = (2,) + tuple(target_size) + (1,)


def inputs_generator_inference(t_start, t_end, image_folder):
    t = t_start
    while t <= t_end:
        top_frames = []

        for j in range(t, t + 2):
            pil_img = load_img(image_folder.decode('UTF-8') + f'{j:05}' + '.png', color_mode='grayscale',
                               target_size=target_size)
            top_frames.append(
                img_to_array(pil_img, data_format='channels_last', dtype='uint8').astype(np.float32) / 255)
        top_frames = np.array(top_frames)

        yield top_frames  # 0 represents the target output of the model
        t += 1


start_frame_tr = 0  # first frame inded in the valid range
end_frame_tr = T_tr - 2  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
samples_tr = end_frame_tr - start_frame_tr + 1
start_frame_val = l - 1  # first frame inded in the valid range
end_frame_val = T_val - 2  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
start_frame_test = l - 1  # first frame inded in the valid range
end_frame_test = T_test - 2  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1

ds_train = tf.data.Dataset.from_generator(
    inputs_generator_inference,
    args=[start_frame_tr, end_frame_tr, train_dir],
    output_types=(tf.float32),
    output_shapes=top_frames_shape)

ds_val = tf.data.Dataset.from_generator(
    inputs_generator_inference,
    args=[start_frame_val, end_frame_val, validation_dir],
    output_types=(tf.float32),
    output_shapes=top_frames_shape)

ds_test = tf.data.Dataset.from_generator(
    inputs_generator_inference,
    args=[start_frame_test, end_frame_test, test_dir],
    output_types=(tf.float32),
    output_shapes=top_frames_shape)

print(ds_train.element_spec)

ds_train = ds_train.batch(batch_size, drop_remainder=False)
ds_val = ds_val.batch(batch_size, drop_remainder=False)
ds_test = ds_test.batch(batch_size, drop_remainder=False)

# Run encoder.predict
Ztr = encoder.predict(ds_train)
np.save(r'C:\Users\MrLin\Documents\Experiments\Deep Video Embedding\Results Z2V\Turf Valley\Embedding data\Z_tr_last-'+architecture_ID, Ztr)
Ztr.shape
Zval = encoder.predict(ds_val)
np.save(r'C:\Users\MrLin\Documents\Experiments\Deep Video Embedding\Results Z2V\Turf Valley\Embedding data\Z_val_last-'+architecture_ID, Zval)
Ztest = encoder.predict(ds_test)
np.save(r'C:\Users\MrLin\Documents\Experiments\Deep Video Embedding\Results Z2V\Turf Valley\Embedding data\Z_test_last-'+architecture_ID, Ztest)

# Plot----------------------
Z = np.load(
    r'C:\Users\MrLin\Documents\Experiments\Deep Video Embedding\Results Z2V\Turf Valley\Embedding data\Z_tr_last-' + architecture_ID + '.npy')
T = np.arange(1, Z.shape[0], 1)
print(Z.shape)
loop1end = 2251
loop2end = 4444

fig2 = go.Figure(data=[go.Scatter3d(
    name='Loop 1',
    x=Z[:loop1end, 0],
    y=Z[:loop1end, 1],
    z=Z[:loop1end, 2],
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