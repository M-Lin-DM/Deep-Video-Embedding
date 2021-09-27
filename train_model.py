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

import wandb
from wandb.keras import WandbCallback

train_dir = 'D:/Datasets/Turf Valley Resort/train/New folder/'
validation_dir = 'D:/Datasets/Turf Valley Resort/validation/New folder/'

architecture_ID = "O-R-F32"

checkpoint_path = 'C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/saved models/Turf Valley Z2V/DVE/last-' + architecture_ID
checkpoint_path_best_validation = 'C:/Users/MrLin/Documents/Experiments/Deep Video Embedding/saved models/Turf Valley Z2V/DVE/bestVAL-' + architecture_ID

# wandb.login()
id = wandb.util.generate_id()
print(f"Wandb ID: {id}")
wandb.init(entity='your-wandb', project='DVE', group='Z2V-2', name=architecture_ID, id=id, resume="allow",  # if you want to resume a crashed run, lookup the run id in W&B and paste it into id= and set resume="allow"
           config={"l": 75,
                   "batch_size": 4,
                   "target_size": (150, 268),  # (img_height, img_width)
                   "epochs": 12,
                   "dropout": 0.0,
                   "L1": 0.000,
                   "L2": 0.000,
                   "alpha": 0.001,  # F forecasting loss weight. 1.5
                   "beta": 1,  # R reconstruction loss weight, 1
                   "gamma": 0.000,  # O off-center loss weight, 0.01
                   "delta": 0,  # C curvature loss weight, 2
                   "checkpoint_path": checkpoint_path})
config = wandb.config

l = config.l  # clip length, the number of frames in the video segment fed to the ConvLSTM, should be odd # if decimating by a factor 2
target_size = tuple(config.target_size)
batch_size = config.batch_size
gamma = config.gamma  # off-center loss weight
beta = config.beta  # reconstruction loss weight
alpha = config.alpha  # forecasting loss weight
delta = config.delta  # curvature loss weight

def count_files(path):
    Number_Of_Files = 0
    for file in os.walk(path):
        #     print(file)
        for files in file[2]:
            Number_Of_Files = Number_Of_Files + 1
    return Number_Of_Files

print(count_files(train_dir))

T_tr = count_files(train_dir)  # number of images in the dataset
T_val = count_files(validation_dir)
decimation = 1  # time decimation factor. "take a frame every _decimation_ frames"

clip_tensor_shape = (l,) + tuple(target_size) + (1,)
top_frames_shape = (2,) + tuple(target_size) + (1,)
latent_dim = 3

steps_per_epoch_tuning = 100 #11230  # fraction the dataset's total batches (11231.5 batches, batchsize=4) or (5615.75 batchsize=8)
print(f"clip_tensor_shape: {clip_tensor_shape} | steps_per_epoch_tuning: {steps_per_epoch_tuning}")


def inputs_generator(t_start, t_end, image_folder):
    t = t_start
    while t <= t_end:
        clip_tensor = []
        top_frames = []
        for j in range(t - l + 1, t + 1, decimation):
            pil_img = load_img(image_folder.decode('UTF-8') + f'{j:05}' + '.png', color_mode='grayscale', target_size=target_size)
            clip_tensor.append(
                img_to_array(pil_img, data_format='channels_last', dtype='uint8').astype(np.float32) / 255)  # pixel values are normalized to [0,1]
        clip_tensor = np.array(clip_tensor)

        for j in range(t, t + 2):
            pil_img = load_img(image_folder.decode('UTF-8') + f'{j:05}' + '.png', color_mode='grayscale', target_size=target_size)
            top_frames.append(
                img_to_array(pil_img, data_format='channels_last', dtype='uint8').astype(np.float32) / 255)
        top_frames = np.array(top_frames)

        yield ((clip_tensor, top_frames), 0)  # 0 represents the target output of the model. the metric in .compile is computed using this. the addloss layer outputs the loss itself just for convienience.
        t += 1


# Load ground truth labels and define parameters for train dataset

start_frame_tr = l - 1  # first frame inded in the valid range
end_frame_tr = T_tr - 2  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
samples_tr = end_frame_tr - start_frame_tr + 1
steps_per_epoch_tr = int(np.floor((end_frame_tr - start_frame_tr + 1) / config.batch_size))  # number of batches

start_frame_val = l - 1  # first frame inded in the valid range
end_frame_val = T_val - 2  # last frame index in the valid range, assuming images start at t=0 and go to t=T-1
steps_per_epoch_val = int(np.floor((end_frame_val - start_frame_val + 1) / config.batch_size))

ds_train = tf.data.Dataset.from_generator(
    inputs_generator,
    args=[start_frame_tr, end_frame_tr, train_dir],
    output_types=((tf.float32, tf.float32), tf.float32),
    output_shapes=(((l,) + tuple(target_size) + (1,), (2,) + tuple(target_size) + (1,)), ()))

ds_val = tf.data.Dataset.from_generator(
    inputs_generator,
    args=[start_frame_val, end_frame_val, validation_dir],
    output_types=((tf.float32, tf.float32), tf.float32),
    output_shapes=(((l,) + tuple(target_size) + (1,), (2,) + tuple(target_size) + (1,)), ()))

ds_validation = ds_val.batch(batch_size, drop_remainder=True).repeat(config.epochs)

# inspect
print(ds_train.element_spec)

# print('shuffling dataset')
ds_train = ds_train.shuffle(int(samples_tr * 0.33),
                            # turn off shuffle if you are training on a subset of the data for hyperparameter tuning and plan to visualize performance on the first steps_per_epoch_tuning batches of the TRAINING set
                            reshuffle_each_iteration=False)  # the argument into shuffle is the buffer size. this can be smaller than the number of samples, especially when using larger datasets
ds_train = ds_train.batch(batch_size, drop_remainder=True)  # insufficient data error was thrown without adding the .repeat(). I added the +1 dataset at the end for good measure
ds_train = ds_train.repeat(config.epochs + 1)

# Below is for training on a subset of the data during hyperparameter tuning/prototyping------------------
# ds_train = ds_train.take(steps_per_epoch_tuning).repeat(
#     config.epochs+1)  # 600 batches of batch_size =8 is about 10% of a dataset with 44000 samples ie (~5488 batches)

print("shapes after batching", ds_train.element_spec)

# Build model------------------

#Build Encoder module
encoder_input = Input(shape=tuple(target_size + (1,)), name='encoder_input')
x = Conv2D(32, 3, strides=(1, 1), activation='relu', padding='valid', kernel_initializer='RandomNormal', bias_initializer='zeros')(encoder_input)
x = MaxPool2D(pool_size=(2, 2), padding='valid', data_format='channels_last')(x)
x = Conv2D(64, 3, strides=(1, 1), activation='relu', padding='valid', kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
x = MaxPool2D(pool_size=(2, 2), padding='valid', data_format='channels_last')(x)
x = Conv2D(128, 3, strides=(1, 1), activation='relu', padding='valid', kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
x = MaxPool2D(pool_size=(2, 2), padding='valid', data_format='channels_last')(x)
x = Conv2D(128, 3, strides=(1, 1), activation='relu', padding='valid', kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
x = MaxPool2D(pool_size=(2, 2), padding='valid', data_format='channels_last')(x)
x = Flatten(data_format='channels_last')(x)
z_encoder = Dense(latent_dim, activation='linear', name='z_encoder', kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
Encoder = Model(encoder_input, z_encoder, name='Encoder')

# build Decoder Module
decoder_input = Input(shape=(3,), name='decoder_input')
x = Dense(10336, activation='relu', use_bias=False, kernel_initializer='RandomNormal')(decoder_input)
x = Reshape((19, 34, 16))(x)
x = UpSampling2D(2, name='decoder_upsample1')(x)
x = Conv2DTranspose(64, 3, strides=(1, 1), activation='relu', padding='same', kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
x = BatchNormalization(axis=-1)(x)
x = UpSampling2D(2, name='decoder_upsample2')(x)
x = Conv2DTranspose(64, 3, strides=(1, 1), activation='relu', padding='same', kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
x = BatchNormalization(axis=-1)(x)
x = UpSampling2D(2, name='decoder_upsample3')(x)
x = Cropping2D(cropping=(1, 2), name='crop')(x)
x = Conv2DTranspose(128, 3, strides=(1, 1), activation='relu', padding='same', kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
x = BatchNormalization(axis=-1)(x)
x = Conv2DTranspose(256, 1, strides=(1, 1), activation='relu', padding='same', kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
x = BatchNormalization(axis=-1)(x)
z_decoded = Conv2DTranspose(1, 1, strides=(1, 1), activation='sigmoid', padding='same', kernel_initializer='RandomNormal', bias_initializer='zeros')(x)
Decoder = Model(decoder_input, z_decoded, name='Decoder')

# Input branch 1: prediction from frame sequence
input_clip_tensor = Input(shape=clip_tensor_shape, name='input_clip_tensor')
z = TimeDistributed(Encoder)(input_clip_tensor)  # returns shape (batch, timesteps, features)
w = LSTM(32, activation="tanh", return_sequences=False, dropout=0.1, name='LSTM_0')(z)
vhat = Dense(latent_dim, activation='linear', name='vhat')(w)  # Predicted target

# Input branch 2: compute target
input_top_frames = Input(shape=top_frames_shape, name='input_top_frames')  # Input is (batch, 2 timesteps, height, width, 1)
x_i = input_top_frames[:, 0, :, :, :]  # take frame at current time-step
z_top_frames = TimeDistributed(Encoder)(input_top_frames)
z_i = z_top_frames[:, 0, :]  # shape (batch, features) embedding of frame at current time-step
z_i_plus_1 = z_top_frames[:, 1, :]
v = z_i_plus_1 - z_i  # Self-supervised ground truth target. velocity vector in latent_dim-dimensional latent space

# Output branch 0: Decoder, reconstruct top image
x_i_hat = Decoder(z_i)

# Output branch 2: compute loss
class Add_model_loss(keras.layers.Layer):

    def compute_loss(self, inputs): # later this may also include the reconstruction loss from decoder. loss-weight parameters can be used to weight the impact of the reconstruction and forcasting loss
        v_, vhat_, x_i_, x_i_hat_, z_i_, z_ = inputs
        off_center_loss = K.mean(K.square(z_i_), axis=1)
        recon_loss = keras.metrics.binary_crossentropy(K.flatten(x_i_), K.flatten(x_i_hat_))  # The binary cross entropy averaged over all pixels should have a value around ~ 0.5-2 when model untrained
        # forcasting_loss = K.mean(K.square(vhat_ - v_))  # The value of this is harder to know in advance since it depends on to what spatial scale the encoder maps the images. Use negative cosine similarity instead
        forcasting_loss = tf.keras.losses.CosineSimilarity(axis=-1)(vhat_, v_)  #shape:(batch size,) in [-1, 1] where 1 indicates diametrically opposed. I believe the batch axis is still 0

        # Get curvature loss
        VSegment = K.l2_normalize(z_[:, 1:, :] - z_[:, 0:-1, :], axis=-1)  # normalized velocity vectors in frame sequence. shape (batch, timesteps, features)
        neg_cosine_theta = -K.sum(VSegment[:, 0:-1, :]*VSegment[:, 1:, :], axis=-1)  # take dot product of each pair of consecutive normalized velocity vectors= cos(theta)--> *-1 makes opposing vectors have a value -cos(theta)=1 ie high loss
        curvature_loss = K.mean(neg_cosine_theta, axis=1)  # shape:(batch size,).  mean (across time) of -cos similarity between each consecutive pair of velocity vectors. encourages embeddings with lower curvature

        loss = alpha*forcasting_loss + beta*recon_loss + gamma*off_center_loss + delta*curvature_loss
        return loss, off_center_loss, recon_loss, forcasting_loss, curvature_loss

    def call(self, layer_inputs):
        if not isinstance(layer_inputs, list):
            ValueError('Input must be list of [v, vhat, x_i, x_i_hat, z_i, z_]')
        loss, off_center_loss, recon_loss, forcasting_loss, curvature_loss = self.compute_loss(layer_inputs)
        self.add_loss(loss)
        self.add_metric(off_center_loss, name='off_center_loss')
        self.add_metric(recon_loss, name='recon_loss')
        self.add_metric(forcasting_loss, name='forecasting_loss')
        self.add_metric(curvature_loss, name='curvature_loss')
        return loss  # we dont need to return vhat again since this layer wont be used during inference. it doesnt matter what is returned here as long as its a scalar


output = Add_model_loss()([v, vhat, x_i, x_i_hat, z_i, z])

# ---- Load model ----  if you are resuming training. (It takes a long time..) In that case, Do NOT recompile model or you will lose the optimizer state.------------
# model = keras.models.load_model(r"C:\Users\MrLin\Documents\Experiments\Deep Video Embedding\saved models\last-no-forecast")
# model.summary()

# Create new model for training-----------
model = Model([input_clip_tensor, input_top_frames], output)
model.compile(loss=None,  # compute loss internally
              optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False))
# model.summary()

tf.keras.utils.plot_model(
    model, to_file='full_model.png', show_shapes=True,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

# END build model----------------

# # Train model-------------------------------

# Make checkpoint callback
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    monitor='loss', #the loss, weighted by class weights
    # change to 'loss' instead of 'val_loss' if youre tuning hyperparams and training on a small subset of the data
    mode='min',
    save_best_only=False,
    save_freq='epoch')  # change to false if youre tuning hyperparams and training on a small subset of the data

model_checkpoint_callback_best_validation = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path_best_validation,
    save_weights_only=False,
    monitor='val_loss',
    # change to 'loss' instead of 'val_loss' if youre tuning hyperparams and training on a small subset of the data
    mode='min',
    save_best_only=True)  # change to false if youre tuning hyperparams and training on a small subset of the data

model.fit(ds_train,
          steps_per_epoch=steps_per_epoch_tr,
          # 5607,  # steps_per_epoch needs to equal exactly the number of batches in the Dataset generator
          epochs=config.epochs,
          verbose=2,
          validation_data=ds_validation,
          validation_steps=steps_per_epoch_val,  # 436,  # =#batches in validation dataset
          callbacks=[WandbCallback(), model_checkpoint_callback, model_checkpoint_callback_best_validation])  #
