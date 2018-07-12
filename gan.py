#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

import sys
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
        
# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    df.columns = ['X', 'Y']
    return df

# create a differenced series
def difference(dataset, interval=1):
    assert interval >= 1
    assert isinstance(dataset, pd.DataFrame) or isinstance(dataset, pd.Series)
    return dataset.diff(interval).dropna()

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# create a differenced series
def difference(dataset, interval=1):
    assert interval >= 1
    assert isinstance(dataset, pd.DataFrame) or isinstance(dataset, pd.Series)
    return dataset.diff(interval).dropna()

class GAN():
    def __init__(self):
        self.data_shape = (1, 1,)
        self.latent_dim = (1, 1)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input((1, 1,))
        noise_datum = self.generator(z)

        # For the combined model we will only train the generator
        # self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(noise_datum)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='mean_squared_error', optimizer=optimizer)

    def save(self, logdir='saved_model'):
        print('Saving model...')
        self.discriminator.save(os.path.join(logdir, 'discriminator.h5'))
        self.generator.save(os.path.join(logdir, 'generator.h5'))
        self.combined.save(os.path.join(logdir, 'combined.h5'))
        print('Model saved in {}'.format(logdir))

    def build_generator(self, batch_size=1):

        model = Sequential()

        model.add(LSTM(4, batch_input_shape=(batch_size, 1, 1), stateful=True))
        model.add(Dense(1))
        model.add(Reshape(self.data_shape))
        model.summary()

        #noise = Input(shape=(batch_size, 1, 1))
        #data = model(noise)

        #return Model(noise, data)
        return model

    def build_discriminator(self, batch_size=1):

        model = Sequential()

        model.add(LSTM(4, batch_input_shape=(batch_size, 1, 1), stateful=True))
        model.add(Dense(1))
        model.summary()

        #data = Input(shape=(batch_size, 1, 1))
        #validity = model(data)

        #return Model(data, validity)
        return model

    def train(self, epochs, batch_size=128, sample_interval=50, logpath='saved_model'):

        # Load the dataset
        data = pd.read_csv('data/spy-daily.csv', index_col='Date')
        adj_close_data = data['Adj Close']
        sup_data = timeseries_to_supervised(adj_close_data)
        X = adj_close_data.values
        X = X.reshape(len(X), 1)

        # Rescale -1 to 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(X)
        scaled_X = scaler.transform(X)
        
        # Split training and test set
        supervised_values = sup_data.values
        train_lim = int((4/5) * supervised_values.shape[0])
        self.train, self.test = supervised_values[:train_lim, :], supervised_values[train_lim:, :]

        # Scale training and test sets
        self.train_scaled, self.test_scaled = scaler.transform(self.train), scaler.transform(self.test)
        
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        callback = TensorBoard(logpath)
        callback.set_model(self.combined)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of data
            idx = np.random.randint(0, self.train.shape[0], batch_size)
            random_data = self.train_scaled[idx]

            noise = np.random.normal(0, 1, (batch_size, 1, 1))
            print(noise.dtype, noise.shape)

            # Generate a batch of new data
            gen_data = self.generator.predict(noise, batch_size=1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(random_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            self.discriminator.reset_states()

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 1, 100))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot and log the progress
            dloss = d_loss[0]
            acc = d_loss[1]
            gloss = g_loss
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, dloss, 100*acc, gloss))
            write_log(callback, ['D_loss', 'acc', 'G_loss'], [dloss, acc, gloss], epoch)

            # If at save interval => save generated image samples
            #if epoch % sample_interval == 0:
            #    self.sample_images(epoch)

    def sample_images(self, epoch, callback=None):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise, batch_size=1)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()
        if callback is not None:
            summary_name = 'gen-{}'.format(epoch)
            buff = tf.summary.image(summary_name,
                                    gen_imgs,
                                    max_outputs=r*c)
            # summary = tf.Summary(value=[tf.Summary.Value(tag=summary_name,
            #                                              image=buff)])
            writer = tf.summary.FileWriter('saved_model')
            writer.add_summary(buff, epoch)

if __name__ == '__main__':
    gan = GAN()
    try:
        gan.train(epochs=3000, batch_size=32, sample_interval=200)
    except KeyboardInterrupt:
        print('Training stopped!')
    finally:
        gan.save()
