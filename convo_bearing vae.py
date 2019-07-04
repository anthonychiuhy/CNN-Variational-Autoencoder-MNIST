# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:53:50 2019

@author: Anthony
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.layers import Lambda
from keras.models import Model
from keras import backend as K
from keras import regularizers
from keras import losses

from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()



x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

def encoder_layers(input_dim, latent_dim):
    # Encoder
    input_layer = Input(shape=input_dim)
    x = Conv2D(8, 3, activation='relu', padding='same')(input_layer)
    x = Conv2D(8, 3, activation='relu', padding='same')(x)
    x = Conv2D(8, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2, padding='same')(x) # Size 14x14x8
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2, padding='same')(x) # Size 7x7x16
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Flatten()(x) # Size 1568 = 7x7x32
    
    x =  Dense(100, activation='relu')(x)
    
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    return Model(input_layer, (z_mean, z_log_var))

def sampling_layer(latent_dim):
    
    z_mean = Input(shape=(latent_dim,))
    z_log_var = Input(shape=(latent_dim,))
    
    stats = [z_mean, z_log_var]
    
    def sampling(stats):
        z_mean, z_log_var = stats
        epsilon = K.random_normal(shape=K.shape(z_mean), mean=0., stddev=1) # standard normal distribution
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    z = Lambda(sampling)(stats)
    
    return Model(stats, z)


def decoder_layers(latent_dim):
    # Decoder
    z = Input(shape=(latent_dim,))
    
    x = Dense(100, activation='relu')(z)
    
    x = Dense(1568)(x)
    x = Reshape((7, 7, 32))(x) # Size 7x7x32
    x = Conv2D(32,  3, activation='relu', padding='same')(x)
    x = Conv2D(32,  3, activation='relu', padding='same')(x)
    x = Conv2D(32,  3, activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x) # Size 14x14x16
    x = Conv2D(16,  3, activation='relu', padding='same')(x)
    x = Conv2D(16,  3, activation='relu', padding='same')(x)
    x = Conv2D(16,  3, activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)  # Size 28x28x16
    x = Conv2D(8,  3, activation='relu', padding='same')(x) # Size 28x28x8
    x = Conv2D(8,  3, activation='relu', padding='same')(x) # Size 28x28x8
    output_layer = Conv2D(1, 3, activation='relu', padding='same')(x)
    
    return Model(z, output_layer)


batch_size = 128
epochs = 2

input_dim = (28, 28, 1)
latent_dim = 10

encoder = encoder_layers(input_dim, latent_dim)
sampler = sampling_layer(latent_dim)
decoder = decoder_layers(latent_dim)

inputs = Input(shape=input_dim)
stats = encoder(inputs)
z = sampler(stats)
outputs = decoder(z)

model_vae = Model(inputs, outputs)


z_mean, z_log_var = stats

def vae_loss(inputs, outputs):
    x_loss = K.sum(K.square(inputs - outputs), axis=(1,2,3))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.exp(z_log_var) - K.square(z_mean), axis=1)
    return K.mean(x_loss + kl_loss)

model_vae.compile(optimizer='adam', loss=vae_loss)
model_vae.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, x_test))


decoded_imgs = model_vae.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)



z = Input((latent_dim,)) 
outputs = decoder(z)
generator = Model(z, outputs)



n = 10
z = np.random.multivariate_normal(np.zeros(latent_dim), np.eye(latent_dim), n**2)
gen_img = generator.predict(z)

plt.figure(figsize=(n,n))
for i in range(n*n):
        ax = plt.subplot(n, n, i + 1)
        plt.imshow(gen_img[i].reshape((28, 28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)





