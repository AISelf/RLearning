"""
-*- coding:utf-8 -*-
@Author  :   liaoyu
@Contact :   doitliao@126.com
@File    :   keras-api.py
@Time    :   2019/12/24 16:31
@Desc    :
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def overview():
	model = tf.keras.Sequential()
	# Adds a densely-connected layer with 64 units to the model:
	model.add(layers.Dense(64, activation='relu'))
	# Add another:
	model.add(layers.Dense(64, activation='relu'))
	# Add a softmax layer with 10 output units:
	model.add(layers.Dense(10, activation='softmax'))
	model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
	              loss='mse',  # mean squared error
	              metrics=['mae'])  # mean absolute error

	data = np.random.random((1200, 32))
	labels = np.random.random((1200, 10))

	model.fit(data, labels, epochs=10, batch_size=32)


def functional_api():
	inputs = tf.keras.Input(shape=(784,))
	hidden1 = tf.keras.layers.Dense(64, activation='relu')(inputs)
	hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
	outputs = tf.keras.layers.Dense(10, activation='softmax')(hidden2)

	model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'doitliao-model')

	model.summary()


def define_multiple_model():
	encoder_input = tf.keras.Input(shape=(28, 28, 1), name='img')
	x = tf.keras.layers.Conv2D(16, 3, activation='relu')(encoder_input)
	x = tf.keras.layers.MaxPooling2D(3)(x)
	x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
	x = tf.keras.layers.Conv2D(16, 3, activation='relu')(x)
	encoder_output = layers.GlobalMaxPooling2D()(x)

	encoder = tf.keras.Model(encoder_input, encoder_output, name='encoder')
	encoder.summary()

	x = tf.keras.layers.Reshape((4, 4, 1))(encoder_output)
	x = tf.keras.layers.Conv2DTranspose(16, 3, activation='relu')(x)
	x = tf.keras.layers.Conv2DTranspose(32, 3, activation='relu')(x)
	x = tf.keras.layers.UpSampling2D(3)(x)
	x = tf.keras.layers.Conv2DTranspose(16, 3, activation='relu')(x)
	decoder_output = tf.keras.layers.Conv2DTranspose(1, 3, activation='relu')(x)

	autoencoder = tf.keras.Model(encoder_input, decoder_output, name='autoencoder')
	autoencoder.summary()

if __name__ == '__main__':
	functional_api()
	#define_multiple_model()