# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 08:07:17 2021

@author: abhij
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('mnist_train.csv')
X=df.iloc[:, 1:].values
y=df.iloc[:,0].values

X = (X - 127.5) / 127.5

BUFFER_SIZE=60000
BATCH_SIZE=32

train_dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


from keras.models import Sequential
from keras.layers import InputLayer ,Convolution2D, Dense, Flatten, BatchNormalization, LeakyReLU, Conv2DTranspose, Reshape
from keras.optimizers import Adam

cri=Sequential()
cri.add(InputLayer(input_shape=(28,28,1)))
cri.add(Convolution2D(128,(3,3)))
cri.add(LeakyReLU(alpha=0.1))
cri.add(BatchNormalization())
cri.add(Convolution2D(64,(3,3)))
cri.add(LeakyReLU(alpha=0.1))
cri.add(BatchNormalization())
cri.add(Flatten())
cri.add(Dense(64))
cri.add(LeakyReLU(alpha=0.1))
cri.add(Dense(1))
cri.summary()

gen=Sequential()
gen.add(InputLayer(input_shape=(100,)))
gen.add(Dense(7*7*256))
gen.add(BatchNormalization())
gen.add(LeakyReLU(alpha=0.1))
gen.add(Reshape((7,7,256)))
gen.add(Conv2DTranspose(128,(3,3),padding='same',strides=(1,1)))
gen.add(BatchNormalization())
gen.add(LeakyReLU(alpha=0.1))
gen.add(Conv2DTranspose(64,(3,3),padding='same',strides=(2,2)))
gen.add(BatchNormalization())
gen.add(LeakyReLU(alpha=0.1))
gen.add(Conv2DTranspose(1,(3,3),padding='same',strides=(2,2)))
gen.summary()



def gradient_pen(real,fake,batch_size):
    alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
    arr1=np.array(fake)
    arr2=np.array(real)
    diff = arr1-arr2
    interpolated = arr2 + alpha * diff
    with tf.GradientTape() as g:
        g.watch(interpolated)
        pred=cri(interpolated)
        grad=g.gradient(pred,[interpolated])[0]
    norm=tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
    pen= tf.reduce_mean((norm-1.0)**2)
    return pen 

def w_loss_cri(real,fake):
    l=10
    reg=gradient_pen(real,fake,32)
    real_output = cri(real, training=True)
    fake_output = cri(fake, training=True)

    loss= tf.reduce_mean(fake_output)-tf.reduce_mean(real_output)+l*reg
    return loss

def w_loss_gen(fake):
    fake_output = cri(fake, training=True)
    loss=-tf.reduce_mean(fake_output)
    return loss 


generator_optimizer = Adam(1e-4)
discriminator_optimizer = Adam(1e-4)

def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])
    real_images=tf.reshape(images,(32,28,28,1))
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = gen(noise, training=True)

      gen_loss =  w_loss_gen(generated_images)
      disc_loss = w_loss_cri(real_images, generated_images)

    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss,cri.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,cri.trainable_variables))

import time
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()  
    for image_batch in dataset:
      train_step(image_batch)
    
    noise = tf.random.normal([1, 100])
    generated_image = gen(noise, training=False)
    print ('Time for epoch {} is {} sec'.format(epoch + 1,time.time()-start))
    plt.imshow(generated_image[0,:,:,0], cmap='gray')

train(train_dataset, 5)

noise = tf.random.normal([1, 100])
generated_image = gen(noise, training=False)
plt.imshow(generated_image[0,:,:,0], cmap='gray')
   
gen.save('generator')
gene=tf.keras.models.load_model('generator')

noise = tf.random.normal([1, 100])
generated_image = gene(noise, training=False)
plt.imshow(generated_image[0,:,:,0], cmap='gray')
