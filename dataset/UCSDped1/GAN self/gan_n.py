import os
import numpy as np
import matplotlib.pyplot as plt



from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from pathlib import Path
import pickle
from PIL import Image
import os

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def flat(image_list):
    list_img = []
    for x in image_list:
        x = x.flatten()
        list_img.append(x)
    
    return list_img

def image_to_npmat():
    currdir = os.listdir()
    os.chdir(currdir[1])
    currdir = os.listdir()
    os.chdir(currdir[0])
    currdir = os.listdir()
    image_list = list() 
    opt_image_list = list()
    for img_name in currdir:
       print(img_name) 
       image_list.append(load_image(img_name))
    
    currdir = os.listdir('C:\\Users\\papa\\Documents\\GitHub\\ABSA\\hasiee_humanTrafficking\\dataset\\UCSDped1\\GAN self\\A\\train\\') 
    for img_name in currdir:
        print(img_name)
        opt_image_list.append(load_image(img_name))
       
    print(len(image_list))
    print(len(opt_image_list))
       
    image_list, opt_image_list = flat(image_list), flat(opt_image_list)
    return np.array(image_list), np.array(opt_image_list)
       
    
x_img, x_img_opt = image_to_npmat()


# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(10)

# The dimension of our random noise vector.
random_dim = 112812

def load_minst_data():
    # load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize our inputs to be in the range[-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    
    return (x_train, y_train, x_test, y_test)
  
def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)
  
def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(random_dim, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator
  

def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(1024, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator

def get_gan_network(discriminator, random_dim, generator, optimizer):
    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(random_dim,))
    # the output of the generator (an image)
    x = generator(gan_input)
    # get the output of the discriminator (probability if the image is real or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

# Create a wall of generated MNIST images
def plot_generated_images(epoch, generator, examples=1, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)
    print(generated_images)
    
  
def train(epochs=1, batch_size=1):
    # Get the training and testing data
    #x_train, y_train, x_test, y_test = load_minst_data()
    # Split the training data into batches of size 128
    #x_img, x_img_opt = image_to_npmat()
    print('start')
    x_img_t = (x_img.astype(np.float32) - 127.5)/127.5
    x_img_opt_t = (x_img_opt.astype(np.float32) - 127.5)/127.5
    batch_count = x_img.shape[0] / batch_size

    # Build our GAN netowrk
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, random_dim, generator, adam)

    for e in range(1, epochs+1):
        
        for i in range(int(batch_count)):
            # Get a random set of input noise and images
            j = np.random.randint(0, x_img.shape[0], size=batch_size)
            image_batch = x_img_t[j]
            image_batch_opt = x_img_opt_t[j]
            # Generate fake MNIST images
            generated_images = generator.predict(image_batch)
            X = np.concatenate([image_batch_opt, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(image_batch, y_gen)
        print('over')

        #if e == 1 or e % 20 == 0:
         #   plot_generated_images(e, generator)

if __name__ == '__main__':
    train(400, 128) 