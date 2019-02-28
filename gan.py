
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from pathlib import Path
import pickle
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU

from keras.optimizers import Adam
from keras import initializers


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
    os.chdir(currdir[2])
    currdir = os.listdir()
    image_list = list() 
    opt_image_list = list()
    image_list_test = list()
    opt_image_list_test = list()
    for img_name in currdir:
 #      print(img_name) 
       image_list.append(load_image(img_name))
    
    currdir = os.listdir('/home/nischay/Desktop/iitd/trainB/') 
    for img_name in currdir:
       # print(img_name)
        opt_image_list.append(load_image('/home/nischay/Desktop/iitd/trainB/'+img_name))
       
    currdir = os.listdir('/home/nischay/Desktop/test optical flow/A/test/')
    for img_name in currdir:
        print(img_name)
        image_list_test.append(load_image('/home/nischay/Desktop/test optical flow/A/test/'+img_name))
        
    currdir = os.listdir('/home/nischay/Desktop/test optical flow/B/test/')
    for img_name in currdir:
       # print(img_name)
        opt_image_list_test.append(load_image('/home/nischay/Desktop/test optical flow/B/test/'+img_name))
    
          
    image_list, opt_image_list,image_list_test,opt_image_list_test = flat(image_list), flat(opt_image_list),flat(image_list_test), flat(opt_image_list_test)
    return ((np.array(image_list).astype(np.float32) - 127.5)/127.5),((np.array(opt_image_list).astype(np.float32) - 127.5)/127.5),((np.array(image_list_test).astype(np.float32) - 127.5)/127.5),((np.array(opt_image_list_test).astype(np.float32) - 127.5)/127.5)
       
    
global x_img,x_img_opt,test_img,test_opt,x_img_t,x_img_opt_t

    
# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(10)

# The dimension of our random noise vector.
flat_dim = 112812


def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)
  
def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(100, input_dim=flat_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(200))
    generator.add(LeakyReLU(0.2))

    #generator.add(Dense(1024))
    #generator.add(LeakyReLU(0.2))

    generator.add(Dense(flat_dim, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator
  

def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(100, input_dim=flat_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(200))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    #discriminator.add(Dense(100))
    #discriminator.add(LeakyReLU(0.2))
    #discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator

def get_gan_network(discriminator, flat_dim, generator, optimizer):
    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(flat_dim,))
    # the output of the generator (an image)
    x = generator(gan_input)
    # get the output of the discriminator (probability if the image is real or not)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    return gan

# Create a wall of generated MNIST images
def plot_generated_images(img,f_img):
    gen_image = generator.predict(img)
    print(gen_images)
    
  
def train(epochs, batch_size):
    # Get the training and testing data
   # x_train, y_train, x_test, y_test = load_minst_data()
    # Split the training data into batches of size 128
   # print(x_img.shape[0])
    batch_count = int(x_img.shape[0] / batch_size)
    #print(batch_count)
    # Build our GAN netowrk
    adam = get_optimizer()
    generator = get_generator(adam)
    discriminator = get_discriminator(adam)
    gan = get_gan_network(discriminator, flat_dim, generator, adam)

    for e in range(1, epochs+1):
        #print(e)
        for i in range(0,int(batch_count)+1):
           # print(i)
            # Get a random set of input noise and images
            #noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            #print(batch_size)
            #print(i)
            s = i*batch_size
            
            if (i+1)*batch_size <= x_img.shape[0]:
                end = (i+1)*batch_size
            else:
                end = x_img.shape[0]
                
            batch_size_f = end-s
            print(batch_size_f)
            
            image_batch_opt_t = x_img_opt[s:end]
            image_batch_t = x_img[s:end]
            
            # Generate fake MNIST images
            generated_images = generator.predict(image_batch_t)
            X = np.concatenate([image_batch_opt_t, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size_f)
            # One-sided label smoothing
            y_dis[:batch_size_f] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # Train generator
            #noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            
            y_gen = np.ones(batch_size_f)
            discriminator.trainable = False
            gan.train_on_batch(image_batch_opt_t, y_gen)

        print('over')            
           
        #if e == 1 or e % 20 == 0:
            #plot_generated_images(e, generator)

if __name__ == '__main__':
    x_img,x_img_opt,test_img,test_opt = image_to_npmat()
    print(x_img.shape)
    print(x_img_opt.shape)
   
    
  
    train(1, 32) 
