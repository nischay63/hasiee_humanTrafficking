# -*- coding: utf-8 -*-
"""
DCGAN for Analoly detection

@author: Saurabh
"""

from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras.layers import Input

from keras.datasets import mnist
import numpy as np
from PIL import Image
import math

import os

import glob
import numpy as np
import os.path as path
from scipy import misc


class DCGAN():
    def __init__(self):
        self.image_row = 158
        self.image_cols = 238
        self.channel = 1
        self.image_shape = (self.image_row , self.image_cols , self.channel)
        
        optimizer = Adam(0.0002, 0.5)
        
        
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer)
        self.generator = self.build_generator()
        self.generator.compile(loss= 'binary_crossentropy', optimizer = optimizer)
        
        self.gan = Sequential()
        self.gan.add(self.generator)
        self.gan.add(self.discriminator)
        self.gan.compile(loss= 'binary_crossentropy', optimizer = optimizer)
        
        
        
        
        
        
    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64,(5,5),input_shape=self.image_shape,activation = 'tanh'))
        model.add(MaxPooling2D(2,2))
        model.add(Conv2D(128,(5,5),activation = 'tanh'))
        model.add(MaxPooling2D(2,2))
        model.add(Flatten())
        model.add(Dense(1024,activation = 'tanh'))
        model.add(Dense(1,activation = 'sigmoid'))
        
        print (model.summary())
        
        img  = Input(self.image_shape)
        valid = model(img)
        return Model(img,valid)
        
          
    def build_generator(self):
        
        noise_shape = (100,)
        model = Sequential()
        
        model.add(Dense(1024,input_shape=noise_shape,activation = 'tanh'))
        model.add(Dense(128*79*119))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((79,119,128),input_shape = (128*79*119,)))
#         model.add(UpSampling2D(size = (2,2)))
#         model.add(Conv2D(64,(5,5),padding = 'same',activation = 'tanh'))
        model.add(UpSampling2D(size = (2,2)))
        
        model.add(Conv2D(1,(5,5),padding = 'same',activation = 'tanh'))
        
        print (model.summary())
        
        return model
    
    def combine_images(self,generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[1:3]
        image = np.zeros((height*shape[0], width*shape[1]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
                img[:, :, 0]
        return image
   
    def load_data(self):
        print ("loading data from", os.getcwd())
        IMAGE_PATH = os.getcwd() +  '/dataset/UCSDped1/A/train'
        file_paths = os.listdir(IMAGE_PATH)
        for i in file_paths:
            print (i)
        images = [misc.imread(IMAGE_PATH +'/' +path) for path in file_paths]
        images = np.asarray(images) 
        print (images.shape)
        #image_size = np.asarray([images.shape[1], images.shape[2], images.shape[3]])
        # print(image_size)
        return images 
    
    def train(self,epochs,batch_size,save_interval):
        
        X_train = self.load_data()
        # (temp,_),(_,_) = mnist.load_data()
        # print (temp.shape)
        X_train = (X_train.astype(np.float32)-127.5)/127.5
        
        for epoch in range(epochs):
            #print ('Epoch : ',epoch)
            num_batches = int(X_train.shape[0]/batch_size)
            #print ('Number of Batches : ', int(X_train.shape[0]/batch_size))
            
            for index in range(num_batches):
                z_noise = np.random.uniform(-1,1,size = (batch_size,100))
                
                
                real_img = np.ndarray((batch_size,158,238,1))
                for i in range(index*batch_size,(index+1)*batch_size):
                    real_img[i-index*batch_size] = X_train[i].reshape(X_train[i].shape[0],X_train[i].shape[1],1)
                    
                    
                
                
                gen_img = self.generator.predict(z_noise,verbose = 0)
                
               
                d_loss_real  = self.discriminator.train_on_batch(real_img,np.ones((batch_size,1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_img,np.zeros((batch_size,1)))
                
                d_loss = d_loss_real+d_loss_fake
                d_loss/=2
                z
            
                self.discriminator.trainable = False
                
                '''Train #1'''
                g_loss_tr1 = self.gan.train_on_batch(z_noise,np.ones((batch_size,1)))
                
                '''Train #2'''
                z_noise = np.random.uniform(-1,1,size = (batch_size,100))
                gen_img = self.generator.predict(z_noise,verbose = 0)
                g_loss_tr2 = self.gan.train_on_batch(z_noise,np.ones((batch_size,1)))
                
                print ('Epoch :%d, Batch Number: %d, D_loss: %f, G_loss_train_1: %f, G_loss_train_2: %f'%(epoch,index,d_loss,g_loss_tr1,g_loss_tr2))
                
            if epoch%save_interval:
                image = self.combine_images(gen_img)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save('images'+
                    str(epoch)+"_"+str(index)+".png")
                
                
                
                
                
if __name__=='__main__':
    print ('The Process is about to start \n\n\n\n\n\n\n')
    dcgan = DCGAN()
    dcgan.train(20,1,5)
    
    
    

