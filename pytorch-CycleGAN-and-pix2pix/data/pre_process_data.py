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


def image_to_npmat():
    currdir = os.listdir()
    print(currdir)
    os.chdir(currdir[0])
    #exit()
    currdir = os.listdir()
    os.chdir(currdir[0])
    currdir = os.listdir()
    print(currdir)
    exit()
    image_list = list()    
    for img_name in currdir:
        image_list.append(load_image(img_name))
    file_img = 'img_mat'
    outfile = open(file_img,'wb')
    pickle.dump(image_list,outfile)
    outfile.close()
    print(len(image_list))
    
image_to_npmat()

    
