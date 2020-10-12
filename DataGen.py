import tensorflow as tf
import os,random
import cv2 as cv
import numpy as np
import glob
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import ResNet101,ResNet50
from tensorflow.keras.preprocessing.image import load_img
import tensorflow.keras.backend as K
#A trial with an implementation using DataGenerators for using the memory more efficiently
class DataGenerator(Sequence):
  #Generates data for the model down the line

  def __init__(self,path_list,batch_size = 32, dims = (256,256,3), shuffle = True):
   #This is define if the data is from the train set or the validation set
   #self.path = path
   self.list_IDs = path_list
   #Define the batch size and if we want to shuffle on each iteration
   self.batch_size = batch_size
   self.shuffle = shuffle
   self.dims = dims
   self.on_epoch_end()

  def on_epoch_end(self):
    #Define the things to do at the end of each epoch
    #Generate the path for the data
    self.indexes = np.arange(len(self.list_IDs))
    if(self.shuffle == True):
      np.random.shuffle(self.indexes)

  def __getitem__(self,index):
    #Generate the lists of indices for the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    #Find the list of IDs for the given batch
    list_IDs_temp = [self.list_IDs[k] for k in indexes]

    #Generate the data
    X,y = self.__data_generation(list_IDs_temp)
    return X,y



  def __len__(self):
    #Return the number of batches per epoch during the training process
    return int(np.floor(len(self.list_IDs) / self.batch_size))

  def __data_generation(self,list_IDs_temp):
    #Generates the data in batches that we specify
    #We begin by taking the dstype_pair and doing some quick pre-processing in every batch

    img_dat = np.zeros((self.batch_size,) + (self.dims[0],self.dims[1]) + (3,), dtype="float32")
    img_mask = np.zeros((self.batch_size,) + (self.dims[0],self.dims[1]) + (35,), dtype="uint8")
    for j,ID_pair in enumerate(list_IDs_temp):
      #We extract the label and image from each pair
      mask = cv.imread(ID_pair[0],0)
      image = cv.imread(ID_pair[1])
      image = image/255.0

      #We resize both so that it fits the input of the ResNet
      mask = cv.resize(mask, (self.dims[0],self.dims[1]), interpolation = cv.INTER_NEAREST)
      image = cv.resize(image, (self.dims[0],self.dims[1]), interpolation = cv.INTER_NEAREST)

      #Now we one hot encode the mask
      mask = tf.keras.utils.to_categorical(mask,num_classes = 35)
      #Now we append the images and the masks to the appropriate location
      img_mask[j] = mask
      img_dat[j] = image

    #Return the generated data values
    return img_dat,img_mask
