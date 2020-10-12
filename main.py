import tensorflow as tf
import os,random
import cv2 as cv
import numpy as np
import glob
from data_functions import get_image_pair_fnames,iou_coef,dice_coef
from DataGen import DataGenerator
from Display import metric_disp, plot_imgs, disp_samples
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import ResNet101,ResNet50
from tensorflow.keras.preprocessing.image import load_img
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

#Now we take the ouput of the ResNet50 and save it to a vairable
def Model_PSP(pre_trained_model, num_classes = 35):
    last_pretrained_layer = pre_trained_model.get_layer('conv3_block4_out')
    last_output = last_pretrained_layer.output
    last_output = layers.Conv2D(filters = 128, kernel_size = (1,1), name = 'Compress_out')(last_output)

    #Define the params for the pooling module
    #This has to be 1/4 times the input channel depth
    INPUT_CHANNEL_DEPTH = 128
    INPUT_DIM = 32
    TARGET_CHANNEL_DEPTH = INPUT_CHANNEL_DEPTH/4
    Y_KERNEL_DIM = (INPUT_DIM//2,INPUT_DIM//2)
    B_KERNEL_DIM = (INPUT_DIM//4,INPUT_DIM//4)
    G_KERNEL_DIM = (INPUT_DIM//8,INPUT_DIM//8)
    #Now we define the pyramidal pooling architecture
    base = last_output
    #Define the GAP with 1*1 block size for 1x1 bin
    red_blk = layers.GlobalAvgPool2D(name = 'red_block_pooling')(base)
    red_blk = layers.Reshape((1,1,INPUT_CHANNEL_DEPTH))(red_blk)
    red_blk = layers.Conv2D(filters = TARGET_CHANNEL_DEPTH,kernel_size = (1,1),name = 'red_1x1_conv')(red_blk)
    red_blk = layers.UpSampling2D(size = (256,256),interpolation = 'bilinear', name = 'red_upsample')(red_blk)

    #Define the average pooling for the yellow block for 2x2 bin
    y_blk = layers.AvgPool2D(pool_size = Y_KERNEL_DIM, name = 'yellow_blk_pooling')(base)
    y_blk = layers.Conv2D(filters = TARGET_CHANNEL_DEPTH, kernel_size = (1,1), name = 'yellow_1x1_conv')(y_blk)
    y_blk = layers.UpSampling2D(size = (128,128),interpolation = 'bilinear', name = 'yellow_upsample')(y_blk)

    #Define the average pooling for the blue block for 4x4 bin
    blue_blk = layers.AvgPool2D(pool_size = B_KERNEL_DIM, name = 'blue_blk_pooling')(base)
    blue_blk = layers.Conv2D(filters = TARGET_CHANNEL_DEPTH, kernel_size = (1,1), name = 'blue_1x1_conv')(blue_blk)
    blue_blk = layers.UpSampling2D(size = (64,64), interpolation = 'bilinear', name = 'blue_upsample')(blue_blk)

    #Define the average pooling for the green block for 8x8 bins
    green_blk = layers.AvgPool2D(pool_size = G_KERNEL_DIM, name = 'green_blk_pooling')(base)
    green_blk = layers.Conv2D(filters = TARGET_CHANNEL_DEPTH, kernel_size = (1,1), name = 'green_1x1_conv')(green_blk)
    green_blk = layers.UpSampling2D(size = (32,32), interpolation = 'bilinear', name ='green_upsample')(green_blk)

    #Now we upsample the base and check all output shapes to ensure that they match
    base = layers.UpSampling2D(size = (256//INPUT_DIM,256//INPUT_DIM), interpolation = 'bilinear', name = 'base_upsample')(base)
    print(base.get_shape)
    print(red_blk.get_shape)
    print(y_blk.get_shape)
    print(blue_blk.get_shape)
    print(green_blk.get_shape)

    #Generate the final output and check shape
    PPM = tf.keras.layers.concatenate([base, green_blk, blue_blk, y_blk, red_blk])
    print(PPM.get_shape)

    #Now we define the final convolutional block
    output = layers.Conv2D(filters = num_classes, kernel_size = (3,3), padding = 'same', name = 'final_3x3_conv_blk', activation = 'softmax')(PPM)
    return output

#Print the current working directory for a quick sanity check
cwd = os.getcwd()
#Use when running on local runtime
cwd = os.path.join(cwd,'Data')
print(cwd)
#Get the variables
train_pairs = get_image_pair_fnames(cwd,'train')
val_pairs = get_image_pair_fnames(cwd,'val')
params = {'dims':(256,256,3),
            'batch_size':32,
            'shuffle':True}
train_datagen = DataGenerator(train_pairs,**params)
val_datagen = DataGenerator(val_pairs, **params)
print('ho')
#Now we build the model
#We begin by defining the ResNet101 to be used in the netowrk
pre_trained_model = ResNet50(input_shape = (256,256,3),
                             include_top = False,
                             weights = None)
output = Model_PSP(pre_trained_model, num_classes = 35)
#We plot the model for better clarity
#Compile the model and check summary/plot it
model = models.Model(pre_trained_model.input,output)
model.compile(optimizer = 'adam',
              loss="categorical_crossentropy",
              metrics = ['acc',iou_coef,dice_coef])

history = model.fit(train_datagen, validation_data = val_datagen, epochs = 30)
metric_disp(history)
disp_samples(6)
