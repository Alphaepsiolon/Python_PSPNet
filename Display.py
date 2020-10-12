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
import matplotlib.pyplot as plt

def metric_disp(history):
    fig = plt.gcf()
    axs = plt.gca()
    axs.set_ylim([0,1])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    fig2 = plt.gcf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    fig3 = plt.gcf()
    plt.plot(history.history['iou_coef'])
    plt.plot(history.history['val_iou_coef'])
    plt.title('model iou_coef')
    plt.ylabel('iou_coef')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    fig3 = plt.gcf()
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model dice_coef')
    plt.ylabel('dice_coef')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_imgs(img,mask,pred):
  fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,10))
  ax1.imshow(img)
  ax1.axis('off')
  ax2.imshow(mask, cmap = 'gray')
  ax2.axis('off')
  ax3.imshow(pred, cmap = 'gray')
  ax3.axis('off')

def disp_samples(batch_no_to_check):
    #Quick sanity sceck to see if the generated labels are reasonable
    batch_no_for_test = 6
    x,y = val_datagen.__getitem__(batch_no_for_test)
    pred_masks = model.predict(val_datagen.__getitem__(batch_no_for_test))
    print(pred_masks.shape)
    print(x.shape)
    print(y.shape)
    for i in range(3):
      mask = tf.argmax(pred_masks[i], axis=-1)
      print('Labels in the prediction:',np.unique(mask))
      print('Labels in the ground truth:',np.unique((tf.argmax(y[i],axis = -1))))
      plot_imgs(x[i],np.squeeze(tf.argmax(y[i], axis = -1)),np.squeeze(mask))
