# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:34:42 2020

@author: rupert
"""

from PIL import Image
import numpy as np
import dl.losses as pctloss
import dl.metrics as pctmetrics
import dl.optimizers as pctopts

import tensorflow as tf

class BaseImage():
    def __init__(self, path):
        self.path = path 

    def row(self, rowi):
     row = self.imageArray[rowi]
     return row

    def getImage(self):
     return self.image 

class GreyImage(BaseImage):
      
  def __init__(self, path):
    super().__init__( path) 
    
  def open(self):
    self.image  = Image.open(self.path)
    self.rchannel = self.image.getchannel("R")
    self.imageArray =np.asarray(self.rchannel)
    #print (self.image.format, self.image.size, self.image.mode)



def get_optimizer( opt_type, learning_rate):
    opt_type = opt_type.lower()
    
    if opt_type == 'ecoli' :
        optimizer = pctopts.Ecoli(learning_rate)

    if opt_type == 'sgd' :
        optimizer = tf.keras.optimizers.SGD(learning_rate)

    if opt_type == 'adam' :
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    if opt_type == 'adadelta' :
        optimizer = tf.keras.optimizers.Adadelta(learning_rate)

    if opt_type == 'rmsprop' :
        optimizer = tf.keras.optimizers.RMSprop(learning_rate)

    return optimizer
    
def get_metric_type( metrics):
    if len(metrics)>0:        
        if metrics[0] == 'my_mse':
           metrics[0] = pctmetrics.MyMeanSquaredError()
            
        if metrics[0] == 'sse':
           metrics[0] = pctmetrics.SquaredSumError()

        if metrics[0] == 'rsse':
           metrics[0] = pctmetrics.RootSquaredSumError()

    return metrics

def get_loss_function( loss_type):
    loss_name = 'Loss_'+loss_type
    loss_fn=None
    if loss_type == 'my_mse':
        loss_fn = pctloss.MyMeanSquaredError(name=loss_name )
        
    if loss_type == 'sse':
        loss_fn = pctloss.SquaredSumError(name=loss_name )
               
    if loss_type == 'rsse':
        loss_fn = pctloss.RootSquaredSumError(name=loss_name )

    if loss_type == 'rsuse':
        loss_fn = pctloss.RootSumSquaredError(name=loss_name )



    return loss_fn
