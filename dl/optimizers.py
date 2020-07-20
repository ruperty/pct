# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:22:55 2020

@author: rupert
"""

import tensorflow as tf
import numpy as np
from pct.utilities.rmath import sigmoid
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.util.tf_export import keras_export




class RegressionCase(object):

  def __init__(self, model, optimizer):
    self.model = model
    self.optimizer=optimizer
    

  def __call__(self):
    self.model.update(self.optimizer(self.model))
  

class EcoliPeriodic(object):

  def __init__(self, weights, learning_rate, lossfn):
    self.previous_loss = 0
    self.dWeights = np.random.uniform(-1,1,weights)
    self.lossfn=lossfn
    self.learning_rate = learning_rate

  def __call__(self, model):
    current_loss = self.lossfn(model.outputs, model(model.inputs))
    if current_loss>=self.previous_loss:
        self.dWeights = np.random.uniform(-1,1,2)
    
    self.previous_loss = current_loss
    self.updates = [sigmoid( current_loss*self.learning_rate * self.dWeights[0], 5, 0.1), 
                    sigmoid( current_loss*self.learning_rate * self.dWeights[1], 5, 0.1)]

    return self.updates




@keras_export("keras.optimizers.Ecoli")
class Ecoli(optimizer_v2.OptimizerV2):
  """Ecoli optimizer.

  Computes:
  ```
  if loss has increased
  theta(t+1) = theta(t) - learning_rate * loss * random_gradient
  ```

  # References
      Powers, WT
      http://www.livingcontrolsystems.com/demos/ecoli/ecoli.pdf
  """

  def __init__(self,
               learning_rate=0.01,
               name="Ecoli",
               **kwargs):
    """Construct a new Ecoli optimizer.

    Arguments:
      learning_rate: float hyperparameter >= 0. Learning rate.
      name: Optional name prefix for the operations created when applying
        gradients.  Defaults to 'Ecoli'.
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
    """
    super(Ecoli, self).__init__(name, **kwargs)
    self.loss_historical=tf.Variable(0.0)
    self.learning_rate=learning_rate

  def minimize(self, loss, var_list, grad_loss=None, name=None):
        grads_and_vars = self._compute_gradients(
            loss, var_list=var_list, grad_loss=grad_loss)
    
        return self.apply_gradients(grads_and_vars, name=name)

  def _resource_apply_dense(self, grad, var, apply_state=None):
      #print(grad.numpy())
      #self.minimize( loss, var):
      tf.print(grad)      
      tf.print(var)      
      return apply_ecoli(var, self.learning_rate, grad, self.loss_historical)

  def _compute_gradients(self, loss, var_list, grad_loss=None):
        loss_value = loss()
        print("ls", loss_value)
        print("hlv", self.loss_historical)
        print("wts", var_list)
        
        if loss_value >= self.loss_historical:
            grads = tf.random.uniform(shape=(len(var_list),), minval=-1., maxval=1.)
            
        self.loss_historical.assign(loss_value)
        
        print("dws", grads)
        #var_list += grads
        grads_and_vars = list(zip(grads, var_list))
        self._assert_valid_dtypes([
            v for g, v in grads_and_vars
            if g is not None and v.dtype != dtypes.resource
        ])
    
        return grads_and_vars

  def _create_slots(self, var_list):
    pass #x=0 # do nothing



  def get_config(self):
    config = super(Ecoli, self).get_config()
    config.update({
        "learning_rate": self._serialize_hyperparameter("learning_rate"),
    })
    return config


def apply_ecoli(var, lr, grad, loss):
    result = var + lr * grad * loss
    return result
    