# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 20:56:17 2020

@author: rupert
"""


import pct.utilities.rmath as rm

class CartpoleData(object):

  def __init__(self, loss_fn, loss_smooth):
    self.global_error= 0
    self.loss_fn= loss_fn
    self.loss_smooth= loss_smooth
 
    
  def add_error_data(self, current, target):
    self.error=self.loss_fn(current, target)
    self.global_error=rm.smooth( self.error, self.global_error, self.loss_smooth) 
    #print(self.error, self.global_error)
    return self.global_error
        
  def get_error(self):
      return self.global_error

  def get_local_error(self):
      return self.error
