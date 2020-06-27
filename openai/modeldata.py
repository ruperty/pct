# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 20:56:17 2020

@author: rupert
"""

from pct.utilities.errors import root_mean_squared_error
import pct.utilities.rmath as rm

class CartpoleData(object):

  def __init__(self, error_type, error_smooth):
    self.global_error= 0
    self.error_type= error_type
    self.error_smooth= error_smooth
 
    
  def add_error_data(self, errors):
    if self.error_type == "rms":
        self.error=root_mean_squared_error(errors)
        self.global_error=rm.smooth( self.error, self.global_error, self.error_smooth) 
        #print(self.error, self.global_error)
        
  def get_error(self):
      return self.global_error

  def get_local_error(self):
      return self.error
