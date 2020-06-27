# -*- coding: utf-8 -*-
"""
Created on Sat May 30 11:42:36 2020

@author: rupert
"""

import math

def smooth( newVal, oldVal, weight) :
    return newVal * (1 - weight) + oldVal * weight;



def sigmoid(x, range, scale) :
    return -range / 2 + range / (1 + math.exp(-x * scale / range));
    


class Counter(object):

  def __init__(self, limit, init=0, step=1):
      self.limit=limit
      self.counter=init
      self.step=step

  def __call__(self):
      self.counter+=self.step
  
  def get(self):
      return self.counter
    
  def get_limit(self):
      return self.limit

  def set_limit(self, limit):
      self.limit=limit



"""
rang=0.4
scale=.5

for x in range(-10,10):
    print(x/2.5, sigmoid(x/2.5, rang, scale))
"""