# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 11:28:04 2020

@author: rupert
"""


import tensorflow as tf
import itertools
import gym
import math
import pct.utilities.rmath as rm
import numpy as np


from tensorflow import keras
from pct.dl.utils import get_optimizer
from pct.dl.utils import get_loss_function
from pct.openai.utils import EnvState
from pct.plots.matplots import SubPlotter
from pct.utilities.rmath import Counter
from pct.openai.modeldata import CartpoleData
from pct.plots.matplots import move_figure
from pct.openai.cpplots import CartpoleDataPlot

from enum import Enum



class PCTVarType(Enum):
    REFERENCE = 1
    PERCEPTION = 2
    ERROR = 3


class CartpoleTuning(object):

    def __init__(self, model_name, env_name, max_episode_steps, trainable, print=True):
        self.print=print
        self.live_display=False
        self.model_name=model_name
        self.create_model(model_name, trainable)
        self.create_env(env_name, max_episode_steps)
        self.create_dataset()
        self.prev_power=0
        
    def configure(self, weights, opt_type, learning_rate, loss_type,  loss_smooth=0.9, plot=13, print=100, num_runs=100 ):
        self.learning_rate=learning_rate
        self.opt_type=opt_type
        self.loss_type=loss_type 
        self.optimizer = get_optimizer(opt_type, learning_rate)
        self.loss_fn= get_loss_function(loss_type)        
        self.set_weights(weights)
        self.counter = Counter(limit=num_runs, plot=plot, print=print)
        self.data = CartpoleData(self.loss_fn, loss_smooth)

    def not_finished(self):
        return self.counter.get() < self.counter.get_limit()
        
    def set_weights(self, weights):
        wts = self.model.get_weights()
        #print("wts ", wts)
        for wt in range(len(weights)):
            wts[wt][0]=weights[wt]
            
        self.model.set_weights(wts)
        #if self.print:
        #   print("starting weights     %-10.3f %-10.3f %-10.3f %-10.3f" % (wts[0][0],wts[1][0],wts[2][0],wts[3][0]))

        
    def batch(self, epoch, batch_size, training=False):
        for inputs, desired_output in self.ds_counter.take(batch_size):
            if training:
                loss_value=self.training_step(inputs, desired_output)
            else:
                loss_value=self.non_training_step(inputs, desired_output)
             
            self.display(loss_value)
            
            self.counter()
            if self.env_state():
                wts = self.model.get_weights()
                if self.print:
                   #print("finished")
                   #print("wts ", np.concatenate(np.concatenate(wts)))
                   #print(inputs)
                   #print(inputs['perception_pa'])
                   print("Cartpole failed with a poleangle of %4.2f degrees and global error of %4.2f."
                          % (math.degrees( inputs['perception_pa']), loss_value))
                   print("An optimised control system for this case would have residual error of around 0.20 or less")
                                           
                self.counter.set_limit(epoch) 
                break 
            
        return self.model.get_weights()
    
    
    def display(self, loss_value):
        if (self.counter.get()+1) % self.counter.plot ==0 :
            if self.live_display:
                self.plotter.add_data(0, self.counter.get(), [loss_value, self.data.get_local_error()])
                self.plotter.draw()
            if self.render:
                self.env.render()
                
        if self.widget != None:
            if self.widget_frequency % 100 == 0:
                newy = self.widget.data[0].y + loss_value
                self.widget.data[0].y=newy
        
        if (self.counter.get()+1) % self.counter.print == 0:
            wts = self.model.get_weights()
            if self.print:
               print("%4d loss %-10.3f %-10.3f %-10.3f %-10.3f %-10.3f " % (self.counter.get()+1, loss_value, wts[0][0],wts[1][0],wts[2][0],wts[3][0]))
        


    def training_step(self, inputs, target):

        with tf.GradientTape() as tape:
           model_outputs = self.model(inputs, training=True)
           loss = self.data.add_error_data(model_outputs[1], target)
           #loss = self.data.add_error_data([1,1,5,1]* model_outputs[1], target)
           
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        loss_value = self.data.get_error()
        if self.offline:
            plot_errors =  model_outputs[1][0].numpy()
            plot_errors = np.append(plot_errors,loss_value)
            #print(plot_errors)
            self.offline_figure.add_points(self.counter.get(),plot_errors )
        
        self.env_state.set_action(self.get_action(model_outputs[0][0][0].numpy()))
        return loss_value
            
    def non_training_step(self, inputs, target):

        model_outputs = self.model(inputs)  
        self.data.add_error_data( model_outputs[1], target)
        loss_value = self.data.get_error()

        if self.offline:
            plot_errors =  model_outputs[1][0].numpy()
            plot_errors = np.append(plot_errors,loss_value)
            self.offline_figure.add_points(self.counter.get(),plot_errors )
                    
        self.env_state.set_action(self.get_action(model_outputs[0][0][0].numpy()))
        return loss_value
            
    def get_action(self, power):
        action=0
        power= rm.smooth(power, self.prev_power, 0.75)
        self.prev_power=power
        if power>=0:
          action=1
        
        return action
        
            
    def display_configure(self, x=200, y=200, width=5, height=4, window=1000, live=True, 
                          offline=True, render=True, plot_type=PCTVarType.ERROR, widget=None, widget_frequency=100):
        self.widget=None
        self.widget_frequency=widget_frequency
        self.live_display=live
        self.offline=offline
        self.render=render
        if live: 
            self.plotter = SubPlotter(width, height, self.model_name, [["Loss", "iter", "error", 2, window, 111]])
            move_figure(self.plotter.fig, x, y)
        if offline:     
            vartitle="Errors"
            if plot_type==PCTVarType.REFERENCE:
                vartitle="Reference"
            if plot_type==PCTVarType.PERCEPTION:
                vartitle="Perception"
                
            self.offline_figure=CartpoleDataPlot(vartitle)

        
    def create_env(self, env_name, max_episode_steps):
        self.env = gym.make(env_name)
        self.env._max_episode_steps = max_episode_steps
        self.env.reset()
        self.env_state = EnvState()
        
    def gen_obs(self):
        for i in itertools.count(1):
            obs, rew,dn,inf = self.env.step(self.env_state.get_action())
            self.env_state.set_state(dn)
            #print(obs, rew,dn,inf)
            yield {"reference_pa":[0.0], "perception_pa":[obs[0]], "perception_pv":[obs[1]], "perception_cp":[obs[2]], 
                   "perception_cv":[obs[3]]}, [0.0, 0.0, 0.0, 0.0]
        
    def create_dataset(self):
        self.ds_counter = tf.data.Dataset.from_generator(self.gen_obs, 
        output_types=({"reference_pa":tf.float32, "perception_pa":tf.float32, "perception_pv":tf.float32, 
                       "perception_cp":tf.float32, "perception_cv":tf.float32}, tf.float32))
        
    def create_model(self, model_name, trainable):
        inputZero = keras.layers.Input(shape=[1], name="reference_pa")
        inputPA = keras.layers.Input(shape=(1,), name="perception_pa")
        inputPV = keras.layers.Input(shape=(1,), name="perception_pv")
        inputCP = keras.layers.Input(shape=(1,), name="perception_cp")
        inputCV = keras.layers.Input(shape=(1,), name="perception_cv")
        
        errorPA = keras.layers.Subtract(name="error_pa")([inputZero, inputPA])
        referencePV = keras.layers.Dense(1, name="reference_pv",use_bias=False,trainable=trainable[0])(errorPA)
        
        errorPV = keras.layers.Subtract(name="error_pv")([referencePV, inputPV])
        referenceCP = keras.layers.Dense(1, name="reference_cp",use_bias=False,trainable=trainable[1])(errorPV)
        
        errorCP = keras.layers.Subtract(name="error_cp")([referenceCP, inputCP])
        referenceCV = keras.layers.Dense(1, name="reference_cv",use_bias=False,trainable=trainable[2])(errorCP)
        
        errorCV = keras.layers.Subtract(name="error_cv")([referenceCV, inputCV])
        output = keras.layers.Dense(1, name="action",use_bias=False,trainable=trainable[3])(errorCV)
        outputs = keras.layers.Concatenate(name="errors")([errorPA, errorPV,errorCP,errorCV])
        
        self.model = keras.models.Model(inputs=[inputZero , inputPA, inputPV, inputCP, inputCV], 
            outputs=[output, outputs], name=model_name)               
        
    def summary(self):
        self.model.summary()
        
    def plot_model(self, filename, show_shapes=False):
        keras.utils.plot_model(self.model, filename, show_shapes=show_shapes) 

    def run(self, batch_size, training=False):
        out=[]
        runs=self.counter.get_limit()
        for epoch in range(runs):
            if self.not_finished():
                out = self.batch(epoch, batch_size, training)
        return out
    
    def get_error_data(self):
        return self.offline_figure.get_data_trace(4)
    
    def show(self):
        if self.offline:
            self.offline_figure.show()
            
    def close(self):
        self.env.close()
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        