# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:44:52 2020

@author: ryoung
"""


from plots.plotlys import  SingleScatterPlot
from plots.plotlys import  MultipleScatterSubPlots
from plots.plotlys import add_point_to_widget
from plots.plotlys import add_point_to_subplot_widget
import numpy as np


class BaseSpecificPlots():
    #def add_point(self, subplot, trace, x, y):    
    #    self.figure.add_point(subplot, trace, x, y)
 
    def show(self):
        self.figure.show()

    def getFigure(self):
        return self.figure.getFigure()



class CartpolePositionPlot(BaseSpecificPlots):
    def __init__(self, width, height):
        trace_names=["pole_position_ref", "pole_position"]
        self.figure = SingleScatterPlot("Pole Postion", trace_names,  2,  width, height)

    def add_points(self, ctr, pole_position_ref, pole_position):
        self.figure.add_point(0, ctr, pole_position_ref)
        self.figure.add_point(1, ctr, pole_position)



class CartpoleControlPlots(BaseSpecificPlots):
    def __init__(self,sub_title, width, height):
        subplots_titles=["pole_angle", "pole_velocity", "cart_position", "cart_velocity", "action", "global error (rms)"]
        trace_names=["pole_angle_ref", "pole_angle", "pole_velocity_ref", "pole_velocity", 
                     "cart_position_ref","cart_position", "cart_velocity_ref", "cart_velocity",
                     "action", "", "error", "serror"]
        self.figure = MultipleScatterSubPlots("Control Unit Values"+sub_title, subplots_titles, trace_names, 6, 1, 2,  width, height)

        
    def add_points(self, ctr, pole_angle_ref, pole_angle, pole_velocity_ref, pole_velocity, 
                   cart_position_ref, cart_position, cart_velocity_ref, cart_velocity, action, error, serror):
        self.figure.add_point(0, 0, ctr, np.rad2deg(pole_angle_ref))
        self.figure.add_point(0, 1, ctr, np.rad2deg(pole_angle))
        self.figure.add_point(1, 0, ctr, pole_velocity_ref)
        self.figure.add_point(1, 1, ctr, pole_velocity)
        self.figure.add_point(2, 0, ctr, cart_position_ref)
        self.figure.add_point(2, 1, ctr, cart_position)
        self.figure.add_point(3, 0, ctr, cart_velocity_ref)
        self.figure.add_point(3, 1, ctr, cart_velocity)
        if action==0: action = -1
        self.figure.add_point(4, 0, ctr, action)
        self.figure.add_point(5, 0, ctr, error)
        self.figure.add_point(5, 1, ctr, serror)


def add_cartpolepoints_to_widget(widget, tps, ctr, pole_angle_ref, pole_angle, pole_velocity_ref, pole_velocity, 
                   cart_position_ref, cart_position, cart_velocity_ref, cart_velocity, action, error, serror):
        add_point_to_subplot_widget(widget, tps,0, 0, ctr, np.rad2deg(pole_angle_ref))
        add_point_to_subplot_widget(widget, tps,0, 1, ctr, np.rad2deg(pole_angle))
        add_point_to_subplot_widget(widget, tps,1, 0, ctr, pole_velocity_ref)
        add_point_to_subplot_widget(widget, tps,1, 1, ctr, pole_velocity)
        add_point_to_subplot_widget(widget, tps,2, 0, ctr, cart_position_ref)
        add_point_to_subplot_widget(widget, tps,2, 1, ctr, cart_position)
        add_point_to_subplot_widget(widget, tps,3, 0, ctr, cart_velocity_ref)
        add_point_to_subplot_widget(widget, tps,3, 1, ctr, cart_velocity)
        if action==0: action = -1
        add_point_to_subplot_widget(widget, tps,4, 0, ctr, action)
        add_point_to_subplot_widget(widget, tps,5, 0, ctr, error)
        add_point_to_subplot_widget(widget, tps,5, 1, ctr, serror)
        
        
def add_cartpole_positions_to_widget(widget, ctr, pole_position_ref, pole_position):
    add_point_to_widget(widget, 0, ctr, pole_position_ref)
    add_point_to_widget(widget, 1, ctr, pole_position)

    

class CartpoleControlUnitPlots(BaseSpecificPlots):
    def __init__(self,sub_title):
        subplots_titles=["pole_angle", "pole_velocity", "cart_position", "cart_velocity"]
        trace_names=["pole_angle_ref", "pole_angle", "pole_velocity_ref", "pole_velocity", 
                     "cart_position_ref","cart_position", "cart_velocity_ref", "cart_velocity"]
        self.figure = MultipleScatterSubPlots("Control Unit Values"+sub_title, subplots_titles, trace_names, 4, 1, 2)

        
    def add_points(self, ctr, pole_angle_ref, pole_angle, pole_velocity_ref, pole_velocity, 
                   cart_position_ref, cart_position, cart_velocity_ref, cart_velocity):
        self.figure.add_point(0, 0, ctr, np.rad2deg(pole_angle_ref))
        self.figure.add_point(0, 1, ctr, np.rad2deg(pole_angle))
        self.figure.add_point(1, 0, ctr, pole_velocity_ref)
        self.figure.add_point(1, 1, ctr, pole_velocity)
        self.figure.add_point(2, 0, ctr, cart_position_ref)
        self.figure.add_point(2, 1, ctr, cart_position)
        self.figure.add_point(3, 0, ctr, cart_velocity_ref)
        self.figure.add_point(3, 1, ctr, cart_velocity)
    
        
        
class CartpoleControlValuesPlots(BaseSpecificPlots):
    def __init__(self,sub_title):
        subplots_titles=["Action", "Global Error"]
        trace_names=["action", "error"]
        self.figure = MultipleScatterSubPlots("Control Values"+sub_title, subplots_titles, trace_names, 2, 1, 1)


    def add_points(self, ctr, action, error):
        if action==0: action = -1
        self.figure.add_point(0, 0, ctr, action)
        self.figure.add_point(1, 0, ctr, error)
    