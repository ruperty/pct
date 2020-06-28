# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 20:27:36 2020

@author: rupert
"""

import matplotlib
import matplotlib.pyplot as plt


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    #print(backend)
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)
        
        
class SubPlotter(object):

  def __init__(self, width, height, title, plotsconfig=[["title", "xlabel", "ylabel", 1, 0, 111]]):
      

    self.fig = plt.figure(figsize=[width, height])
    self.fig.canvas.set_window_title(title)
    
    self.plots=[]
    self.colors=['b',  'r', 'g', 'c', 'm', 'y', 'k', 'w']

    for plotconfig in plotsconfig:
        #print(plotconfig)
        ys=[]
        for line in range(plotconfig[3]):
            ys.append([])
        plot = dict([("title", plotconfig[0]), ("xlabel", plotconfig[1]), 
                     ("ylabel", plotconfig[2]), ("window", plotconfig[4]), 
                     ("subplot", plt.subplot(plotconfig[5])), ("x", []), ("ys", ys)])
        #print(plot)

        self.plots.append(plot)
    
  def add_data(self, index, x, ys):
    #print(ys)
    plot= self.plots[index]
    plot["x"].append(x)
    for i in range(len(ys)):
        plot["ys"][i].append(ys[i])
    window = plot["window"]
    if window > 0:
        if x > window:
            plot["x"].pop(0)
            for i in range(len(ys)):
                plot["ys"][i].pop(0)
            
        
  def show(self):
    plt.show()
        
  def draw(self):
    for plot in self.plots:
        plot["subplot"].clear()
        #print(plot["ys"])
        ctr=0
        for y in plot["ys"]:
            plot["subplot"].plot(plot["x"], y, self.colors[ctr])
            ctr+=1
        plot["subplot"].set_title(plot["title"])
        plot["subplot"].set_xlabel(plot["xlabel"])
        plot["subplot"].set_ylabel(plot["ylabel"])
        #self.ax1.margins(x=5,y=10)

    plt.tight_layout()


class CartpolePlotter(object):

  def __init__(self, width, height, title, xlabel, ylabel):
    self.global_error, self.xs= [],[]
    self.fig = plt.figure(figsize=[width, height])
    self.title=title
    self.xlabel=xlabel
    self.ylabel=ylabel
    self.ax1 = plt.subplot(121)
    
  def add_data(self, epoch, error):
    self.xs.append(epoch)
    self.global_error.append(error)    

        
  def draw(self):
    self.ax1.clear()
    self.ax1.plot(self.xs, self.global_error)
    self.ax1.set_title(self.title)
    self.ax1.set_xlabel(self.xlabel)
    self.ax1.set_ylabel(self.ylabel)
    #self.ax1.margins(x=5,y=10)
    plt.tight_layout()