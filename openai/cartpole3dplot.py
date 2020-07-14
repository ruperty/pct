# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:12:34 2020

@author: ryoung
"""




import numpy
import pct.utilities.timing as tim
from pct.dl.models.cartpole import CartpoleTuning

live_plot = False
offline_plot=False#True#False
training=False#True
render_gym=False#True
batch_size=10
num_runs=1000
prnt=False

order = {"value1":0, "value2":1}

pa=[]
pv=[]
cp=[]
list1=pa
list2=pv


# weights=[1.6,       1,       1,       -0.5]

test =1

if test ==1: # all
    step=1
    scale=10

    pa_start= 0
    pa_stop= 8.0

    pv_start= 0.5
    pv_stop=2.0

if test ==2: # single
    step=1
    scale=10

    pa_start= -1.0
    pa_stop= 10.0

    pv_start= -1
    pv_stop=2.0


pa_ss = '[[{:-04.1f}][{:-04.1f}]'.format(pa_start,pa_stop)
pv_ss = '[{:-04.1f}][{:-04.1f}]]'.format(pv_start,pv_stop)
#print(pv_ss)


for pa_wt in range(int(pa_start* scale), int(pa_stop* scale), step):
    pa.append(pa_wt/scale)

for pv_wt in range(int(pv_start* scale), int(pv_stop* scale), step):
    pv.append(pv_wt/scale)


#for cp_wt in range(2, 30, 1):
#    cp.append(cp_wt/scale)



#print(pa)        
#print(pv)        
#print(cp)        


def run_cartpole(ctr, weights):
    timer = tim.Timer()
    timer.start()
    ct = CartpoleTuning("Cartpole", "CartPole-v1", num_runs+1, [True, True, True, False], print= prnt)
    ct.configure(weights=weights, opt_type="sgd", learning_rate=0.01, loss_type="rsuse",  loss_smooth=0.9, plot=13, print=100, num_runs=num_runs)
    ct.display_configure(x=800, y=200, width=5, height=4, window=1000, live=live_plot, offline=offline_plot, render=render_gym)
    
    wts=ct.run( batch_size, training)        
    actual_runs= ct.counter.get()
    if actual_runs<200:
        mean =0 
    else: 
        mean = ct.data.get_mean().numpy()
        #mean = 1000*ct.data.get_mean().numpy()/actual_runs
        #if mean > 0.5: mean =0.5
    elapsed= timer.stop()
    print("%0.3f ctr %04d runs %04d mean %-7.3f wts %-7.3f %-7.3f %-7.3f %-7.3f " % 
          (elapsed, ctr, actual_runs, mean, wts[0][0],wts[1][0],wts[2][0],wts[3][0]))
    if offline_plot:
        ct.show()
    ct.close()  
    return mean      


def run_batch(list1, list2, order, weights, scale):
    ctr = len(list1)*len(list2)
    losses=[]
    for value2 in list2:
        weights[order["value2"]]=value2/scale
        inner_losses=[]
        for value1 in list1:
            weights[order["value1"]]=value1/scale
            loss=run_cartpole(ctr, weights)
            inner_losses.append(loss)    
            ctr-=1
        losses.append(inner_losses)
    
    return losses


   

def run_data(list1,list2,type,pa_ss, pv_ss,other_wt_name,other_wt,other_wt_index, order):
    weights=[0,       0,       0,       -0.05]
    weights[other_wt_index]=other_wt
    losses=run_batch(list1, list2, order, weights, 1)
    #print(list1)
    #print(list2)
    #print(losses)
    
    data=[]
    data.append(type)
    data.append(other_wt_name)
    data.append(other_wt)
    data.append(list1)
    data.append(list2)
    data.append(losses)
    
    #print(data)

    filename=get_filename(type,pa_ss, pv_ss,other_wt_name,other_wt)
    numpy.save(filename, data)


def get_filename(type,pa_ss, pv_ss, other_wt_name,other_wt):
    return  'data-{:s}-{:s}-{:s}-{:s}-{:03.1f}'.format(type,pa_ss, pv_ss, other_wt_name,other_wt)
    



def run_multiple(pa_ss, pv_ss,other_wt):
    timer = tim.Timer()
    timer.start()
    run_data(list1,list2,type,pa_ss, pv_ss,other_wt_name,other_wt,other_wt_index, order)
    print("Time %0.3f points %d runs %d" % ( timer.stop(), len(list1)*len(list2), num_runs))




type="papv"
other_wt_name="cp"
other_wt_index=2

"""
filen = get_filename(type,pa_ss, pv_ss, other_wt_name,1.5)
print(filen)
x = [1]
numpy.save(filen, x)

"""



if test ==1: # all
    cp_start=0.8
    cp_stop=1.6
    cp_step=1
    cp_scale=10


if test ==2: # single
    cp_start=1
    cp_stop=1.1
    cp_step=1
    cp_scale=10

for wt in range(int(cp_start*cp_scale), int(cp_stop*cp_scale), cp_step):
        run_multiple(pa_ss, pv_ss,wt/cp_scale)

    

 


        
        