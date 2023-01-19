#Python imports:

#***********************************************************************************
# Master import
#***********************************************************************************

# Plot data in this window
#%matplotlib inline

# Make the ipython cell width the size of the window
#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

# Imports some libraries
import glob
import sys
import time
import numpy as np
import os

# You might need to install pylab, matplotlib, scipy. They  comes with anaconda, amoung others. >> pip install anaconda
import pylab
from pylab import *
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import chi2
from scipy import optimize
import random
import json


import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.autograd import Variable


# The current directory
cwd = os.getcwd() 

# The location where we will save figures
save_location = cwd + '/Figures/' 

if not os.path.isdir(save_location):
    os.mkdir(save_location)




# Define your own colors for the plots, you can also use 'r' for red or 'g' for green.
mycolors = ['#c70039','#ff5733','#ff8d1a','#ffc300','#eddd53','#add45c','#57c785',
               '#00baad','#2a7b9b','#3d3d6b','#511849','#900c3f','#900c3f'] 
import warnings
warnings.filterwarnings("ignore")
print('Imports complete ...')


def SER(V,t):
    tau = 4
    sigma = 0.5
    return V * np.exp(-1/2* (np.log(t/tau)/sigma)**2)


Test = 1
n_events = 3
max_photons = 7
n_samples = 500
n_noise = 0.2 



### Generation of the number of photons for each event.
### A random integer between 0 and max_photons.

n_photons = np.random.randint(0, max_photons+1, size = n_events) ## The +1 is because the function is exclusive in the max value.

t = np.linspace(0,n_samples-1,n_samples)/5


All = []
Truth = []
#n_signals = np.sum(n_photons)
#print('Total number of photons', n_signals)


### Generation of the t0s and of the max values of the peaks up to 5 mV


# plt.hist(V_0)
# plt.show()
# Truth = {'time':t0, 'voltage': V_0}
# print('Truth',Truth)


for ii  in np.arange(n_events):

    ph = n_photons[ii] 

    t0 = np.random.rand(ph)*(80-15)+15
    V_0 = np.random.normal(1,0.35, ph)*5
    ##Generation of noise for this event
    noise = np.random.normal(0,n_noise,n_samples)

    
    ys =  []

    ##Array for saving the True values
    ccs = np.zeros((len(t),2*max_photons))
    for jj in np.arange(ph):

        ###Selection rules

        ####The photon has not hitted
        sel =  t < t0[jj]
        ####The wave form is increasing
        t_max = t0[jj]+4 ##The 4 is from tau in the SER funcion
        sel_inc = (t>=t0[jj])&(t<=t_max)
        ####The wave form is decreasing
        sel_dec = (t >= t_max)

        
        ###For the signal      
        y = SER(V_0[jj],t-t0[jj])
        y[sel] = 0
        ys.append(y)

        ##For setting the different contributions at each time
        
        ccs[:,2*jj][sel_inc] = 1
        ccs[:,2*jj+1][sel_dec] = 1


        #print(ccs)
        
    ys = np.array(ys)
    # print('sum over 0',np.sum(ys, axis =0 ))

    y_final = np.sum(ys, axis = 0)
    y_noise = np.sum(ys, axis = 0)+noise
    
    All.append(y_noise)
    Truth.append(ccs)

    print(y_final.shape, y_noise.shape)

    plt.plot(t,y_final)
    plt.plot(t,y_noise)

    plt.show()
    plt.savefig('Figures/Event_'+str(ii)+'_Test_'+str(Test)+'.png')
    plt.clf()

    

    ##For plotting the max of each signal
    # plt.scatter(4+t0,V_0)

    

All = np.array(All)
#print(All.shape)

All = torch.from_numpy(All)
Truth = torch.Tensor(Truth)

print(All.type(), Truth.type())

Name = 'Train_'+str(Test)+'_02'

print('Just a test')

torch.save(Truth, 'Truth_'+Name)
print('Second Test')
torch.save(All, Name)

new_All = torch.load(Name)
new_Truth = torch.load('Truth_'+Name)

print(new_Truth)

    # truth = [2.5]
    # #print(yn)
    # np.save("test",yn)
    # new_yn = np.load("test.npy")
    # V0 = np.random.normal(1,0.35,new_yn)*5
    # plt.hist(V0)
    # plt.show()






