import numpy as np 
import torch
from torch.nn import BCELoss
import matplotlib
import matplotlib.pyplot as plt


NN_x = np.array([0.86, 0.26, 0.2 , 0.51, 0.44, 0.78, 0.9 , 0.25, 0.22, 0.68, 0.55,0.06, 0.66, 0.04, 0.17, 0.84, 0.26, 0.81, 0.07, 0.6 , 0.83, 0.26, 0.99, 0.61, 0.75, 0.04, 0.76, 0.16, 0.26, 0.25, 0.56, 0.76])
y = np.array([0., 0., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0.])


#Write out the equation to calculate the cross entropy here, you may need to use np.sum to sum and np.log to calculate logarithmic values
H = np.sum(-y*np.log(NN_x) -(1-y)*np.log(1 - NN_x))


print("The Cross Entropy is %.3f, Average CE per event is %.3f"%(H, H/len(y)))


NN_x = torch.FloatTensor([0.86, 0.26, 0.2 , 0.51, 0.44, 0.78, 0.9 , 0.25, 0.22, 0.68, 0.55,0.06, 0.66, 0.04, 0.17, 0.84, 0.26, 0.81, 0.07, 0.6 , 0.83, 0.26, 0.99, 0.61, 0.75, 0.04, 0.76, 0.16, 0.26, 0.25, 0.56, 0.76])
y = torch.FloatTensor([0., 0., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0.])
BinaryCrossEntropy = BCELoss(reduction="mean")    #reduction="mean" means that we are calculating the BCE loss of each element instead of the whole vector
loss = BinaryCrossEntropy(NN_x,y)
print("The PyTorch calculation of binary cross entropy is %.3f, does it agree with average CE per event in your previous calculation?"%(loss.item()))

##Assuming a new network output after some training (The training is not being done in this code)

NN_x_new = np.array([0.26, 0.16, 0.72 , 0.51, 0.74, 0.28, 0.9 , 0.75, 0.22, 0.68, 0.85,0.06, 0.66, 0.84, 0.67, 0.74, 0.26, 0.81, 0.57, 0.6 , 0.23, 0.26, 0.99, 0.61, 0.35, 0.04, 0.76, 0.16, 0.26, 0.25, 0.56, 0.76])
y = np.array([0., 0., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0.])
#Write out the equation to calculate the cross entropy here, you may need to use np.sum to sum and np.log to calculate logarithmic values
H = np.sum(-y*np.log(NN_x_new) -(1-y)*np.log(1 - NN_x_new))
print("The Cross Entropy is %.3f, Average CE per event is %.3f"%(H, H/len(y)))

##Using pytorch to check the answer

NN_x_new = torch.FloatTensor([0.26, 0.16, 0.72 , 0.51, 0.74, 0.28, 0.9 , 0.75, 0.22, 0.68, 0.85,0.06, 0.66, 0.84, 0.67, 0.74, 0.26, 0.81, 0.57, 0.6 , 0.23, 0.26, 0.99, 0.61, 0.35, 0.04, 0.76, 0.16, 0.26, 0.25, 0.56, 0.76])
y = torch.FloatTensor([0., 0., 1., 1., 1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0.])
BinaryCrossEntropy = BCELoss(reduction="mean")    #reduction="mean" means that we are calculating the BCE loss of each element instead of the whole vector
loss_new = BinaryCrossEntropy(NN_x_new,y)
print("The PyTorch calculation of binary cross entropy is %.3f, does it agree with Average CE per event in your previous calculation?"%(loss_new.item()))

## Turn y and the NNs into 1-dimensional numpy arrays for ploting
y_plot = y.numpy().flatten()
NN_x_new_plot = NN_x_new.numpy().flatten()
NN_x_plot = NN_x.numpy().flatten()

## These are boolean arrays
signal_index = y_plot == 1     #Select signal event based on their label in y vector
background_index = y_plot == 0 #Select background event based on their label in y vector

print('signal', signal_index)
print('background', background_index)

plt.figure(figsize=(20,5))

plt.subplot(121)

plt.ylim(0,10)
binrange = np.arange(0,1,0.1)

plt.hist(NN_x_plot[signal_index],histtype="step",label="Signal",linewidth=2, bins=binrange)
plt.hist(NN_x_plot[background_index],alpha=0.3,label="Background", bins=binrange)

plt.title("Original Network Output: Cross Entrophy = %.3f"%(loss.item()))
plt.legend()
plt.xlabel("Network Output")
plt.ylabel("Counts")
plt.yscale

plt.subplot(122)
plt.ylim(0,10)
plt.hist(NN_x_new_plot[signal_index],histtype="step",label="Signal",linewidth=2, bins=binrange)
plt.hist(NN_x_new_plot[background_index],alpha=0.3,label="Background", bins=binrange)
plt.title("Updated Network Output: Cross Entrophy = %.3f"%(loss_new.item()))

plt.legend()
plt.xlabel("Network Output")
plt.ylabel("Counts")

plt.show()