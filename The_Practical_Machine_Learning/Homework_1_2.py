import numpy as np 
import torch
from torch.nn import BCELoss
import matplotlib
import matplotlib.pyplot as plt

##Initialize the variables
a = torch.tensor(1., requires_grad=True)
w1 = torch.tensor(2., requires_grad=True)
w2 = torch.tensor(2., requires_grad=True)
w3 = torch.tensor(2., requires_grad=True)
w4 = torch.tensor(2., requires_grad=True)

print("requires_grad: ", a.requires_grad, w1.requires_grad, w2.requires_grad, w3.requires_grad,w4.requires_grad)
print("grad: ", a.grad, w1.grad, w2.grad, w3.grad,w4.grad)

b = a * w1
c = a * w2
d = w3 * b + w4 * c
L = 10 - d


print("[a, w1, w2, w3, w4] is_leaf: ", a.is_leaf, w1.is_leaf, w2.is_leaf, w3.is_leaf, w4.is_leaf)
print("[b, c, d, L] is_leaf: ", b.is_leaf, c.is_leaf, d.is_leaf, L.is_leaf)

grad = torch.autograd.grad(outputs=L, inputs=a, retain_graph=True)
print(grad)

L.backward(retain_graph=True)
print(a.grad)

for i in range(10):
    L.backward(retain_graph=True)
    print(r"The gradient of %dth call is: %.2f"%(i, a.grad))

for i in range(10):
    a.grad.zero_()
    L.backward(retain_graph=True)
    print(r"The gradient of %dth call is: %.2f"%(i, a.grad))


#Define the model
b = a * w1
c = a * w2
d = w3 * b + w4 * c
L = 10 - d
grad = torch.autograd.grad(outputs=L, inputs=a)
print("gradient before detachment: %.2f"%(grad))

#Define the model, but detach non-leaf node c from the graph
b = a * w1
c = a * w2
d = w3 * b + w4 * c.detach()
L = 10 - d
grad = torch.autograd.grad(outputs=L, inputs=a)
print("gradient after c is detached: %.2f"%(grad))