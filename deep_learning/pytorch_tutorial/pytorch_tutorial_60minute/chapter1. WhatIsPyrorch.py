import torch
import numpy as np

#tensor_operations
'''
x = torch.tensor([5.3,3])
x = x.new_ones(5,3,dtype=torch.double)

x = torch.rand_like(x, dtype=torch.float)

y= torch.rand(5,3)

print(x+y)
print(torch.add(x,y))

result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)

print(result[:,1])

x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8)
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())
'''

#Numpy bridge
'''
a = torch.ones(5)
print(a)

b = a.numpy() #translation to numpy from torch tensor
print(b)

a.add_(1)
print(a)
print(b)

a = np.ones(5)
b = torch.from_numpy(a) #translation to tensor from numpy
np.add(a, 1, out=a)
print(a)
print(b)
'''

#CUDA Tensors
'''
x = torch.rand(5,3)
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x+y
    print(z)
    print(z.to("cpu", torch.double))
'''

