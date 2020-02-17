PyTorch로 딥러닝하기: 60분만에 끝장내기 예제코드

#chapter 1.
#Tensors

#초기화 되지 않은 행렬 생성
from __future__ import print_function
import torch

x = torch.empty(5,3)
print (x)

#무작위 행렬 생성
from __future__ import print_function
import torch

x = torch.rand(5,3)
print (x)

#data type이 long이고 0으로 채워진 행렬 생성 
from __future__ import print_function
import torch

x = torch.zeros(5,3, dtype=torch.long)
print (x)

#데이터로 부터 바로 tensor 생성
from __future__ import print_function
import torch

x = torch.tensor([5.5,3])
print (x)

#존재하는 tensor로 부터 tensor 생성
from __future__ import print_function
import torch

x = torch.tensor([5,3])
print(x)

x = x.new_ones(5,3,dtype=torch.double)
print(x)

x = torch.rand_like(x, dtype=torch.float)
print(x)

#행렬의 크기
from __future__ import print_function
import torch

x = torch.tensor([5,3])
print(x)
print(x.size())

x = x.new_ones(5,3,dtype=torch.double)
print(x)
print(x.size())

x = torch.rand_like(x, dtype=torch.float)
print(x)
print(x.size())
#torch.size는 튜플과 같으며, 모든 튜플 연산에서 사용가능

#덧셈 연산
from __future__ import print_function
import torch

x = torch.tensor([5,3])


x = x.new_ones(5,3,dtype=torch.double)

x = torch.rand_like(x, dtype=torch.float)


y = torch.rand(5,3)
print(x)
print(y)
print (x+y)
print(torch.add(x,y)) #위에꺼랑 같음

result = torch.empty(5,3) #인자로 전달 연산
torch.add(x, y,out=result )
print(result)

y.add_(x) #y에 x를 더한값, swap은 _라는 접미사를 갖는다.
print(y)

#tensor의 크기나 모양을 변경하고 싶을때 view사용
from __future__ import print_function
import torch

x = torch.rand(4,4)
y = x.view(16)
z = x.view(-1,8)
print(x)
print(y)
print(z)

#만약 tensor의 하나의 값만 존재한다면 item()을 사용해 숫자값을 얻을 수 있다.
from __future__ import print_function
import torch

x = torch.randn(1)
print(x)
print(x.item())

#Numpy 변환
#Torch Tensor를 numpy배열로 변환하기
from __future__ import print_function
import torch

a = torch.ones(5)
b = a.numpy()
print(a)
print(b)

#numpy 배열을 tensor로 변환
from __future__ import print_function
import torch
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1 , out=a)
print(a)
print(b)

