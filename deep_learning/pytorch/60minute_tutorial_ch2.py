#chaper2. AutoGrad (자동미분)
#define-by-run framework

#패키지의 중심에는 torch.Tensor가 있다.  if .requires_grad 속성을 True로 설정하면, 그 tensor에서 이뤄진 모든 연산을 추적하기 시작한다.
#tensor의 변화도는 .grad에 누적하게 된다.
#Tensor가 기록을 중단하게 하려면, .detach()를 호출하여 연산 기록으로부터 분리하여 이후 연산들이 기록되는 것을 방지할 수 있다.
#연산 기록을 추적하는 것을 막기 위해 withtorch.no.grad(): 로 감싸서 방지할 수 있다.

#Tensor와 Function은 상호 연결되어 있으며, 모든 연산 과정을 부호화하여 순환하지 않은 그래프를 생성한다.
#도함수를 계산하기 위해서는 Tensor의 .backword()를 호출하면 된다.



#tensor를 생성하고 requires_grad=True를 설정하여 연산을 기록한다.
import torch

x = torch.ones(2,2,requires_grad=True)
print(x)

#tensor의 연산을 수행한다.
import torch

x = torch.ones(2,2,requires_grad=True)
print(x)

y = x+2
print(y)

#y는 연산의 결과로 생성된 것이므로, grad_fn을 갖는다.
import torch

x = torch.ones(2,2,requires_grad=True)
print(x)

y = x+2
print(y.grad_fn)

#다른 연산 수행
import torch

x = torch.ones(2,2,requires_grad=True)


y = x+2

z= y*y*3
out = z.mean()
print(z, out)

#require grad 설정
import torch

a = torch.randn(2,2)
a = ((a*3)/ (a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)


