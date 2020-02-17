#chapter3. Neural networks

#신경망은 torch.nn 패키지에서 생성가능하다
#nn model을 정의하고 미분하는데 autograd를 사용한다.
#nn.module layer는 output을 반환하는 forward(input) method를 포함하고 있다.

#정의할 신경망의 weight = weight - learning rate*gradient

#신경망 정의
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):ㅁ
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

#forward 함수만 정의하고 나면 (변화도를 계산하는) backward gkatnsms autograd를 사용하여 자동으로 정의 된다.
#torch.nn은  mini-batch만 지원한다. 
#nnConv2D는 nsamples * nChannels * Height * Width 의 4차원 tensor를 입력으로 한다.


#손실 함수
net = Net()

input = torch.randn(1,1,32,32)
out = net(input)
target = torch.arange(1, 11)
target = target.view(1, -1)
target = target.type(torch.FloatTensor)
criterion = nn.MSELoss()

loss = criterion(out, target)
print(loss)

	# 전체 신경망의 경로
	# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
    # -> view -> linear -> relu -> linear -> relu -> linear
    # -> MSELoss
    # -> loss

#따라서 loss.backward()를 실행하게 된다면, 전체 그래프는 loss에 대해 미분하게 되며,
#그래프 내의 requires_grad = True인 모든 tensor는 gradient rk snwjrehls .grad Tensor를 가지게 된다.

#역전파
#오차를 역전파하기 위해 하는 것은  loss.backward()가 전부이다.
net = Net()

input = torch.randn(1,1,32,32)
out = net(input)
target = torch.arange(1, 11)
target = target.view(1, -1)
target = target.type(torch.FloatTensor)
criterion = nn.MSELoss()

loss = criterion(out, target)

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

#가중치 갱신
#SGD: 가중치 = 가중치 - 학습율 * 변화도
#하지만 torch.optim 라이브러리를 사용하면 간단하게 구현 가능

net = Net()

input = torch.randn(1,1,32,32)
target = torch.arange(1, 11)
target = target.view(1, -1)
target = target.type(torch.FloatTensor)
criterion = nn.MSELoss()
output = net(input)


#optimizer 생성
optimizer = optim.SGD(net.parameters(), lr = 0.01)

#training loop
optimizer.zero_grad() #변화도 누적을 막고자 0으로 설정해줌
loss = criterion(output, target)
loss.backward()
optimizer.step()
