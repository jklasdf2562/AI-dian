import json
import numpy as np
import sys
import torch
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_
train=datasets.MNIST(root='./data',train=True,download=True,transform=transforms.ToTensor())
test=datasets.MNIST(root='./data',train=False,download=True,transform=transforms.ToTensor())
train_loader=DataLoader(train,batch_size=64,shuffle=True)
test_loader=DataLoader(test,batch_size=64,shuffle=False)
class CNN(nn.Module):
    def __init__(self,n_channels):
        super().__init__()
        #卷积层1
        self.conv1=nn.Conv2d(n_channels,32,(3,3))
        kaiming_uniform_(self.conv1.weight,nonlinearity='relu')
        self.act1=nn.ReLU()
        #池化层1
        self.pool1=nn.MaxPool2d((2,2),stride=(2,2))
        #卷积层2
        self.conv2=nn.Conv2d(32,64,(3,3))
        kaiming_uniform_(self.conv2.weight,nonlinearity='relu')
        self.act2=nn.ReLU()
        #池化层2
        self.pool2=nn.MaxPool2d((2,2),stride=(2,2))
        #全连接层1
        self.fc1=nn.Linear(64*5*5,128)
        kaiming_uniform_(self.fc1.weight,nonlinearity='relu')
        self.act3=nn.ReLU()
        #全连接层2
        self.fc2=nn.Linear(128,10)
        xavier_uniform_(self.fc2.weight)
        self.act4=nn.Softmax(dim=1)
    def forward(self,x):
        x=self.conv1(x)
        x=self.act1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.act2(x)
        x=self.pool2(x)
        x=x.view(-1,64*5*5)
        x=self.fc1(x)
        x=self.act3(x)
        x=self.fc2(x)
        x=self.act4(x)
        return x
model=CNN(n_channels=1)
def train_model(train_loader,model):
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    for epoch in range(10):
        for i,(inputs,targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,targets)
            loss.backward()
            optimizer.step()
def evaluate_model(test_loader,model):
    pre,actual=[],[]
    for i,(inputs,targets) in enumerate(test_loader):
        outputs=model(inputs)
        outputs=outputs.detach().numpy()
        actuals=targets.numpy()
        preds=np.argmax(outputs,axis=1)
        actuals=actuals.reshape(len(actuals),1)
        preds=preds.reshape(len(preds),1)
        pre.extend(preds)
        actual.extend(actuals)
    pre,actual=np.vstack(pre),np.vstack(actual)
    accuracy=accuracy_score(actual,pre)
    return accuracy
train_model(train_loader,model)
accuracy=evaluate_model(test_loader,model)
print(f'Test Accuracy: {accuracy*100:.2f}%')
  
  
