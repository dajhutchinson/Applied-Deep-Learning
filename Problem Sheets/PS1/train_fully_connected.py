from sklearn import datasets
import numpy as np
import seaborn as sns
import pandas as pd
import torch

# load data
iris=datasets.load_iris()
iris.keys()

# device to execute on (GPU)
device = torch.device('cuda')

"""LOGGING PERFORMANCE METRICS"""
from torch.utils.tensorboard import SummaryWriter

summary_writer = SummaryWriter('logs', flush_secs=5)

"""PREPARE FOR TRAINING"""
# normalise features to N(0,1)
preprocessed_features=(iris["data"]-iris["data"].mean(axis=0)) / iris["data"].std(axis=0)

# shuffle & split into training & testing sets
from sklearn.model_selection import train_test_split

labels=iris["target"]
train_features,test_features,train_labels,test_labels=train_test_split(preprocessed_features,labels,test_size=1/3)
# 1/3 for testing, 2/3 for training

# convert numpy arrays to pytorch tensors
features={
    "train":torch.tensor(train_features,dtype=torch.float32),
    "test":torch.tensor(test_features,dtype=torch.float32)
         }

# move features to gpu
features["train"]=features["train"].to(device)
features["test"] =features["test"].to(device)

labels={
    "train":torch.tensor(train_labels,dtype=torch.float32),
    "test":torch.tensor(test_labels,dtype=torch.float32)
       }
# move labels to gpu
labels["train"]=labels["train"].to(device)
labels["test"] =labels["test"].to(device)

"""DEFINING MODEL"""
# MLP class with 2 fully connected layers using ReLU on the output layer
from torch import nn
from torch.nn import functional as f
from typing import Callable

class MLP(nn.Module):

    def __init__(self,input_size:int,hidden_layer_size:int,output_size:int,
                 activation_fn:Callable[[torch.Tensor],torch.Tensor]=f.relu):
        super().__init__()
        self.l1=nn.Linear(input_size,hidden_layer_size)  # first hidden layer
        self.activation_fn=activation_fn
        self.l2=nn.Linear(hidden_layer_size,output_size) # second hidden layer

    # forward pass (make prediction)
    def forward(self,inputs:torch.Tensor) -> torch.Tensor:
        x=self.l1(inputs) # first calculation
        x=self.activation_fn(x)
        print(x)
        x=self.l2(x)
        return x

"""ACCURACY MODEL"""
# assess quality of predictions (% correct predictions)
def accuracy(probs:torch.FloatTensor,targets:torch.LongTensor) -> float:
    """
    PARAMETERS
        probs: predicted classification
        targets: target classes
    """
    classes=probs.argmax(axis=1)
    matches=(classes==targets).sum()
    return float(matches)/targets.shape[0]

"""TRAINING"""
from torch import optim

feature_count=4
hidden_layer_size=100
class_count=3
model=MLP(feature_count,hidden_layer_size,class_count) # model to optimize
model = model.to(device) # move model to GPU

optimiser=optim.SGD(model.parameters(),lr=.05) # SGD optimiser used update model parameters (lr=learning rate)
criterion=nn.CrossEntropyLoss() # loss function

# optimise using training set multiple times
for epoch in range(0,100):
    # forward pass (prediction)
    logits=model.forward(features['train']) # forward pass to make predictions
    loss  =criterion(logits,labels["train"].long()) # loss function value

    # output results
    acc=accuracy(logits,labels["train"].long())*100
    print("epoch: {} train accuracy:{:2.2f}, loss:{:5.5f}".format(epoch,acc,loss.item()))

    # log performance metrixs
    summary_writer.add_scalar('accuracy/train', acc, epoch)
    summary_writer.add_scalar('loss/train', loss.item(), epoch)

    # backwards pass (training)
    loss.backward()  # calculate gradients
    optimiser.step() # update model parameters (using gradients)
    optimiser.zero_grad() # zero so next pass doesnt add new gradients to these

"""TESTING"""
# test final model on test dataset
logits=model.forward(features["test"])
test_accuracy=accuracy(logits,labels["test"].long()) * 100
print("test accuracy: {:2.2f}".format(test_accuracy))

summary_writer.close()
