import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
from torch.optim import lr_scheduler 

import numpy as np 
import time 
import os 
import copy
import matplotlib.pyplot as plt 

import torchvision 
from torchvision import datasets, models, transforms

transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

trainset = torchvision.datasets.CIFAR10(root='/home/slr/Desktop/vr/Assignments/VR_Group_Project_2/Classification/masked', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

classes = ('kurta', 't-shirt', 'shirt', 'saree')


alexnet = torchvision.models.alexnet(pretrained=True)
classifier_list = list(alexnet.classifier)
num_features = classifier_list[-1].in_features
classifier_list[-1] = nn.Linear(num_features, 4)
alexnet.classifier = nn.Sequential(classifier_list[0],
                                   classifier_list[1],
                                   classifier_list[2],
                                   classifier_list[3],
                                   classifier_list[4],
                                   classifier_list[5],
                                   classifier_list[6],
                                   nn.Softmax(1))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.parameters(), lr=1e-5, momentum = 0.9)


for epoch in range(25):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = alexnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f'%
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        
            

print('Finished Training')

PATH = './alex_net.pth'
torch.save(net, PATH)

