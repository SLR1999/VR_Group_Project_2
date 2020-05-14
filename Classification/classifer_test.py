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

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'masked'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}
class_names = image_datasets['val'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_model(model, criterion, optimizer, scheduler, epochs=25):
    since = time.time() 

    best_model_wt = copy.deepcopy(model.load_state_dict('weights/best_wts_train.pth))
    best_acc = 0.0 


    for phase in ['val']:
        model.eval()


        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device) 

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train': 
                    loss.backward()
                    optimizer.step()

            running_loss = loss.item() * inputs.size(0)
            running_corrects = torch.sum(preds == labels.data)


        loss = running_loss / dataset_sizes[phase] 
        acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, loss, acc))


    print()

    time_elapsed = time.time() - since
    print('Best val Acc: {:4f}'.format(best_acc))



alexnet = torchvision.models.alexnet(pretrained=True)
# for param in alexnet.parameters():
#     param.requires_grad = False 

# In AlexNet the model is sequentially defined

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

alexnet = alexnet.to(device)

criterion = nn.CrossEntropyLoss()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(alexnet))

optimizer = optim.SGD(alexnet.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

test_model(alexnet, criterion, optimizer, exp_lr_scheduler, epochs=25)