import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
import time
import os
import copy
import matplotlib.pyplot as plt
import pickle

import torchvision
from torchvision import datasets, models, transforms 

def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

PATH = 'weights/yolo_best_model_wts'

class AlexNet:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Initialize model to pretrained resnet18 weights
        self.model = torchvision.models.alexnet(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        classifier_list = list(self.model.classifier)
        num_ftrs = classifier_list[-1].in_features
        classifier_list[-1] = nn.Linear(num_ftrs, 4)

        self.model.classifier = nn.Sequential(classifier_list[0],
                                   classifier_list[1],
                                   classifier_list[2],
                                   classifier_list[3],
                                   classifier_list[4],
                                   classifier_list[5],
                                   classifier_list[6],
                                   nn.Softmax(1))
        print (self.model.parameters())
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        # Observe that only parameters of final layer are being optimized as
        # opposed to before.
        self.trainer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        self.exp_lr_scheduler = lr_scheduler.StepLR(self.trainer, step_size=7, gamma=0.1)
        self.image_datasets = {}
        self.dataloaders = {}
        self.dataset_sizes = {}
        self.class_names = []
        plt.ion()

    def init_data(self, data_dir = 'yolo'):
        
        self.data_dir = data_dir
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                self.data_transforms[x])
                        for x in ['train', 'val']}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=4,
                                                    shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes

        with open('classes.pkl', 'wb') as f:
            pickle.dump(self.class_names, f)

        # Get a batch of training data
        inputs, classes = next(iter(self.dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        
        imshow(out, title=[self.class_names[x] for x in classes])
        

    def train(self, num_epochs=25):
        
        self.init_data()

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.trainer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.trainer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    self.exp_lr_scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model.state_dict(), PATH)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.visualize_model()

    def test(self, image_path):
        import pickle
        from PIL import Image
        from torch.autograd import Variable 
        import cv2

        # since = time.time()

        PATH = '/home/slr/Desktop/vr/Assignments/VR_Group_Project_2/Classification/weights/yolo_best_model_wts.pth'
        self.model.load_state_dict(torch.load(PATH))
        self.model.eval()   # Set model to evaluate mode
        with open('/home/slr/Desktop/vr/Assignments/VR_Group_Project_2/Classification/classes.pkl', 'rb') as f:
            classes = pickle.load(f)

        image = cv2.imread(image_path)

        to_pil = transforms.ToPILImage()
        image = to_pil(image)
        image_tensor = self.data_transforms['val'](image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        image = Variable(image_tensor)
        image = image.to(self.device)
        output = self.model(image)
        _, preds = torch.max(output, 1)
        
        [index] = preds.data.cpu().numpy()
        

        # time_elapsed = time.time() - since
        # print('Training complete in {:.0f}m {:.0f}s'.format(
        #     time_elapsed // 60, time_elapsed % 60))
        
        return classes[index]


    def visualize_model(self,num_images=6):

        was_training = self.model.training
        self.model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}'.format(self.class_names[preds[j]]))
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        self.model.train(mode=was_training)
                        return
            self.model.train(mode=was_training)


        plt.ioff()
        plt.show()

if __name__ == "__main__":
    alexnet = AlexNet()

    label = alexnet.test('/home/slr/Desktop/vr/Assignments/VR_Group_Project_2/Classification/yolo/val/kurti/arora_18.jpg')
    print(label)