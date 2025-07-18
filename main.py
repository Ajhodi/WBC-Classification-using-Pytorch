#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:09:27 2024

@author: javizara
"""
import os
os.chdir("/home/jhodi/bit/Python/WBC Classification using Pytorch/WBC-Classification-using-Pytorch/")

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import os
import argparse
import seaborn as sn
from tqdm import tqdm

# Import custom functions
from src.Stat_visu import *
from src.rise_analysis import *
from models.models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"############## {device} ##############\n")
best_acc = 0  # best test accuracy
_epoch = 25

new_size = 32
batch = 64

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(new_size), 
    # transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(new_size), 
    # transforms.RandomCrop(32, padding=4),
    # transforms.ColorJitter(brightness=0.5),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

args
trainset = torchvision.datasets.ImageFolder(root="../data/images/TRAIN", 
                                            transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root="../data/images/TEST", 
                                           transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch, shuffle=False, num_workers=2)



# Models
net = ResNet18()
# net = ResNet34()
# net = ResNet50()
# net = ResNet101()
# net = ResNet152()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print(f"\nEpoch : {epoch + 1}")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    bar = tqdm(enumerate(trainloader),
               total = len(trainloader),
               ncols = 100)
    for batch_idx, (inputs, targets) in bar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        plot_loss['type'].append("Train")
        plot_loss['epoch'].append(epoch+1)
        plot_loss['loss'].append(train_loss/(batch_idx+1))
        
        plot_acc['type'].append("Train")
        plot_acc['epoch'].append(epoch+1)
        plot_acc['accuracy'].append(100.*correct/total)
        
        bar.set_description(f"Trainning ==> Loss: {round(train_loss/(batch_idx+1), 2)} | Acc: {round(100.*correct/total, 2)}%")
        
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        bar  = tqdm(enumerate(testloader),
                    total = len(testloader),
                    ncols = 100)
        for batch_idx, (inputs, targets) in bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            plot_loss['type'].append("Validation")
            plot_loss['epoch'].append(epoch+1)
            plot_loss['loss'].append(test_loss/(batch_idx+1))
            
            plot_acc['type'].append("Validation")
            plot_acc['epoch'].append(epoch+1)
            plot_acc['accuracy'].append(100.*correct/total)
            
            bar.set_description(f"Testing ==> Loss: {round(test_loss/(batch_idx+1), 2)} | Acc: {round(100.*correct/total, 2)}%")
            
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

plot_loss = {'epoch' : [],
             'loss' : [], 
             'type' : []}

plot_acc = {'epoch' : [],
             'accuracy' : [], 
             'type' : []}

for epoch in (range(_epoch)):
    train(epoch)
    test(epoch)
    scheduler.step()

    
# save metrics 
image_paths = ["../data/images/TRAIN/EOSINOPHIL/_0_651.jpeg",
              "../data/images/TRAIN/LYMPHOCYTE/_0_204.jpeg",
              "../data/images/TRAIN/MONOCYTE/_0_180.jpeg",
              "../data/images/TRAIN/NEUTROPHIL/_0_292.jpeg"
              ]
n = 1
output_path = f"./Results/run{n}/"
while True:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        plot_loss_(plot_loss, plot_acc, output_path = output_path)
        confusion_matrix(classes = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"], 
                         output_path = output_path, testloader = testloader, net = net)
        for ID, image in enumerate(image_paths):
            importance_scores = integrated_gradients_explanation(image_path = image, output_path = output_path, model = net, target_class = ID)
        break
    n +=1
