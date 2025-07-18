#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:38:28 2024

@author: javizara
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import imageio
import matplotlib.pyplot as plt
import seaborn as sn
import os

def plot_loss_(plot_loss, plot_acc):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    data_loss = pd.DataFrame(plot_loss)
    sn.lineplot(x='epoch', y='loss', data=data_loss, hue='type')
    plt.title("Loss function")
    plt.xlabel("Epoch")      
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    data_acc = pd.DataFrame(plot_acc)
    sn.lineplot(x='epoch', y='accuracy', data=data_acc, hue='type')
    plt.title("Accuracy function")
    plt.xlabel("Epoch")      
    plt.ylabel("accuracy")
    
    # Trouver un nom de fichier qui n'existe pas encore...
    n = 0
    while True:
        filename = f'./Results/loss{n}.png'
        if not os.path.exists(filename):
            plt.savefig(filename) # Pour ensuite attribuer son nom à la figure
            break
        n += 1
    plt.tight_layout()
    plt.show()
    plt.subplot(1, 1, 1)

def confusion_matrix(classes, testloader, net):
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    label_ = np.zeros(shape = (len(classes), len(classes)))
    with torch.no_grad():
        for data in tqdm(testloader, desc = "Calculating accuracy for each class", ncols = 100):
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            
            for label, prediction in zip(labels, predictions):

                y = int(label.numpy())
                try : 
                    x = int(prediction.numpy())
                except:
                    x = int(prediction.cpu().numpy())
                label_[y][x] += 1
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    print("")        
    prt = {"Class": [] , "Accuracy" : []} 
    for classname, correct_count in correct_pred.items():
        try:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname} is {accuracy:.1f} %')
            prt["Class"].append(classname)
            prt["Accuracy"].append(round(accuracy, 3))
        except:
            accuracy = 0
            print(f'Accuracy for class: {classname} is {accuracy:.1f} %')
            prt["Class"].append(classname)
            prt["Accuracy"].append(round(accuracy, 3))

    label_ = pd.DataFrame(label_, index = [i for i in classes],
                      columns = [i for i in classes])
    label_ = label_.iloc[:, :]
    plt.figure()
    plt.subplot(1, 1, 1)
    sn.heatmap(label_, annot=True, fmt=".0f", cmap="crest")
    plt.xlabel ("Predicted")      
    plt.ylabel ("Labels") 
    
    # Save file 
    prt = pd.DataFrame(prt)
    n = 0 
    while True:
        if not os.path.exists(f"./Results/Precision{n}.csv"):
            prt.to_csv(f"./Results/Precision{n}.csv")
            plt.savefig(f"./Results/Matrix{n}")
            break
        n += 1
    plt.show() 
            

