#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 21:06:10 2024

@author: jhodi
"""
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from captum.attr import IntegratedGradients
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_model(trained_model):
    model = torch.load(trained_model)
    model.eval()
    return model

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (32, 32))  # Ajustez la taille selon votre modèle
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir BGR à RGB
    image = image / 255.0  # Normalisation
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Changer l'ordre des dimensions
    image = image.unsqueeze(0)  # Ajouter une dimension pour le batch
    return image

# Fonction principale pour Integrated Gradients
def integrated_gradients_explanation(image_path, output_path, 
                                     model=None, target_class=0):
    image = load_and_preprocess_image(image_path)
    
    # Créer un objet Integrated Gradients
    ig = IntegratedGradients(model)

    # Calculer les attributions
    attributions = ig.attribute(image, target=target_class)

    # Convertir les attributions en scores d'importance
    importance_scores = attributions.squeeze().detach().numpy()

    # Convertir en image 2D (par exemple, en prenant la moyenne des canaux)
    importance_scores = np.mean(importance_scores, axis=0)  # Moyenne sur les canaux

    # Normaliser les scores d'importance
    importance_scores = (importance_scores - np.min(importance_scores)) / (np.max(importance_scores) - np.min(importance_scores))

    # Charger l'image originale pour la superposition
    original_image = cv2.imread(image_path)
    original_image = cv2.resize(original_image, (32, 32))  # Ajustez la taille selon votre modèle
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convertir BGR à RGB

    # Appliquer le masque d'attribution sur l'image originale
    heatmap = np.uint8(255 * importance_scores)  # Convertir en valeurs d'intensité
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Appliquer une carte de couleur
    superimposed_img = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)  # Superposer

    # Visualiser les scores d'importance
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.imshow(superimposed_img)
    plt.title('Image avec masque d\'attribution')
    plt.colorbar(label='Importance des attributs')

    n = 0
    while True:
        if not os.path.exists(f"./{output_path}/Integrated_Gradients_Mask{n}.png"):
            plt.savefig(f"./{output_path}/Integrated_Gradients_Mask{n}.png")
            break
        n += 1
    plt.tight_layout()
    plt.show()

    return importance_scores



