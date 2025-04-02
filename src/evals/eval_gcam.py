import torch
import h5py
import pandas as pd
import os
import argparse
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None

        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.features = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        output = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            print(class_idx)

        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()

        # Calculate Grad-CAM
        gradients = self.gradients.cpu().data.numpy()
        features = self.features.cpu().data.numpy()

        # Average gradient and feature maps
        weights = np.mean(gradients, axis=(2, 3), keepdims=True)
        cam = np.sum(weights * features, axis=1)

        # Standarize
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam
    
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def eval_gcam(args, model):
    # Initial Grad-CAM
    target_layer = model.net[-6] 
    grad_cam = GradCAM(model, target_layer)

    input_image = preprocess_image(args.test_image)
    cam = grad_cam(input_image)

    # Visualize Grad-CAM
    cam = cam.squeeze()
    cam = cv2.resize(cam, (224, 224))
    
    image = cv2.imread(args.test_image)
    image = cv2.resize(image, (224, 224))
    output_path = os.path.join(args.result_path, args.dataset+"_resize.png")
    cv2.imwrite(output_path, image)
    
    image = np.float32(image) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    output_path = os.path.join(args.result_path, args.dataset+"_heat.png")
    cv2.imwrite(output_path, heatmap)
    
    # Overlap gcam on image
    result = show_cam_on_image(image, cam)
    output_path = os.path.join(args.result_path, args.dataset+"_gcam.png")
    cv2.imwrite(output_path, result)