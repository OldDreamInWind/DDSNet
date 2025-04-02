import torch
import h5py
import pandas as pd
import os
import argparse
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg11, resnet18, densenet121, squeezenet1_0, inception_v3
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from evals.one_epoch_eval import one_epoch_eval


def eval_tsne(args, test_loader, labels, device, model):
    features = []

    def hook(module, input, output):
        features.append(output.detach().cpu().numpy())

    # register hook on Flatten layer
    flatten_layer = model.net[-1]  # Get parameter of Flatten layer 
    hook_handle = flatten_layer.register_forward_hook(hook)

    label_list, pred_list = one_epoch_eval(test_loader, model, device)
    features = np.vstack(features)
    y = np.concatenate(label_list, axis=0)
    y1 = np.concatenate(pred_list, axis=0)

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(features)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'gray']

    # Save figure based on groundtruth labels
    plt.figure(figsize=(10, 8))
    for i in range(args.output_channel): 
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], color=colors[i], label=labels[i])
        
    plt.legend(loc="best")
    plt.title('t-SNE visualization of DDSNet Features for '+args.dataset)
    plt.savefig(args.dataset+"_DDS_tsne_1.png")
    # plt.show()
    plt.close()
    
    # Save figure based on predicted labels
    plt.figure(figsize=(10, 8))
    for i in range(args.output_channel): 
        plt.scatter(X_tsne[y1 == i, 0], X_tsne[y1 == i, 1], color=colors[i], label=labels[i])
        
    plt.legend(loc="best")
    plt.title('t-SNE visualization of DDSNet Features for '+args.dataset)
    plt.savefig(args.dataset+"_DDS_tsne_1_pred.png")
    # plt.show()
    plt.close()
    
    hook_handle.remove()

