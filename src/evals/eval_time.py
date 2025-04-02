import torch
import h5py
import pandas as pd
import os
import argparse
import time
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def eval_time(args, device, model):
    input_tensor = torch.randn(1, 3, args.image_size, args.image_size).to(device) 

    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)

        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100): 
            _ = model(input_tensor)
        torch.cuda.synchronize() 
        end_time = time.time()

    avg_inference_time = (end_time - start_time) / 100
    print(f"Average inference time per image: {avg_inference_time:.6f} seconds")