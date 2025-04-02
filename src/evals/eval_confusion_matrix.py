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


def eval_confusion_matrix(args, test_loader, labels, device, model):
    total_y=[]
    real_y=[]

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_y.append(pred.argmax(1))
            real_y.append(y)
    
    total_y=torch.cat(total_y)
    real_y=torch.cat(real_y)
    cm = confusion_matrix(real_y.cpu(), total_y.cpu())
    report = classification_report(real_y.cpu(), total_y.cpu(), digits=4)
    print(report)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title("Confusion_Matrix_"+args.dataset)
    fig_path = os.path.join(args.result_path, "Confusion_Matrix_"+args.dataset+".png")
    plt.savefig(fig_path)
    return cm