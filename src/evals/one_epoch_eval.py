import torch
import numpy as np
from train import prepare
from evals.eval_tsne import eval_tsne
from evals.eval_confusion_matrix import eval_confusion_matrix
from evals.eval_gcam import eval_gcam
from evals.eval_time import eval_time

def one_epoch_eval(dataloader, model, device):
    model.eval()
    label_list = []
    pred_list = []
    with torch.no_grad():
        for X, y in dataloader:
            label_list.append(y.numpy())
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_list.append(np.argmax(pred.detach().cpu().numpy(), axis=1))
            
    return label_list, pred_list