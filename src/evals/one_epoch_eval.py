import torch
import numpy as np

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