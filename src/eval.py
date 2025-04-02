import torch
import numpy as np
from train import prepare
from evals.eval_tsne import eval_tsne
from evals.eval_confusion_matrix import eval_confusion_matrix
from evals.eval_gcam import eval_gcam
from evals.eval_time import eval_time

def eval(args):
    _, test_loader, labels, device, model = prepare(args)
    model.load_state_dict(torch.load(args.pretrained))

    model.eval()
    if args.eval == 'tsne': 
        eval_tsne(args, test_loader, labels, device, model)
    if args.eval == 'confusion_matrix':
        eval_confusion_matrix(args, test_loader, labels, device, model)
    if args.eval == 'gcam':
        eval_gcam(args, model)
    if args.eval == 'time':
        eval_time(args, device, model)
    