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
from customized_datasets.customized_datasets import PCamDataset, SkinDataset
from models.ddsnet import DDSNet

    
def train_one_epoch(dataloader, model, loss_fn, optimizer, scheduler, device, model_name):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        if model_name!='Inception':
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()

        elif model_name=='Inception':
            outputs = model(X)
            loss1 = F.cross_entropy(outputs.logits, y) 
            loss2 = F.cross_entropy(outputs.aux_logits, y) 
            loss = loss1 + 0.4 * loss2
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]", flush=True)
    scheduler.step()

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, accuracy = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy /= size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return accuracy, test_loss


def prepare(args):
    
    image_transforms_H5 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])
    
    image_transforms_CC = transforms.Compose([
    transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])

    image_transforms_RS = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ])

    device=args.device
    if args.dataset=='GC':
        train_dataset = datasets.ImageFolder(root=os.join(args.dataset_path, "custom_dataset/train"), 
                                             transform=image_transforms_RS)
        val_dataset = datasets.ImageFolder(root=os.join(args.dataset_path, "custom_dataset/val"), 
                                           transform=image_transforms_RS)
        labels = val_dataset.classes
        
    if args.dataset=='PCam':
        train_dataset = PCamDataset(os.join(args.dataset_path, "pcam/training_split.h5"), 
                                    os.join(args.dataset_path, "Labels/Labels/camelyonpatch_level_2_split_train_y.h5"), 
                                    transform=image_transforms_H5)
        val_dataset = PCamDataset(os.join(args.dataset_path, "pcam/validation_split.h5"), 
                                    os.join(args.dataset_path, "Labels/Labels/camelyonpatch_level_2_split_valid_y.h5"), 
                                    transform=image_transforms_H5)
        labels = ['normal', 'tumor']
        
    if args.dataset=='Skin':
        train_dataset = SkinDataset(csv_file=os.join(args.dataset_path, "HAM10000_metadata.csv"), 
                                    img_dir=os.join(args.dataset_path, "Skin_Cancer"),
                                    split='train',
                                    transform=image_transforms_CC)
        val_dataset = SkinDataset(csv_file=os.join(args.dataset_path, "HAM10000_metadata.csv"), 
                                  img_dir=os.join(args.dataset_path, "Skin_Cancer"), 
                                  split='val',
                                  transform=image_transforms_CC)
        labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    if args.dataset=='Sipak':
        dataset = datasets.ImageFolder(root=os.join(args.dataset_path, "SipakMed"), transform=image_transforms_RS)
        targets = np.array(dataset.targets)
        train_idx, val_idx = train_test_split(np.arange(len(targets)), test_size=0.2, stratify=targets, random_state=42)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        labels = dataset.classes
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    if args.model=='DDS':
        model = DDSNet(args.input_channel, args.output_channel).to(device)
    elif args.model=='ResNet':
        model = resnet18(num_classes=args.output_channel).to(device)
    if args.model=='DenseNet':
        model = densenet121(num_classes=args.output_channel).to(device)
    if args.model=='VGG':
        model = vgg11(num_classes=args.output_channel).to(device)
    if args.model=='Squeeze':
        model = squeezenet1_0(num_classes=args.output_channel).to(device)
    if args.model=='Inception':
        model = inception_v3(num_classes=args.output_channel).to(device)

    return train_loader, test_loader, labels, device, model

def train(args):
    train_loader, test_loader, device, model = prepare(args)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    epochs = args.epoch
    accuracy = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------", flush=True)
        train_one_epoch(train_loader, model, loss_fn, optimizer, scheduler, device, args.model)
        accuracy_temp, loss = test(test_loader, model, loss_fn, device)
        if(accuracy_temp > accuracy):
            accuracy = accuracy_temp
            ckpt = args.model + "_" + args.dataset + "_" + str(args.learning_rate)+".pth"
            torch.save(model.state_dict(), os.join(args.result_path, ckpt))
            print("the best acc of myNet is at epoch " + str(t+1) + " now.")


