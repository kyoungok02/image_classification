import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR

# dataset and transformation
from ops import dataset, models
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
import time
import copy

# model
# from networks import vgg,inception

# for tensorboard
from tensorboardX import SummaryWriter

# for parser
import argparse

np.random.seed(0)
torch.manual_seed(0)

# train device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# for apple m1
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = ('cpu')

def main(parser):
    model_type = parser.model_type
    dataset_name = parser.dataset
    learning_rate = parser.lr
    num_epochs = parser.epochs
    # store name ex) VGG_STL10_e10_cpu_23d10:10 (model type, dataset, epochs, device, todays'day hour minute)
    now = time.localtime()
    store_name = f'{model_type}_{dataset_name}_e{num_epochs}_{device}_{now.tm_mday:>02d}d{now.tm_hour:>02d}:{now.tm_min:>02d}'
    print(f"The summary of classification\n=========================================")
    print(f"The model is : {model_type} ")
    print(f"The dataset is : {dataset_name} ")
    print(f"The training device is : {device} ")
    print(f"The directory name is : {store_name} ")
    print(f"=========================================")
    # make directories
    data_dir = parser.data_dir
    log_dir = f'{parser.log_dir}/{store_name}'
    tf_write = SummaryWriter(log_dir = log_dir)
    createFolder(parser.checkpoint)
    weight_dir = f'{parser.checkpoint}/{store_name}'
    createFolder(data_dir)
    
    running_lr = parser.running_lr
    aux_layer = parser.aux_layer
    sum_on = False
    # best_loss = float('inf')
    best_metric = 0
    
    # load dataset
    train_ds, val_ds = dataset.load_dataset(data_dir, dataset_name)
    num_classes = len(train_ds.classes)
    in_channels = train_ds[0][0].size(0)

    # create dataloader
    batch_size = 4
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    # creat VGGnet object
    model = models.load_model(model_type,in_channels,num_classes,True,device,sum_on)
    
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    start_lr = opt.param_groups[0]['lr']
    print('start lr={}'.format(start_lr))
    # define learning rate scheduler
    # lr_scheduler = CosineAnnealingLR(opt, T_max=2, eta_min=1e-5)
    if running_lr:
        lr_scheduler = StepLR(opt, step_size=30, gamma=0.1)

    # # definc the training parameters
    params_train = {
        'optimizer':opt,
        'loss_func':criterion,
        'train_dl':train_dl,
        'aux_layer':aux_layer,
    }
    params_val = {
        'loss_func':criterion,
        'val_dl':val_dl,
        'sanity_check':False,
    }

    # train model
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/ {num_epochs} \n ---------------------------------------")
        print("lr: ", opt.param_groups[0]['lr'])
        start_time = time.time()
        train_loss, train_metric = train(model, params_train)
        print(f"train loss : {train_loss:>8f}, train accuracy : {(100*train_metric):>0.1f}%, training time : {(time.time()-start_time):>8f}s \n")
        if running_lr:
            lr_scheduler.step()
        if epoch % 5 == 0:
            print(f"===valiation step {epoch+1} ===\n")
            val_loss, val_metric = validation(model, params_val)
            # if val_loss < best_loss:
                # best_loss = val_loss
            if val_metric > best_metric:
                best_metric = val_metric
                # best_model_wts = copy.deepcopy(model.state_dict())
                # # store weights into a local file
                torch.save(model.state_dict(), weight_dir)
                print("Copied best model weights!")
            print(f"Validation Result: \n Accuracy: {(100*val_metric):>0.1f}%, Avg loss: {val_loss:>8f} \n")
        
        # save the tensorboard log
        tf_write.add_scalar('loss/train', train_loss, epoch)
        tf_write.add_scalar('loss/val', val_loss, epoch)
        tf_write.add_scalar('acc/train', train_metric, epoch)
        tf_write.add_scalar('acc/val', val_metric, epoch)
        tf_write.add_scalar('lr', opt.param_groups[0]['lr'], epoch)


def train(model,params):
   # extract model parameters
    loss_func=params["loss_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    aux_layer=params["aux_layer"]
    model.train()
    size = len(train_dl.dataset)
    batch_size = len(train_dl)
    running_loss, correct = 0, 0
    for itr, (X, y) in enumerate(train_dl):
        X = X.to(device)
        y = y.to(device)
        if aux_layer:
            pred, aux1, aux2 = model(X)
            out_loss = loss_func(pred,y)
            aux1_loss = loss_func(aux1,y)
            aux2_loss = loss_func(aux2,y)
            loss = out_loss + 0.3 * (aux1_loss + aux2_loss)
        else:
            pred = model(X)
            loss = loss_func(pred,y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        running_loss += loss
           
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if itr % 100 == 1:
            loss, current = loss.item(), itr*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    total_loss = running_loss / float(batch_size)
    accuracy = correct / float(size)

    return total_loss, accuracy
    

def validation(model, params):

    model.eval()
    loss_func = params["loss_func"]
    val_dataset = params["val_dl"]

    size = len(val_dataset.dataset)
    batch_size = len(val_dataset)
    val_loss, correct = 0,0
    with torch.no_grad():
        for X, y in val_dataset:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            val_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    val_loss = val_loss / float(batch_size)
    correct = correct / float(size)
    
    return val_loss, correct

# create the directory that stores weights.pt
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",type=str, help="class model \
                        (VGG11, VGG13, VGG16, VGG19, \n\
                        InceptionV1, InceptionV2, InceptionV3, InceptionResNet, \n\
                        ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)", default="VGG11")
    parser.add_argument("--aux_layer", type=bool, help="Add the aux layer or not, when inceptionV1", default=False)
    parser.add_argument("--dataset",type=str, help="type of dataset", default="STL10")
    parser.add_argument("--lr", type=float, help="start learning rate", default=0.0001)
    parser.add_argument("--epochs", type=int,help="number of epoch",default=50)
    parser.add_argument("--running_lr",type=bool, help="learning rate variation or not",default=False)
    parser.add_argument("--data_dir", type=str, help="dataset download directory",default="./data")
    parser.add_argument("--checkpoint", type=str, help="checkpoint directory",default="./checkpoint")
    parser.add_argument("--log_dir", type=str, help="tensorflow log directory",default="./logs")
    args = parser.parse_args()
    main(args)