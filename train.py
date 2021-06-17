import sys
import torch
from torch._C import wait
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models.vgg import Vgg
from models.alexnet import AlexNet
from torchsummary import summary
from utils import accuracy, plot_loss_acc, get_model, plot_confusion_matrix
import time
from config import *
import os
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from datasets import custom_dataloader
from sklearn.metrics import confusion_matrix


from torch.utils.tensorboard import SummaryWriter
import torchvision

device = 'cuda' if torch.cuda.is_available() else 'cpu'


best_acc = 0
history = defaultdict(list)
# Initialize the prediction and label lists(tensors)
predicts_list = torch.zeros(0,dtype=torch.long, device='cuda')
label_list = torch.zeros(0,dtype=torch.long, device='cuda')

parser = argparse.ArgumentParser(description= 'Image Classification')
parser.add_argument('--epochs', type=int, help = 'epoch')
parser.add_argument('--model', type=str, help = "model name")
parser.add_argument('--dataset', type = str, help = 'dataset name')

args = parser.parse_args()

params = dict(lr= [0.001], batch_size = [32])

# laoding datalaoder for dataset
train_dl , val_dl = custom_dataloader(dataset_name = args.dataset, model_name = args.model, batch_size= 32)


if args.dataset == "MNIST":
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

else:
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')


print('==> Building model..')
print("[INFO] Model: {}".format(args.model))
print("[INFO] Dataset: {}".format(args.dataset))

# loading model
model = get_model(model_name = args.model, dataset_name = args.dataset)

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer 
optimizer = optim.Adam(model.parameters(), lr=0.001)
# learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience= 10, verbose=True)

# to write in tensorboard
writer = SummaryWriter("runs/CIFAR10")


# training method
def train(epoch):

    print("[INFO] Training {}....".format(args.model))
    print("\Epoch: [{}/{}]".format(epoch + 1, args.epochs))
    start_time = time.time()
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    # loop in batches of img and label 
    for i, (input, targets) in enumerate(tqdm(train_dl)):

        img = input.to(device)
        label = targets.to(device)


        optimizer.zero_grad()
        outputs = model(img)
        
        
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        
        train_loss += loss.item()
        _, predicts = outputs.max(1)

        total += label.size(0)
        correct += predicts.eq(label).sum().item()

    # writing sacaler to tensorboard
    writer.add_scalar("Train Loss", train_loss, epoch)
    writer.add_scalar("Train Correct", correct, epoch)
    writer.add_scalar("Train Accuracy", correct / len(train_dl), epoch)




    train_acc = accuracy(correct, total)
    train_loss = train_loss / len(train_dl)
    
    end_time = time.time()

    print('TrainLoss: %.3f | TrainAcc: %.3f%% (%d/%d) | Time Elapsed %.3f sec' % (train_loss, train_acc, correct, total, end_time-start_time))

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)

    torch.cuda.empty_cache()
    return (train_loss, train_acc)
    


# evaluating on validation dataset
def evaluate(epoch):
    print("[INFO] Evaluating {}...".format(args.model))
    print("\Epoch: [{}/{}]".format(epoch + 1, args.epochs))
    global best_acc, predicts_list, label_list
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (input, targets) in enumerate(tqdm(val_dl)):
            img = input.to(device)
            label = targets.to(device)
            outputs = model(img)
            loss = criterion(outputs, label)

            test_loss += loss.item()
            _, predicts = outputs.max(1)
            total += label.size(0)
            correct += predicts.eq(label).sum().item()

            # Append batch prediction results
            predicts_list = torch.cat([predicts_list, predicts.view(-1).cuda()])
            label_list = torch.cat([label_list, label.view(-1).cuda()])
        
        val_acc = accuracy(correct, total)
        val_loss = test_loss / len(val_dl)

        writer.add_scalar("Val Loss", val_loss, epoch)
        writer.add_scalar("Val Correct", correct, epoch)
        writer.add_scalar("Val Accuracy", correct / len(val_dl), epoch)
        

        print('TestLoss: %.3f | TestAcc: %.3f%% (%d/%d)' % (val_loss, val_acc, correct, total))

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        acc = val_acc
        if acc > best_acc:
            print("Saving...")

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            f_name = '{}.pth'.format(args.model)
            torch.save(model.state_dict(), os.path.join('./checkpoint/' + f_name))
            best_acc = acc

    
    return (val_loss, val_acc, best_acc)


# Training and Validation Loop
for epoch in range(args.epochs):
    train_loss, train_acc = train(epoch)
    eval_loss, eval_acc, best_acc = evaluate(epoch)
    writer.close()


# plot loss and accuracy 
plot_loss_acc(args.epochs, history, args.model, args.dataset)

# Confusion matrix
conf_mat=confusion_matrix(label_list.detach().cpu().numpy(), predicts_list.detach().cpu().numpy())


# plot confusion matrix
plt.figure(figsize = (10,10))
plot_confusion_matrix(args.model, args.dataset, conf_mat, classes)
