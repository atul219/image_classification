import os
import sys
import time
import math
import torch
from config import VGG_types
from models.vgg import Vgg
from models.alexnet import AlexNet
from models.googlenet import GoogleNet
from models.efficientnet import EfficientNet
from models.resnet import ResNet50, ResNet101, ResNet152
import numpy as np
import itertools

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else 'cpu'

def get_model(model_name, dataset_name):
    if dataset_name == "MNIST":
        in_channels = 1
    else:
        in_channels = 3

    if model_name == "VGG16":
        model = Vgg(in_channels = in_channels, n_cls = 10, architecture = VGG_types[model_name])

    elif model_name == 'AlexNet':
        model = AlexNet(in_channels = in_channels, n_cls = 10)
    
    elif model_name == 'GoogleNet':
        model = GoogleNet(in_channels = in_channels ,n_cls= 10, aux_out= False)

    elif model_name == "ResNet50":
        
        model = ResNet50(img_channels = in_channels, n_cls = 10)
    
    else:
        print("[Error] Please Write Correct Model Name..")
    


    
    model.to(device)

    return model



    

def accuracy(pred_correct, total):
    acc = 100.*pred_correct/total

    return acc


def plot_loss_acc(epoch, history, model_name, dataset_name):
    
    if not os.path.isdir('results'):
        os.mkdir('results')

    # plot loss progress
    plt.title("Train-Val Loss")
    plt.plot(range(1, epoch + 1), history["train_loss"],label="train")
    plt.plot(range(1, epoch + 1),history["val_loss"],label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.grid()


    fig_name = '{}_loss_{}.png'.format(model_name, dataset_name)
    full_fig_name = os.path.join('./results/' + fig_name)
    plt.savefig(full_fig_name)
    print("[INFO] Saving Loss...")

    plt.show()


    # plot accuracy progress
    plt.title("Train-Val Accuracy")
    plt.plot(range(1, epoch + 1),history["train_acc"],label="train")
    plt.plot(range(1, epoch + 1),history["val_acc"],label="val")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    plt.grid()


    fig_name = '{}_acc_{}.png'.format(model_name, dataset_name)
    full_fig_name = os.path.join('./results/' + fig_name)
    plt.savefig(full_fig_name)
    print("[INFO] Saving Accuracy...")

    plt.show()



def plot_confusion_matrix(model_name, dataset_name, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fig_name = 'Connfusion Matrix of {}_{}.png'.format(model_name, dataset_name)
    full_fig_name = os.path.join('./results/' + fig_name)
    plt.savefig(full_fig_name)
    print("[INFO] Saving Confusion Matrix...")