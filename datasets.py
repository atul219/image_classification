from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader



def custom_transform(model_name, dataset_name):
    # checking if dataset name is CIFAR10
    if dataset_name == "CIFAR10":
        
        # checking if model name is AlexNet
        if model_name == 'AlexNet':
            transform_train = transforms.Compose([
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

            ])

            transform_test = transforms.Compose([
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

            transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    else:
        if model_name == 'AlexNet':
            transform_train = transforms.Compose([
                                transforms.Resize((227, 227)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
            ])

            transform_test = transforms.Compose([
                            transforms.Resize((227, 227)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
            ])

        else:
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                ])

            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
               ])

        
    return transform_train, transform_test




def custom_dataloader(dataset_name, model_name, batch_size):

    transform_train, transform_test = custom_transform(dataset_name= dataset_name, model_name = model_name)

    if dataset_name == 'CIFAR10':
        # downloading dataset CIFAR10
        train_set = torchvision.datasets.CIFAR10(root = '../data', train = True, download = True, transform= transform_train)
        val_set = torchvision.datasets.CIFAR10(root = '../data', train = False, download = True, transform  = transform_test)

    
    if dataset_name == 'MNIST':
        #downloading dataset MNIST
        train_set = torchvision.datasets.MNIST(root = '../data', train = True, download = True, transform = transform_train)
        val_set = torchvision.datasets.MNIST(root = '../data', train = False, download = True, transform = transform_test)

    

    train_dl = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers= 0)
    val_dl = DataLoader(val_set, batch_size = batch_size, shuffle = False, num_workers = 0)
    
    return (train_dl, val_dl)


