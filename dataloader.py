import os
import pdb
import torch
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

#=============================================================================
#                         Data loading 
#============================================================================= 
# Custom dataset prepration in Pytorch format
class SkinData(Dataset):
    def __init__(self, df, transform = None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))
        
        if self.transform:
            X = self.transform(X)
        
        return X, y

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label  

def data_load(args):
    #=============================================================================
    #                         Data preprocessing
    #=============================================================================  
    # Data preprocessing: Transformation
    if args.chan == 3:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif args.chan == 1:
    	mean = [0.5]
    	std = [0.5]

    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), 
                        transforms.RandomVerticalFlip(),
                        transforms.Pad(3),
                        transforms.RandomRotation(10),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = mean, std = std)
                        ])
    
    test_transforms = transforms.Compose([
                        transforms.Pad(3),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = mean, std = std)
                        ])   

    num_users = args.num_users
    if args.dataset == 'HAM':
        df = pd.read_csv(args.data_path)
        #print(df.head())
        
        lesion_type = {
            'nv': 'Melanocytic nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign keratosis-like lesions ',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
            }

        # merging both folders of HAM1000 dataset -- part1 and part2 -- into a single directory
        imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                for x in glob(os.path.join("HAM", '*', '*.jpg'))}

        #print("path---------------------------------------", imageid_path.get)
        df['path'] = df['image_id'].map(imageid_path.get)
        df['cell_type'] = df['dx'].map(lesion_type.get)
        df['target'] = pd.Categorical(df['cell_type']).codes
        print(df['cell_type'].value_counts())
        print(df['target'].value_counts())
    

        #=============================================================================
        # Train-test split          
        train, test = train_test_split(df, test_size = 0.2)
        train = train.reset_index()
        test = test.reset_index()

        # With augmentation
        dataset_train = SkinData(train, transform = train_transforms)
        dataset_test = SkinData(test, transform = test_transforms)

    elif args.dataset == 'MNIST':
        dataset_train = datasets.MNIST(root='./', train=True, transform=train_transforms, download=True)
        dataset_test = datasets.MNIST(root='./', train=False, transform=test_transforms, download=True)
    
    elif args.dataset == 'F-MNIST':
        dataset_train = datasets.FashionMNIST(root='./', train=True, transform=train_transforms, download=True)
        dataset_test = datasets.FashionMNIST(root='./', train=False, transform=test_transforms, download=True)
    elif args.dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(root='./', train=True, transform=train_transforms, download=True)
        dataset_test = datasets.CIFAR10(root='./', train=False, transform=test_transforms, download=True)
    #----------------------------------------------------------------
    dict_users = dataset_iid(dataset_train, num_users)
    dict_users_test = dataset_iid(dataset_test, num_users)

    return dataset_train, dataset_test, dict_users, dict_users_test

#=====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
def dataset_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace = False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users  