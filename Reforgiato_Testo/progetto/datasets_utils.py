# birds
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# chanchong
from dataset_functions import from_path_to_dataloader 

def get_train_valid_test_split(df, SPLITS=[0.8,0.1,0.1], SEED=42):
    assert sum(SPLITS) == 1, "SPLITS must sum to 1"
    assert len(SPLITS) == 3, "SPLITS must have 3 elements"
    assert [i for i in SPLITS if i < 0] == [], "SPLITS must be positive"
    assert [i for i in SPLITS if i > 1] == [], "SPLITS must be less than 1"

    ratio_train = SPLITS[0]
    ratio_val = SPLITS[1]
    ratio_test = SPLITS[2]

    train_df, test_df = train_test_split(df, test_size=ratio_test, random_state=SEED)

    ratio_remaining = 1 - ratio_test
    ratio_val_adjusted = ratio_val / ratio_remaining
    
    train_df, val_df = train_test_split(train_df, test_size=ratio_val_adjusted,  random_state=SEED)

    return train_df, val_df, test_df


# Example of use:
#   import datasets_utils
#   train_loader, val_loader, test_loader, classes = datasets_utils.get_CUB_loaders(BATCH_SIZE=16, SEED=42, SPLITS=[0.8,0.1,0.1])
#   N_CLASSES = len(classes)
def get_CUB_loaders(BATCH_SIZE = 16, SEED=42, SPLITS=[0.8,0.1,0.1]):
    PATH = './CUB_200_2011/images/'
    MY_TRANSFORMATIONS = A.Compose([
        A.Resize(256,256),
        ToTensorV2()
        ])
    
    class BirdDataset(Dataset):
        def __init__(self,DF, TRANSFORMATIONS):
            self._df = DF
            self._transformations = TRANSFORMATIONS

            self.images = self._df['path'].values
            self.classes = self._df['class'].values
            self.classes = np.array([classes.index(i) for i in self.classes])
            
        def __len__(self):
            return len(self._df)
        
        def __getitem__(self,idx):
            image = self.images[idx]
            image = plt.imread(image)
            #image = np.transpose(image,(2,0,1))
            #if is grayscale, convert to rgb
            if image.shape[0] == 1:
                image = np.repeat(image,3,0)
            #cast to float and normalize
            image = image.astype(np.float32)
            image = image/255.0 
            image =  self._transformations(image=image)['image']
            class_ = self.classes[idx]
            class_ = torch.tensor(class_,dtype=torch.long)
            image = image.type(torch.FloatTensor)
            if image.shape[0] == 1:
                image = torch.repeat_interleave(image,3,0)
            return image,class_
        
    #read all files from the folder CUB_200_2011 and assign the subfolder as a class
    #the subfolder name is the class name
    classes = os.listdir(PATH)
    classes.sort()
    print(classes)

    #read all files from the subfolders
    data = []
    for i in range(len(classes)):
        folder = os.path.join(PATH,classes[i])
        files = os.listdir(folder)
        for j in range(len(files)):
            data.append([classes[i],os.path.join(folder,files[j])])

    #convert the list to a dataframe
    df = pd.DataFrame(data,columns=['class','path'])
    df.head()

    #split the data into train and test and validation
    train_df, val_df, test_df = get_train_valid_test_split(df, SPLITS=SPLITS, SEED=SEED)
    
    #create the dataloaders
    train_loader = DataLoader(BirdDataset(train_df, MY_TRANSFORMATIONS),batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(BirdDataset(val_df, MY_TRANSFORMATIONS),batch_size=BATCH_SIZE,shuffle=False)
    test_loader = DataLoader(BirdDataset(test_df, MY_TRANSFORMATIONS),batch_size=BATCH_SIZE,shuffle=False)

    return train_loader, val_loader, test_loader, classes, 256



def get_Chaoyang_loaders(BATCH_SIZE = 16, SEED=42, SPLITS=[0.8,0.1,0.1]):
    PATH_TRAIN = './chaoyang-data/train'
    PATH_TEST = './chaoyang-data/test'
    classes = [0,1,2,3]
    #TODO transformation are applied internally in from_path_to_dataloader 
    MY_TRANSFORMATIONS = A.Compose([
        A.Resize(256,256),
        ToTensorV2()
        ])
    
    train_dataloader = from_path_to_dataloader(PATH_TRAIN, BATCH_SIZE, shuffle=True, need_augmentation=True)
    test_dataloader  = from_path_to_dataloader(PATH_TEST,  BATCH_SIZE, shuffle=False, need_augmentation=False)
    
    # # classes = train_dataloader.dataset.classes

    # # concatenate train and test dataloaders using ConcatDataset
    # from torch.utils.data.dataset import ConcatDataset
    # merged_dataset = ConcatDataset([train_dataloader.dataset, test_dataloader.dataset])
    # train_df, val_df, test_df = get_train_valid_test_split(merged_dataset, SPLITS=SPLITS, SEED=SEED)

    #split the train dataset into train and validation
    train_size = int(0.8 * len(train_dataloader.dataset))
    val_size = len(train_dataloader.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataloader.dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, classes, 224
    

def get_vegetables_dataloader(BATCH_SIZE = 16, SEED=42, SPLITS=[0.8,0.1,0.1]):
    PATH = './vegetables/'
    MY_TRANSFORMATIONS = A.Compose([
        A.Resize(256,256),
        ToTensorV2()
        ])
    
    class VegetableDataset(Dataset):
        def __init__(self,DF, TRANSFORMATIONS):
            self._df = DF
            self._transformations = TRANSFORMATIONS

            self.images = self._df['path'].values
            self.classes = self._df['class'].values
            self.classes = np.array([classes.index(i) for i in self.classes])
            
        def __len__(self):
            return len(self._df)
        
        def __getitem__(self,idx):
            image = self.images[idx]
            image = plt.imread(image)
            #image = np.transpose(image,(2,0,1))
            #if is grayscale, convert to rgb
            if image.shape[0] == 1:
                image = np.repeat(image,3,0)
            #cast to float and normalize
            image = image.astype(np.float32)
            image = image/255.0 
            image =  self._transformations(image=image)['image']
            class_ = self.classes[idx]
            class_ = torch.tensor(class_,dtype=torch.long)
            image = image.type(torch.FloatTensor)
            if image.shape[0] == 1:
                image = torch.repeat_interleave(image,3,0)
            return image,class_
        
    #read all files from the folder CUB_200_2011 and assign the subfolder as a class
    #the subfolder name is the class name
    classes = os.listdir(PATH)
    classes.sort()
    print(classes)

    #read all files from the subfolders
    data = []
    for i in range(len(classes)):
        folder = os.path.join(PATH,classes[i])
        files = os.listdir(folder)
        for j in range(len(files)):
            data.append([classes[i],os.path.join(folder,files[j])])

    #convert the list to a dataframe
    df = pd.DataFrame(data,columns=['class','path'])
    df.head()

    #split the data into train and test and validation
    train_df, val_df, test_df = get_train_valid_test_split(df, SPLITS=SPLITS, SEED=SEED)
    
    #create the dataloaders
    train_loader = DataLoader(VegetableDataset(train_df, MY_TRANSFORMATIONS),batch_size=BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(VegetableDataset(val_df, MY_TRANSFORMATIONS),batch_size=BATCH_SIZE,shuffle=False)
    test_loader = DataLoader(VegetableDataset(test_df, MY_TRANSFORMATIONS),batch_size=BATCH_SIZE,shuffle=False)

    return train_loader, val_loader, test_loader, classes, 256