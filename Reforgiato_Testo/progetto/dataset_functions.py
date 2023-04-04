import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import random
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

path_train = './chaoyang-data/train'
path_test = './chaoyang-data/test'

def read_dataset(info_filename):
    # for each image in the folder, read the image and the corresponding label
    # return a list of images and a list of labels
    dataset = {}

    # read the info file json which contains the labels for each image
    # read the json file
    # Opening JSON file
    f = open(info_filename+'.json')
    structure = json.load(f)
    # Closing file
    f.close()
    # the structure is a dictionary with the keys: 'label','name' bring the labels and the names of the images
    for i in range(len(structure)):
        dataset[i] = (structure[i]['name'],structure[i]['label'])

    return dataset

def get_train_transforms():
    return A.Compose([
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #rotate the image by 90 180 or 270 degrees
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(384, 384, scale=(0.5, 1.0), p=0.5),
        #A.GridDistortion(p=0.5, num_steps=5, distort_limit=0.3),
        ToTensorV2(p=1.0),
    ], p=1.)

def get_valid_transforms():
    return A.Compose([
        A.Resize(384, 384),
        ToTensorV2(p=1.0),
    ], p=1.)

def augment(dataset):
    #for each distribution of the dataset, calculate the number of images that need to be added
    #to have a balanced dataset
    #return the augmented dataset
    augmented_dataset = {}
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i][1])
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    max_count = max(class_distribution.values())
    # for each class, create a sub dataset
    sub_datasets = {}
    for i in range(len(unique)):
        sub_datasets[i] = {}
    for i in range(len(dataset)):
        sub_datasets[dataset[i][1]][i] = dataset[i]
    # for each sub dataset, calculate the number of images that need to be added
    # to have a balanced dataset
    for i in range(len(unique)):
        sub_datasets[i]['number_of_images_to_add'] = max_count - class_distribution[i]
    # for each sub dataset, add the images random images from the same class
    for i in range(len(unique)):
        for j in range(sub_datasets[i]['number_of_images_to_add']):
            random_index = random.choice(list(sub_datasets[i].keys()))
            #if the extrcted element is a numpy.int64, then extract another element
            while type(sub_datasets[i][random_index]) == np.int64:
                random_index = random.choice(list(sub_datasets[i].keys()))
            augmented_dataset[len(augmented_dataset)] = sub_datasets[i][random_index]
            
    # add the original dataset
    for i in range(len(dataset)):
        augmented_dataset[len(augmented_dataset)] = dataset[i]
    #remove rows if the type is numpy.int64
    augmented_dataset = {k: v for k, v in augmented_dataset.items() if type(k) != np.int64}

    

    return augmented_dataset
path_swin="microsoft/swinv2-base-patch4-window16-256"
path_vit="google/vit-base-patch16-384"
from transformers import AutoImageProcessor, AutoModelForImageClassification
image_processor = AutoImageProcessor.from_pretrained(path_vit)
class Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join('./chaoyang-data/', self.dataset[idx][0])
        image = cv2.imread(img_name)
        #print(image.shape)
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        if image.shape[1] == 1:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image=image_processor(images=image, return_tensors="pt")
        #print(image['pixel_values'].shape)

        # read the label
        label = self.dataset[idx][1]

        return image, label
    
def from_path_to_dataloader(path, batch_size, shuffle, need_augmentation):
    # read the dataset
    dataset = read_dataset(path)
    # augment the dataset
    if need_augmentation:
        dataset = augment(dataset)
    # create the dataloader
    dataloader = DataLoader(Dataset(dataset, transform=get_train_transforms()), batch_size=batch_size, shuffle=shuffle)
    # return dataloader
    return dataloader   

if __name__ == "__main__":
    #check dataloader
    train_dataloader = from_path_to_dataloader(path_train, 32, True, True)
    test_dataloader = from_path_to_dataloader(path_test, 32, False, False)

    #check the dataloader
    verbose = False
    if verbose:
        for i, data in enumerate(train_dataloader):
            print(i, data)
            if i == 3:
                break