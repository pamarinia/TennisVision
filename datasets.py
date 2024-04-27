from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2


class TrackNetDataset(Dataset):
    def __init__(self, mode, input_height=360, input_width=640):
        self.path_dataset = 'datasets/TrackNet'
        assert mode in ['train', 'test'], 'incorrect mode'
        self.data = pd.read_csv(os.path.join(self.path_dataset, 'labels_{}.csv'.format(mode)))
        print('mode = {}, samples = {}'.format(mode, self.data.shape[0]))
        self.height = input_height
        self.width = input_width


    def __len__(self):
        return self.data.shape[0]
    
    
    def __getitem__(self, idx):
        img_path_1, img_path_2, img_path_3, ground_truth_path, x, y, visibility, status = self.data.iloc[idx]

        img_path_1 = os.path.join(self.path_dataset, img_path_1)
        img_path_2 = os.path.join(self.path_dataset, img_path_2)
        img_path_3 = os.path.join(self.path_dataset, img_path_3)
        ground_truth_path = os.path.join(self.path_dataset, ground_truth_path)

        inputs = self.get_inputs(img_path_1, img_path_2, img_path_3)
        output = self.get_output(ground_truth_path)

        return inputs, output, x, y, visibility
    
    
    def get_inputs(self, img_path_1, img_path_2, img_path_3):
        # Resize the images to (360, 640) to speed up the training
        img_1 = cv2.imread(img_path_1)
        img_1 = cv2.resize(img_1, (self.width, self.height))
        
        img_2 = cv2.imread(img_path_2)
        img_2 = cv2.resize(img_2, (self.width, self.height))
        
        img_3 = cv2.imread(img_path_3)
        img_3 = cv2.resize(img_3, (self.width, self.height))
        
        # Concatenate the images
        imgs = np.concatenate((img_1, img_2, img_3), axis=2)
        # Normalize the images
        imgs = imgs.astype(np.float32) / 255.0
        # Change the shape to (C, H, W)
        imgs = np.rollaxis(imgs, 2, 0)

        return imgs
    
    
    def get_output(self, ground_truth_path):
        gt = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (self.width, self.height))
        gt = np.reshape(gt, (self.height * self.width))

        return gt
