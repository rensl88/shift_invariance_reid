#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image

from random_erasing import RandomErasing


class Market1501(Dataset):
    '''
    a wrapper of Market1501 dataset
    '''
    def __init__(self, pids_list, array, is_train = True, *args, **kwargs):
        super(Market1501, self).__init__(*args, **kwargs)
        self.is_train = is_train
        self.pids = pids_list
        self.array = array
        self.img_names = [el.split("/")[1] for el in self.pids if os.path.splitext(el)[1] == '.jpg']
        self.lb_ids = [int(el.split('_')[0]) for el in self.img_names]
        self.lb_cams = [int(el.split('_')[1][1]) for el in self.img_names]
        if is_train:
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((288, 144)),
                transforms.RandomCrop((256, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
                RandomErasing(0.5, mean=[0.0, 0.0, 0.0])
            ])
        else:
            self.trans_tuple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
                ])
            self.Lambda = transforms.Lambda(
                lambda crops: [self.trans_tuple(crop) for crop in crops])
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((288, 144)),
                transforms.TenCrop((256, 128)),
                self.Lambda
            ])

        # useful for sampler
        self.lb_img_dict = dict()
        self.lb_ids_uniq = set(self.lb_ids)
        lb_array = np.array(self.lb_ids)
        for lb in self.lb_ids_uniq:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        img = self.array[idx]
        img = self.trans(img)
        return img, self.lb_ids[idx], self.lb_cams[idx]



if __name__ == "__main__":
    ds = Market1501('./Market-1501-v15.09.15/bounding_box_train', is_train = True)
    im, _, _ = ds[1]
    print(im.shape)
    print(im.max())
    print(im.min())
    ran_er = RandomErasing()
    im = ran_er(im)
    cv2.imshow('erased', im)
    cv2.waitKey(0)