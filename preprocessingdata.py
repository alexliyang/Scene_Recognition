#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from PIL import Image
import numpy as np


def default_loader(path):
    return Image.open(path).convert('RGB')


class myImageFloder(Data.Dataset):

    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
        fn = open(label)
        c = 0
        imgs = []

        for line in fn.readlines():
            cls = line.strip().split('  ')
            fn = cls.pop(0)
            if os.path.isfile(os.path.join(root, fn)):
                imgs.append([fn, int(cls[0])])
            c=c+1

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, fn))

        """
        Note:
            numpy image: H * W * C
            torch image: C * H * W  
        """

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.Tensor(list([label]))

    def __len__(self):
        return len(self.imgs)


def testmyImageFolder():
    dataset = myImageFloder(
        root='/home1/haoyanlong/AI/scenerecognition/scene_datasets/train_rawdata/images',
        label='/home1/haoyanlong/AI/scenerecognition/scene_datasets/train_rawdata/train_data.txt',
        transform=transforms.Compose(
            [transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()]
        )
    )

    imgLoader = Data.DataLoader(dataset=dataset,batch_size=100,shuffle=True,num_workers=2)
    return imgLoader


    # for i,data in enumerate(imgLoader, 0):
    #     print i
    #     print type(data)
    #     print len(data)
    #     print type(data[0])
    #     print data[0].size()
    #
    #     print type(data[1])
    #     print data[1].size()
    #
    #     if i == 0:
    #         break

if __name__ == '__main__':
    train_dataset = testmyImageFolder()


    print len(train_dataset)



    print '-------------------------------------'

    print '一千个伤心的理由'









