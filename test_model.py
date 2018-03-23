#!/usr/bin/python
# -*- coding:utf-8 -*

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.utils.data as Data
from PIL import Image
import numpy as np
from vgg16places365_torch import vgg16



# setting the parameters
parser = argparse.ArgumentParser(description='Pytorch Scene Recognition Testing')
parser.add_argument('--num_classes', type=int, default=80, metavar='N',
                    help='the total classes kinds number of scene classifier')
parser.add_argument('--batch_size', type=int, default=70, metavar='N',
                   help='input batch size for training(default:100)')
parser.add_argument('--testdata_path', type=str, default='/home1/haoyanlong/AI/scenerecognition/scene_datasets/ai_challenger_scene_test_a_20170922/scene_test_a_images_20170922',
                    help='the test images path')
parser.add_argument('--predmodel', type=str, default='/home1/haoyanlong/AI/scenerecognition/place365_premodel/whole_densenet161_places365.pth.tar',
                    help='the pretrained model')
parser.add_argument('--trainedmodel', type=str, default='/home1/haoyanlong/AI/scenerecognition/modelcheckpoint/densenet_finetuning_fully/densenet161_finetuning_fully_best.path.tar',
                    help='the finetunned model path')
parser.add_argument('--JSONpath', type=str, default='/home1/haoyanlong/AI/scenerecognition/result/densenet_finetuning_fully/densenet161.json',
                    help='the file path used to save the json result')


def default_loader(path):
    return Image.open(path).convert('RGB')


# define the test dataset class
class testImageFloder(Data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):

        if os.path.isdir(root):
            imgs = os.listdir(root)
            # print type(imgs[0])

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img_name = self.imgs[index]

        img_file = self.loader(os.path.join(self.root, img_name))

        """
        Note:
            numpy image: H * W * C
            torch image: C * H * W  
        """

        if self.transform is not None:
            img_file = self.transform(img_file)

        img = [img_name,img_file]
        return img

    def __len__(self):
        return len(self.imgs)


def pretestdata(testdatapath):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_dataloader = testImageFloder(
        root=testdatapath,
        transform=transforms.Compose(
            [transforms.Scale(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             normalize]
        ))

    test_data = Data.DataLoader(dataset=test_dataloader,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=2)
    # print "the test dataset number: {}".format(len(test_dataloader))

    return test_data

def finemodel(predmodel):
    """
    :param predmodel: the model pretrained on places365
    :return: the model fine changed
    """
    # vgg16.load_state_dict(torch.load(predmodel))
    # for i, params in enumerate(vgg16.parameters()):
    #      params.requires_grad = False

    # the last features numbers
    # features_number = vgg16[-1][-1].in_features       # 4096
    # model = nn.Sequential(*list(vgg16[i] for i in range(38)))
    # model.add_module('scene_classify',nn.Sequential(vgg16[38][0], nn.Linear(features_number, args.num_classes)))


    # checkpoint_last = torch.load(args.trainedmodel)
    # print 'last accuracy:',checkpoint_last['best_prec3'], 'epoch:', checkpoint_last['epoch']
    # 'state_dict:', checkpoint_last['state_dict']
    # model.load_state_dict(checkpoint_last['model_arch'])

    densenet161 = torch.load(predmodel)
    features_number = densenet161.classifier.in_features
    del densenet161.classifier
    densenet161.classifier = nn.Sequential(nn.Dropout(), nn.Linear(features_number, args.num_classes))

    checkpoint = torch.load(args.trainedmodel)
    densenet161.load_state_dict(checkpoint['model_arch'])
    return densenet161

def testProcessing(testdata, trainedmodel):
    """
    :param testdata: the test dataset
    :param trainedmodel: haved trained model
    :return:
    """
    trainedmodel.cuda()
    result = []              # used saved the test result list(dict)[{'image_id':'','label_id':[]},...{}]
    image_names = []         # used saved the image names
    image_labels = []        # used saved the image labels
    for i ,data in enumerate(testdata):
        print 'Processing batch:{}'.format(i)

        img_name = list(data[0])
        img_data = data[1]

        image_names.extend(img_name)

        input = Variable(img_data, volatile=True).cuda()

        output = trainedmodel(input)
        # print output

        _, pred = torch.topk(output, 3, dim=1, largest=True, sorted=True)
        img_label = pred.data.cpu().numpy()

        img_label = [eachlabel.tolist() for eachlabel in img_label]

        image_labels.extend(img_label)

    # print image_names
    # print image_labels

    for each in zip(image_names, image_labels):
        cur_image = {}
        cur_image["image_id"] = each[0]
        cur_image["label_id"] = each[1]
        result.append(cur_image)

    return result


def savedJson(testreuslt,filepath):
    """
    :param testreuslt: the result of testing
    :param filepath: the path of saved file
    :return:
    """
    with open(filepath,'w') as file_json:
        json.dump(testreuslt, file_json)
        print "写入JSON文件完成"


def main():
    global args
    args = parser.parse_args()
    print args

    test_data = pretestdata(args.testdata_path)

    trainModel = finemodel(args.predmodel)
    test_result = testProcessing(test_data, trainModel)
    savedJson(testreuslt=test_result, filepath=args.JSONpath)

if __name__ == '__main__':
    main()











