#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import time
import copy
import argparse
import shutil
import cPickle
import torch, cv2
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import torch.utils.data as Data
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter
from preprocessingdata import myImageFloder
from vgg16places365_torch import vgg16


# Training settings
parser = argparse.ArgumentParser(description='Pytorch Scene Recognition Training')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                   help='input batch size for training(default:100)')
parser.add_argument('--num_classes', type=int, default=80, metavar='N',
                    help='the total classes kinds number of scene classifier')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                   help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                   help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', type=str, default='', metavar='PATH',
                    help='path to latest checkpoint (default:none)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                   help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='SGD momentum')
parser.add_argument('--weight_decay', '--wd', type=float, default=1e-4, metavar='W',
                   help='wight decay(default: 1e-4)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                   help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                   help='random seed(default:1)')
parser.add_argument('--log_train_interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
parser.add_argument('--log_val_interval', type=int, default=15,metavar='N',
                    help='how many batches to wait before logging validating staus')
parser.add_argument('--arch', '-a', default='vgg16', metavar='ARCH',
                   help='the deep model architecture (default: vgg16)')
parser.add_argument('--predmodel_path', type=str, default='/home1/haoyanlong/AI/scenerecognition/caffe_pytorch/VGG16/vgg16places365_torch.pth',
                    help='the palces365 pred model path')

parser.add_argument('--log', type=str, default='/home1/haoyanlong/AI/scenerecognition/logs/vgg16_addmaping',
                    help='the parameters logging')

parser.add_argument('--checkpoint_path', type=str, default='/home/haoyanlong/AI/modelcheckpoint/vgg16_addmaping/',
                    help='the directory of model checkpoint saved')
best_prec3 = 0


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        # Because the CrossEntropyLoss() is calculated averagely on mini-batch
        # size_average (bool, optional)–By default, the losses are averaged over observations for each minibatch
        # so, val * n means the total loss on the mini-batch
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30epochs
    :param optimizer:the optimizer class
    :param epoch:int the training epoch
    :return:the lr
    """
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_groups in optimizer.param_groups:
        param_groups['lr'] = lr


def accuracy(output, label, topk=(1,3,5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print pred.type()
    # print pred.size()
    # print label.type()
    # print label.size()
    correct = torch.eq(pred, label.view(1,-1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    :param state: the model state
    :param is_best: the current whether is the best
    :param filename: the name of filename saved
    :return:
    """
    torch.save(state, filename + '_lastest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_lastest.pth.tar', filename + '_best.path.tar')


# def finemodel(predmodel):
#     """
#     :param predmodel: the model pretrained on places365
#     :return: the model fine changed
#     """
#     vgg16.load_state_dict(torch.load(predmodel))
#     for i, params in enumerate(vgg16.parameters()):
#          params.requires_grad = False
#
#     # the last features numbers
#     features_number = vgg16[-1][-1].in_features       # 4096
#     model = nn.Sequential(*list(vgg16[i] for i in range(38)))
#     model.add_module('scene_classify',nn.Sequential(vgg16[38][0], nn.Linear(features_number, args.num_classes)))
#     return model

def addPlace2game(premodel):
    """
    :param premodel: the vgg16 model pretrained on places365
    :return:the model adding module of place365 mapping 80
    """
    vgg16.load_state_dict(torch.load(premodel))
    for i, params in enumerate(vgg16.parameters()):
        params.requires_grad = False

    # the places365 kinds number
    places365_numbers = vgg16[-1][-1].out_features
    vgg16.add_module('places2game', nn.Linear(places365_numbers, args.num_classes))
    return vgg16


# fine-tuning the predmodel
def train(train_dataset, predmodel, criterion, optimizer, epoch, logger):
    """
    :param predmodel: (model) fine-changed model
    :param epoch: the current training epoch
    :return:
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    predmodel.train()

    end = time.time()

    for batch_idx, (image, label) in enumerate(train_dataset,1):

        step = epoch * 539 + batch_idx
        # print "iteration times:", step
        # measure data loading time
        data_time.update(time.time() - end)

        if not args.no_cuda and torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()

        image, label = Variable(image), Variable(torch.squeeze(label.long()))
        # print image.size()
        # print image.data.type()
        # print label.size()
        # print label.data.type()

        # compute output
        output = predmodel(image)
        # print output.size()
        # print output.data.type()
        loss = criterion(output, label)

        # measure accuracy and record loss
        prec1, prec3, prec5 = accuracy(output=output.data, label=label.data, topk=(1,3,5))
        losses.update(loss.data[0], image.size(0))
        top1.update(prec1, image.size(0))
        top3.update(prec3, image.size(0))
        top5.update(prec5, image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print 'EPOCH: {}, batchs: {}, Loss: {}'.format(epoch, batch_idx, loss.data[0])

        if batch_idx % args.log_train_interval == 49:

            print ('Train Epoch: [{0}], Train processing:[{1} / {2}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val[0]:.3f} ({top1.avg[0]:.3f})\t'
                   'Prec@3 {top3.val[0]:.3f} ({top3.avg[0]:.3f})\t'
                   'Prec@5 {top5.val[0]:.3f} ({top5.avg[0]:.3f})\t'.format(
                    epoch, batch_idx, len(train_dataset),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,top3=top3,top5=top5)
            )


            ################# tensorboard logging ######################

            # (1) Log the scalar values
            info = {
                'train_loss_val': losses.val,
                'train_loss_avg': losses.avg,
                'train_top1_val': top1.val[0],
                'train_top1_avg': top1.avg[0],
                'train_top3_val': top3.val[0],
                'train_top3_avg': top3.avg[0],
                'train_top5_val': top5.val[0],
                'train_top5_avg': top5.avg[0]
                }
            for tag, value in info.items():
                logger.add_scalar(tag, value, step)
            # (2) Log values and gradients of the parameter (histogram)
            for tag, value in predmodel.places2game.named_parameters():
                tag = tag.replace('.','/')
                logger.add_histogram('train'+tag, value.data.cpu().numpy(), step)
                logger.add_histogram('train'+tag+'/grad', value.grad.data.cpu().numpy(), step)

            # (3) Log the images
            log_image = image.data.view(-1,3,224,224).cpu()[:10]

            for i in range(len(log_image)):
                logger.add_image('train_image'+str(i), log_image[i], step)


# Validation the finedmodel
def validate(validation_dataset, predmodel, criterion, epoch,logger):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    top5 = AverageMeter()


    # switch to evaluate mode
    predmodel.eval()

    end = time.time()

    for idx, (image, label) in enumerate(validation_dataset,1):

        step = epoch * 539
        if not args.no_cuda and torch.cuda.is_available():
            image, label = image.cuda(), label.cuda()

        image, label = Variable(image, volatile=True), Variable(torch.squeeze(label.long()), volatile=True)

        # compute output
        output = predmodel(image)
        loss = criterion(output, label)

        # measure accuracy and record loss
        prec1, prec3, prec5 = accuracy(output=output.data, label=label.data, topk=(1, 3, 5))
        losses.update(loss.data[0], image.size(0))
        top1.update(prec1, image.size(0))
        top3.update(prec3, image.size(0))
        top5.update(prec5, image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.log_val_interval == 14:

            print ('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val[0]:.3f} ({top1.avg[0]:.3f})\t'
                   'Prec@3 {top3.val[0]:.3f} ({top3.avg[0]:.3f})\t'
                   'Prec@5 {top5.val[0]:.3f} ({top5.avg[0]:.3f})'.format(
                    idx,len(validation_dataset),
                    batch_time=batch_time,
                    loss = losses,
                    top1=top1,top3=top3,top5=top5))


            ################# tensorboard logging ######################
            # (1) Log the scalar values
            info = {
                'vali_loss_val': losses.val,
                'vali_loss_avg': losses.avg,
                'vali_top1_val': top1.val[0],
                'vali_top1_avg': top1.avg[0],
                'vali_top3_val': top3.val[0],
                'vali_top3_avg': top3.avg[0],
                'vali_top5_val': top5.val[0],
                'vali_top5_avg': top5.avg[0]
            }
            for tag, value in info.items():
                logger.add_scalar(tag, value, step)
            # (2) Log values and gradients of the parameter (histogram)
            for tag, value in predmodel.places2game.named_parameters():
                tag = tag.replace('.', '/')
                logger.add_histogram('validation'+tag, value.data.cpu().numpy(), step)

            # (3) Log the images
            log_image = image.view(-1, 3, 224, 224).data.cpu()[:10]

            for i in range(len(log_image)):
                logger.add_image('vali_image'+str(i), log_image[i], step)


    print ('Prec@1 {top1.avg[0]:.3f} Prec@3 {top3.avg[0]:.3f} Prec@5 {top5.avg[0]:.3f}'.format(top1=top1, top3=top3, top5=top5))

    # return top1.avg, top3.avg, top5.avg
    return top3.avg[0]


def main():

    global args, best_prec3, step
    args = parser.parse_args()
    print args

    # logger event
    logger = SummaryWriter(args.log)

    # Preparing data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataloader = myImageFloder(
        root='/home1/haoyanlong/AI/scenerecognition/scene_datasets/ai_challenger_scene_train_20170904/scene_train_images_20170904',
        label='/home1/haoyanlong/AI/scenerecognition/scene_datasets/ai_challenger_scene_train_20170904/train_data.txt',
        transform=transforms.Compose(
            [transforms.RandomSizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             normalize]
        ))
    train_data = Data.DataLoader(dataset=train_dataloader,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=2)

    val_dataloader = myImageFloder(
        root='/home1/haoyanlong/AI/scenerecognition/scene_datasets/ai_challenger_scene_validation_20170908/scene_validation_images_20170908',
        label='/home1/haoyanlong/AI/scenerecognition/scene_datasets/ai_challenger_scene_validation_20170908/validation_data.txt',
        transform=transforms.Compose(
            [transforms.Scale(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             normalize]
        ))
    val_data = Data.DataLoader(dataset=val_dataloader,
                               batch_size=args.batch_size,
                               shuffle=False,
                               num_workers=2)

    # create model
    print "=> creating model"
    VGG16_model = addPlace2game(args.predmodel_path)
    print VGG16_model

    if not args.no_cuda and torch.cuda.is_available():
        VGG16_model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print ("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec3 = checkpoint['best_prec3']

            VGG16_model.load_state_dict(checkpoint['model_arch'])
            print ("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print ("=> no checkpoint found at '{}'".format(args.resume))


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(VGG16_model.places2game.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer=optimizer,
                             epoch=epoch)

        # train for one epoch
        train(train_dataset=train_data, predmodel=VGG16_model, criterion=criterion, optimizer=optimizer, epoch=epoch,logger=logger)

        # evalute on validation set
        prec3 = validate(validation_dataset=val_data, predmodel=VGG16_model, criterion=criterion,epoch=epoch, logger=logger)

        # remember best prec@1 and save checkpoint
        is_best = prec3 > best_prec3
        best_prec3 = max(prec3, best_prec3)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'model_arch': VGG16_model.state_dict(),
            'best_prec3': best_prec3},
            is_best=is_best,
            filename=args.checkpoint_path + args.arch.lower())

    logger.close()


if __name__ == '__main__':
    main()

