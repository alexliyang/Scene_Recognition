#!/usr/bin/python
# -*- coding:utf-8 -*-

import json
import os
import time
from PIL import Image

def readAnno(filename,savefile):
    """
    :param filename:
    :param savefile:
    :return:
    """
    print "The annotations filename(JSON):%s" % filename
    print "The saved filename(txt):%s" % savefile

    since_time = time.time()
    try:
        anno_object = json.load(fp=open(filename))      # read annotation file and return list(dictionary)
    except TypeError,e:
        print e
    # print type(file_object)

    total_number = len(anno_object)
    print "the dataset's total number is %d" % total_number

    img2ids = [each['image_id'] + '  ' + each['label_id'] for each in anno_object]  # list(every
    # print img2ids[0:5]
    save_object = open(savefile,'w')
    for everyimg in img2ids:
        print "Processing: %d / %d" % (img2ids.index(everyimg)+1, total_number)
        save_object.write(everyimg)
        save_object.write('\n')
    save_object.close()
    print "During time: %s" % str(time.time() - since_time)


if __name__ == '__main__':

    # Train dataset
    # readAnno("/home/haoyanlong/AI_CHALLENGE/SceneRecognition/dataset/" +
    #          "ai_challenger_scene_train_20170904/scene_train_annotations_20170904.json",
    #          "/home/haoyanlong/AI_CHALLENGE/SceneRecognition/dataset/" +
    #          "ai_challenger_scene_train_20170904/train_data.txt")

    # Validation dataset
    readAnno("/home/haoyanlong/AI_CHALLENGE/SceneRecognition/dataset/" +
             "ai_challenger_scene_validation_20170908/scene_validation_annotations_20170908.json",
             "/home/haoyanlong/AI_CHALLENGE/SceneRecognition/dataset/" +
             "ai_challenger_scene_validation_20170908/validation_data.txt")