import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
import xlrd
import xlwt
from xlutils.copy import copy
from skimage import io

import pdb

from nets import ssd_vgg_300, ssd_common, np_methods
#from nets import ssd_vgg_512, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

classes_num2str = {'0':'','1':'飞机','2':'自行车','3':'鸟','4':'船','5':'瓶子','6':'公交车','7':'汽车','8':'猫','9':'椅子','10':'牛','11':'餐桌','12':'狗','13':'马','14':'摩车车','15':'人','16':'盆栽','17':'羊','18':'沙发','19':'火车','20':'电视'}


# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

slim = tf.contrib.slim
sys.path.append('../')

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = 'checkpoints/ssd_300_vgg.ckpt'
# ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

#Open the excel file which is needed to be written
'''xlsfile = r"/home/mrc/MCcode/minicv/SSD-Tensorflow-master/article_attr.xls"# 打开指定路径中的xls文件
article = xlrd.open_workbook(xlsfile)#得到Excel文件的book对象，实例化对象
w_article = copy(article)
sheet0 = w_article.get_sheet(0) # 通过sheet索引获得sheet对象'''

xlsfile = r"/home/mrc/MCcode/minicv/article_data/origin.xlsx"# 打开指定路径中的xls文件
r_article = xlrd.open_workbook(xlsfile)#得到Excel文件的book对象，实例化对象
r_sheet = r_article.sheet_by_index(0)

new_xl = xlwt.Workbook(encoding='utf-8',style_compression=0)
new_sheet = new_xl.add_sheet('sheet1',cell_overwrite_ok=True)
new_sheet.write(0,0,'article_id')
new_sheet.write(0,1,'images_url')
new_sheet.write(0,2,'tags')
new_sheet.write(0,3,'contain_human')
new_sheet.write(0,4,'human_tag')
new_sheet.write(0,5,'from_car')
new_sheet.write(0,6,'car_tag')
new_xl.save(r"/home/mrc/MCcode/minicv/SSD-Tensorflow-master/article_attr_phase2.xls")

# Test on some demo image and visualize output.
results_urls = ''
object_tags = ''
contain_human = ''

#Main loop to generate excel which contain the information from images in articles
for xl_row in range(0,r_sheet.nrows):
    # Add the number of article first
    new_sheet.write(xl_row+1,0,str(xl_row+1))


    if r_sheet.cell_value(xl_row,1)=='汽车':
        new_sheet.write(xl_row+1,5,str(1))
    else:
        new_sheet.write(xl_row+1,5,str(0))

    #Obtain the urls from excel. If there is no image url in this cell, continue directly
    url_images = r_sheet.cell_value(xl_row,4)
    if len(url_images) <1:
        continue

    #Split urls according to ','
    url_images = url_images.split(',')

    #We only detect the first five images in this article
    if len(url_images)>10:
        url_images = url_images[0:10]

    #Main loop to analyze images and get their information  
    for url in range(0,len(url_images)):
        #Get the single url every time and read the corresponding image into memory.
        image_url = url_images[url]
        image = io.imread(image_url)

        # Whether the image in url satisfy the demmand of API
        if len(image.shape)>3 or image.shape[2]>3 or image.shape[0]<=50 or image.shape[1]<=50:
            continue

        #Record the images we have analyzed
        results_urls = results_urls + url_images[url] + ';'

        rclasses, rscores, rbboxes =  process_image(image)
        rclasses = rclasses[np.where(rscores>0.7)]

        if rclasses.shape[0] == 0:
            object_tags = object_tags + ';'
            contain_human = contain_human + '0' + ';'
            continue

        for ssd_class in range(0,rclasses.shape[0]):
            object_tags = object_tags + classes_num2str[str(rclasses[ssd_class])] + ','
        object_tags = object_tags[:-1] + ';'

        if np.sum(rclasses == 15)>0:
            contain_human = contain_human + '1' + ';'
        else:
            contain_human = contain_human + '0' + ';'

    if results_urls!='':
        new_sheet.write(xl_row+1,1,results_urls[:-1])
    new_sheet.write(xl_row+1,2,object_tags)
    new_sheet.write(xl_row+1,3,contain_human)
    new_xl.save("article_attr_phase2.xls")
    results_urls = ''
    object_tags = ''
    contain_human = ''
