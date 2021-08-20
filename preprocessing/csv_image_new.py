#!/usr/bin/python
#USER input line: 33
# This script extracts images from a all ag files in a directory.
PKG = 'getimagesfrombag'
import rosbag
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import sys
from io import open
import csv 

def show_all_files_in_directory(input_path,extension):
    'This function reads the path of all files in directory input_path'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(extension):
               files_list.append(os.path.join(path, file))
    return files_list

def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False



path_directory = '/media/batool/MULTIMODAL_DATA/Multimodal_Data/Cat4/car_20_lr_blockage_20_rl/bag_files'
bag_files = show_all_files_in_directory(path_directory,'.bag')
print(bag_files)

print('*************Attaching experiment information to data****************')
for bf in bag_files:
    data = []
    img_files_folder = bf.split('.')[0]+'/camera_images'
    img_files = show_all_files_in_directory(img_files_folder,'.jpg')
    print('len of img_files',len(img_files))

    for images in img_files:          ##sample of images: /media/batool/Seagate/MultiModal_Data/Cat1/same/bag_files/Cat1_15mph_lr_same_3/camera_images/1624914012.5089483.jpg
        # experiment info incldues: Category, speed, direction, lane, episode  
        experiment = images.split('/')[-3]
        experiment_info = [experiment.split('_')[0],experiment.split('_')[1],experiment.split('_')[2],experiment.split('_')[3],experiment.split('_')[4]]
        time_header = images.split('/')[-1].split('.')[0]+'.'+images.split('/')[-1].split('.')[1]
        # print('time header',time_header)
        data.append(experiment_info+[time_header]+[images])


    header = ['Category', 'speed', 'direction', 'lane','episode','Time header','Image_path']
    episode_index = bf.split('/')[-1].split('.')[0].split('_')[-1]
    save_directory = path_directory[:-9]+'Extract/'+bf.split('/')[-1].split('.')[0][:-2]+'/episode_'+str(episode_index)+'/'
    check_and_create(save_directory)

    with open(save_directory+'Image_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)