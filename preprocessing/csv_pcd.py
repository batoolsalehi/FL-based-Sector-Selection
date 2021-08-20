# wriiten in python 3.7.6
# Input: path_directory line 32
# Run the bash script first to generate pcd files: ./pcd_bash.sh /home/batool/Desktop/infocom/bag_files

import bagpy
import os,glob 
from bagpy import bagreader
import pandas as pd
import numpy as np
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




path_directory = '/media/batool/MULTIMODAL_DATA/Multimodal_Data/Cat4/car_20_lr_blockage_20_rl/bag_files'     #cat1
# path_directory = '/media/batool/Seagate/MultiModal_Data/Cat3/static_in_front/bag_files'

bag_files = show_all_files_in_directory(path_directory,'.bag')
print(bag_files)

print('*************Attaching experiment information to data****************')
for bf in bag_files:
    data = []
    pcd_files_folder = bf.split('.')[0]+'/lidar_pcd'+'/ns1'+'/velodyne_points'
    pcd_files = show_all_files_in_directory(pcd_files_folder,'.pcd')
    for files in pcd_files:
        file_name = files.split('/')[-1]
        time_header = file_name.split('.')[0]+'.'+file_name.split('.')[1]
        # time_ns = file_name.split('.')[1]
        lidar_number = files.split('/')[-3]
        experiment_name = files.split('/')[-5]
        # print(time_header,time_ns,lidar_number)
        # experiment info incldues: Category, speed, direction, lane episode  
        experiment_info = [experiment_name.split('_')[0],experiment_name.split('_')[1],experiment_name.split('_')[2],experiment_name.split('_')[3],experiment_name.split('.')[0].split('_')[-1]]
        data.append(experiment_info+[time_header]+[lidar_number]+[files])

    header = ['Category', 'speed', 'direction', 'lane','episode','Time header','lidar number','path to the file']

    episode_index = bf.split('/')[-1].split('.')[0].split('_')[-1]
    print(bf)        #example of bf /media/batool/Seagate/MultiModal_Data/Cat1/same/bag_files/Cat1_20mph_lr_same_9.bag
    save_directory = path_directory[:-9]+'Extract/'+bf.split('/')[-1].split('.')[0][:-2]+'/episode_'+str(episode_index)+'/'           #[:-2] for removing _(episode_number)
    check_and_create(save_directory)

    with open(save_directory+'pcd_data.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)