# wriiten in python 3.7.6
# Input: RF_file_directory line 89, check time line 53

import bagpy
import os,glob 
from bagpy import bagreader
import pandas as pd
import numpy as np
import csv
import time
import datetime


def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False


def extract_RF(text_files,convert_to_unix):
    sweep_file = open(text_files, "r")
    sweep_lines = sweep_file.readlines()
    end_of_sweep_flag = '---------------End of One Sector Sweeping--------------'
    sweep_count = 0

    max_rssi = []
    selected_sectors =[]
    sweep_time_list = []
    tx_mcs_list = []
    rx_mcs_list = []
    sqi_list = []
    all_rssi_list = []
    all_sector_list = []

    rssi_list = []
    sector_list = []
    for line in sweep_lines:
        if 'sec:' in line:
            sector_list.append(int(line[7:10].strip()))
            rssi_list.append(int(line[16:25].strip()))

        elif 'Time:' in line:
            ## convert to unix or not
            time = line[6:21].strip()
            # print('time',time)
            if convert_to_unix:
                hour,minute = time.split(':')[0:2]
                sec, microsec = time.split(':')[2].split('.')
                # print(hour,minute,sec,microsec)
                # if text_files.split('/')[-1].split('.')[0][-1]=='8' or text_files.split('/')[-1].split('.')[0][-1]=='9':   #for pred left to right only(52-56)
                #     print('episode',True,text_files.split('/')[-1].split('.')[0][-1])
                #     time = datetime.datetime(2021,6,29,int(hour),int(minute),int(sec),int(microsec)).timestamp()
                # else:
                #     time = datetime.datetime(2021,6,28,int(hour),int(minute),int(sec),int(microsec)).timestamp()
 
                time = datetime.datetime(2021,6,30,int(hour),int(minute),int(sec),int(microsec)).timestamp()    # change experiment date if required+norma case this is enough
            # print(time)
            sweep_time_list.append(time)

        elif 'Tx_mcs' in line:
            tx_mcs_list.append(int(line[10:].strip()))
        elif 'Rx_mcs' in line:
            rx_mcs_list.append(int(line[10:].strip()))
        elif 'SQI' in line:
            sqi_list.append(int(line[7:].strip()))

        # at the end of each beam sweep
        if line.strip() == end_of_sweep_flag:
            sweep_count+=1
            # Find maximum of RSSI and associated sector
            # max_rssi.append(max(rssi_list))
            # selected_sectors.append(sector_list[np.argmax(np.array(rssi_list))])
            all_rssi_list.append(rssi_list)
            all_sector_list.append(sector_list)
            rssi_list = []
            sector_list = []
            # if value was missing put zero
            if len(tx_mcs_list)<sweep_count: tx_mcs_list.append(0)
            if len(rx_mcs_list)<sweep_count: rx_mcs_list.append(0)
            if len(sqi_list)<sweep_count: sqi_list.append(0)

    return sweep_time_list, all_rssi_list, all_sector_list, tx_mcs_list, rx_mcs_list, sqi_list



##user inputs
RF_file_directory = '/media/batool/MULTIMODAL_DATA/Multimodal_Data/Cat4/car_20_lr_blockage_20_rl/GT'   #for cat1 switch beween same/opposite

RF_files = glob.glob(RF_file_directory+'/Sweep*.txt')
convert_to_unix = True
print(len(RF_files))


for files in RF_files:
    all_data = []
    file_name = files.split('/')[-1]        #sample file_name = Sweep_Cat1_20mph_lr_same_9.txt
    sweep_time_list, all_rssi_list, all_sector_list, tx_mcs_list, rx_mcs_list, sqi_list = extract_RF(files,convert_to_unix)
    print(len(sweep_time_list),len(all_rssi_list),len(all_sector_list),len(tx_mcs_list),len(rx_mcs_list),len(sqi_list))
    #####remove [] values
    #### find index of !=[]
    non_empty_indexes = [inde for inde in range(len(all_rssi_list)) if len(all_rssi_list[inde])!=0]
    sweep_time_list = [sweep_time_list[ind] for ind in non_empty_indexes]
    all_rssi_list = [all_rssi_list[ind] for ind in non_empty_indexes]
    all_sector_list = [all_sector_list[ind] for ind in non_empty_indexes]
    tx_mcs_list = [tx_mcs_list[ind] for ind in non_empty_indexes]
    rx_mcs_list = [rx_mcs_list[ind] for ind in non_empty_indexes]
    sqi_list = [sqi_list[ind] for ind in non_empty_indexes]

    #Find max_rssi and selected sectors
    max_rssi = [max(i) for i in all_rssi_list]
    selected_sectors = [all_sector_list[j][np.argmax(np.array(all_rssi_list[j]))] for j in range(len(all_rssi_list))]
    #Experiment info
    Category = [file_name.split('_')[1]]*len(sweep_time_list)
    speed = [file_name.split('_')[2]]*len(sweep_time_list)
    direction = [file_name.split('_')[3]]*len(sweep_time_list)
    lane = [file_name.split('_')[4]]*len(sweep_time_list)
    episode = [file_name.split('.')[0].split('_')[-1]]*len(sweep_time_list)


    data = [Category,speed,direction,lane,episode,sweep_time_list, max_rssi,selected_sectors,tx_mcs_list,rx_mcs_list,sqi_list,all_rssi_list, all_sector_list]
    if len(all_data) ==0:
        all_data = data
    else:
        all_data = [all_data[i]+data[i] for i in range(len(data))]


    header = ['Category','speed','direction','lane','episode','sweep_time_stamp','max_rssi','selected_sector','tx_mcs','rx_mcs','sqi','all_rssi', 'all_sector']
    unzip_data = [list(i) for i in zip(*all_data)]
    episode_index = file_name.split('.')[0].split('_')[-1]
    save_directory = RF_file_directory[:-2]+'Extract/'+file_name.split('.')[0][6:-2]+'/episode_'+str(episode_index)+'/'
    save = check_and_create(save_directory)
    with open(save_directory+'RF_data.csv', 'w', encoding='UTF8', newline='') as f:    ### last two alphabets are GT in all directories
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(unzip_data)