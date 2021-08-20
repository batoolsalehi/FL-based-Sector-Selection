from __future__ import division

import sys  
import os
import csv
import shutil
import numpy as np
import scipy.spatial.distance as dist
from pypcd import pypcd
from datetime import datetime
import zipfile
import ast
import cv2
from tqdm import tqdm
##user input line: 126

def save_npz(path,train_data,val_data,test_data):
    check_and_create(path)
    np.savez_compressed(path+'train_data.npz', train=train_data)
    np.savez_compressed(path+'val_data.npz', val=val_data)
    np.savez_compressed(path+'test_data.npz', test=test_data)

def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False


def show_all_files_in_directory(input_path,extension):
    'This function reads the path of all files in directory input_path'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(extension):
                files_list.append(os.path.join(path, file))
    return files_list


def quantizeJ(signal, partitions):
    xmin = min(signal)
    xmax = max(signal)
    M = len(partitions)
    delta = partitions[2] - partitions[1]
    quantizerLevels = partitions
    xminq = min(quantizerLevels)
    xmaxq = max(quantizerLevels)
    x_i = (signal-xminq) / delta #quantizer levels
    x_i = np.round(x_i)
    ind = np.where(x_i < 0)
    x_i[ind] = 0
    ind = np.where(x_i>(M-1))
    x_i[ind] = M-1; #impose maximum
    x_q = x_i * delta + xminq;  #quantized and decoded output

    return list(x_i)



def lidar_preprocesing(pcd_file,vehicle_position,QP,TX,max_dist_LIDAR):

    #Tx position
    # Tx = [746, 560, 4]
    max_dist_LIDAR = 100 # in meters

    dx = np.arange(QP['Xmin'],QP['Xmax'],QP['Xp'])
    dy = np.arange(QP['Ymin'],QP['Ymax'],QP['Yp'])
    dz = np.arange(QP['Zmin'],QP['Zmax'],QP['Zp'])

    MD = np.zeros((np.size(dx),np.size(dy),np.size(dz)))
    pc = pypcd.PointCloud.from_path(pcd_file)

    ##############Only keep valid measurnments
    valid_elements_x = np.where(np.logical_and(pc.pc_data['x']<max_dist_LIDAR,pc.pc_data['x']>-max_dist_LIDAR))[0]
    valid_elements_y = np.where(np.logical_and(pc.pc_data['y']<max_dist_LIDAR,pc.pc_data['y']>-max_dist_LIDAR))[0]
    valid_elements_z = np.where(np.logical_and(pc.pc_data['z']<max_dist_LIDAR,pc.pc_data['z']>-max_dist_LIDAR))[0]
    ####Union
    valid_elements = np.intersect1d(np.intersect1d(valid_elements_x, valid_elements_y),valid_elements_z)
    # print('# of valid measurment for this files:',len(valid_elements))
    X = pc.pc_data['x'][valid_elements]
    Y = pc.pc_data['y'][valid_elements]
    Z = pc.pc_data['z'][valid_elements]

    # print('# of positive and negative samples',len(np.where(np.logical_and(X<max_dist_LIDAR,X>0))[0]),len(np.where(np.logical_and(X>-max_dist_LIDAR,X<0))[0]))
    #######Adjust the cooridinates to the universal defined system
    X = X + vehicle_position[0]
    Y = Y + vehicle_position[1]
    Z = Z + vehicle_position[2]

    ####### Detect the index of obstalces=> tmp cloud
    indx = quantizeJ(X,dx)
    indx = [int(i) for i in indx]
    indy = quantizeJ(Y,dy)
    indy = [int(i) for i in indy]
    indz = quantizeJ(Z,dz)
    indz = [int(i) for i in indz]

    ####RX index
    Rx_q_indx = quantizeJ([vehicle_position[0]],dx)
    Rx_q_indy = quantizeJ([vehicle_position[1]],dy)
    Rx_q_indz = quantizeJ([vehicle_position[2]],dz)
    ###Tx index
    Tx_q_indx = quantizeJ([Tx[0]],dx)
    Tx_q_indy = quantizeJ([Tx[1]],dy)           
    Tx_q_indz = quantizeJ([Tx[2]],dz)


    ###################Map tp MD matrix repreenation
    # Obstacles = 1
    for i in range(len(indx)):
        MD[indx[i],indy[i],indz[i]] = 1
    # Tx -1 Rx -2
    MD[int(Tx_q_indx[0]),int(Tx_q_indy[0]),int(Tx_q_indz[0])] = -1
    MD[int(Rx_q_indx[0]),int(Rx_q_indy[0]),int(Rx_q_indz[0])] = -2

    return MD


#####################################################Start
# csv_file = '/home/batool/Desktop/infocom/Synchornized_data.csv'




path_directory = '/media/batool/MULTIMODAL_DATA/Multimodal_Data/Cat4/car_20_lr_blockage_20_rl/bag_files'
bag_files = show_all_files_in_directory(path_directory,'.bag')
print(bag_files)

# TX location, static
# Tx = [(Xmax + Xmin)/2, (Ymax + Ymin)/2, 2]    # Be careful to change later
Tx = [32.7687648776171, -32.43832477769123,1.5]
#max_dist_LIDAR
max_dist_LIDAR = 80
number_of_beams = 64


for bf in tqdm(bag_files):
    bag_file_name = bf.split('/')[-1].split('.')[0]
    episode = bag_file_name.split('_')[4]
    csv_file = path_directory[:-9]+'Extract/'+bag_file_name[:-2]+'/episode_'+str(episode)
    print('csv_file',csv_file)

    with open(csv_file+'/Synchornized_data.csv', 'r') as f:
    	reader = csv.reader(f,delimiter=',')
    	data = list(reader)

    all_x = [float(d[-2]) for d in data[1:]]
    all_y =  [float(d[-1]) for d in data[1:]]
    print('all_x',all_x)
    Xmin, Xmax = min(all_x), max(all_x)
    Ymin, Ymax = min(all_y), max(all_y)
    Zmin, Zmax = 0, 10
    blocks = 20
    Xp, Yp, Zp = (Xmax - Xmin)/ blocks , (Ymax -Ymin)/blocks, (Zmax -Zmin)/blocks

    QP = {'Xp':Xp,'Yp':Yp,'Zp':Zp,'Xmax': Xmax,'Ymax': Ymax, 'Zmax': Zmax, 'Xmin': Xmin,'Ymin': Ymin, 'Zmin': Zmin}
    print('Quantization paramethers',QP)


    print('************preprocessing and prepraing for ML pipeline******************')

    all_MD_generated = np.empty([len(data), blocks, blocks, blocks], dtype=int)
    all_GPS = np.empty([len(data), 2], dtype=float)
    all_RF = np.zeros([len(data), number_of_beams], dtype=float)
    all_image = np.empty([len(data),360,640,3], dtype=float)


    for l in range(1,len(data)):
        vehicle_position = (float(data[l][17]) , float(data[l][18]), 3.5)
        pcd_file = data[l][6]

        ##lidar
        MD = lidar_preprocesing(pcd_file,vehicle_position,QP,Tx,max_dist_LIDAR) 
        ##lidar
        all_MD_generated[l] = MD
        all_GPS[l][0] = vehicle_position[0]
        all_GPS[l][1] = vehicle_position[1]
        ###RF_GT
        all_sectors = ast.literal_eval(data[l][16])
        all_rssi = ast.literal_eval(data[l][15]) 
        # print('compare lens',len(all_sectors),len(all_rssi),all_sectors)
        for r in range(len(all_sectors)): 
            # print(r,all_sectors[r])
            all_RF[l][all_sectors[r]] = all_rssi[r]  
        ####Image
        image = cv2.imread(data[l][7])
        print('shape',image.shape)
        if image.shape[0]!=720 and image.shape[1]!=1280:       #some videos have diffrent size
            print('video has diffrent shape')
            image = cv2.resize(image, (1280,720))

        resized_image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
        all_image[l] = resized_image

    print('check shapes',len(all_MD_generated),len(all_GPS),len(all_image),len(all_RF),len(all_MD_generated[1:]),len(all_GPS[1:]),len(all_image[1:]),len(all_RF[1:]))


    save_directory = csv_file+'/npz/'
    print('save_directory',save_directory)
    check_and_create(save_directory)
    print('************Save lidar to npz******************')
    np.savez_compressed(save_directory+'lidar.npz', lidar=all_MD_generated[1:])     # because the first row is the text of information

    print('************Save GPS to npz******************')
    np.savez_compressed(save_directory+'gps.npz', gps=all_GPS[1:])

    print('************Save RF to npz******************')
    np.savez_compressed(save_directory+'rf.npz', rf=all_RF[1:])

    print('************Save image to npz******************')
    np.savez_compressed(save_directory+'image.npz', img=all_image[1:])


#############################################



###old
# shuffle =  True
# if shuffle:
#     number_of_samples =len(all_MD_generated)
#     randperm = np.random.permutation(number_of_samples)
# else:
#     randperm = range(number_of_samples)


# print('check shapes', all_MD_generated.shape, all_GPS.shape,all_RF.shape)

# print('************Save lidar to npz******************')
# #### split to train/val/test
# lidar_train = all_MD_generated[randperm[:int(0.8*len(all_MD_generated))]]
# lidar_val = all_MD_generated[randperm[int(0.8*len(all_MD_generated)):int(0.9*len(all_MD_generated))]]
# lidar_test = all_MD_generated[randperm[int(0.9*len(all_MD_generated)):]]

# save_npz('/home/batool/Desktop/infocom/npz_files/lidar/',lidar_train, lidar_val, lidar_test)


# print('************Save GPS to npz******************')
# #### split to train/val/test
# GPS_train = all_GPS[randperm[:int(0.8*len(all_GPS))]]
# GPS_val = all_GPS[randperm[int(0.8*len(all_GPS)):int(0.9*len(all_GPS))]]
# GPS_test = all_GPS[randperm[int(0.9*len(all_GPS)):]]

# save_npz('/home/batool/Desktop/infocom/npz_files/GPS/',GPS_train, GPS_val, GPS_test)


# print('************Save RF to npz******************')
# #### split to train/val/test
# RF_train = all_RF[randperm[:int(0.8*len(all_RF))]]
# RF_val = all_RF[randperm[int(0.8*len(all_RF)):int(0.9*len(all_RF))]]
# RF_test = all_RF[randperm[int(0.9*len(all_RF)):]]

# save_npz('/home/batool/Desktop/infocom/npz_files/RF/',RF_train, RF_val, RF_test)



# print('Check train',lidar_train.shape,GPS_train.shape,RF_train.shape,'\n check val shape',lidar_val.shape,GPS_val.shape,RF_val.shape,'\n check test shape',lidar_test.shape,GPS_test.shape,RF_test.shape)

