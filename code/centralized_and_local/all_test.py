########################################################
#Project name: Infocom
#Date: 14/July/2021
########################################################
from __future__ import division

import os
import csv
import argparse
import h5py
import pickle
import numpy as np
from tqdm import tqdm
import random
from time import time
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json,Model, load_model
from tensorflow.keras.layers import Dense,concatenate, Dropout, Conv1D, Flatten, Reshape, Activation,multiply,MaxPooling1D,Add,AveragePooling1D,Lambda,Permute
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adadelta,Adam, SGD, Nadam,Adamax, Adagrad
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from keras.regularizers import l2

import sklearn
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import normalize

from ModelHandler import add_model,load_model_structure, ModelHandler
from custom_metrics import *
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as precision_recall_fscore
from tensorflow.keras import backend as K

############################
# Fix the seed
############################
seed = 0
os.environ['PYTHONHASHSEED']=str(seed)
# tf.set_random_seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
# tf.random_set_seed(seed)
np.random.seed(seed)
random.seed(seed)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

def open_npz(path,key):
    data = np.load(path)[key]
    return data

def save_npz(path,train_name,train_data,val_name,val_data):
    check_and_create(path)
    np.savez_compressed(path+train_name, train=train_data)
    np.savez_compressed(path+val_name, val=val_data)

def custom_label(y, strategy='one_hot'):
    'This function generates the labels based on input strategies, one hot, reg'
    print('labeling stragey is', strategy)
    y_shape = y.shape
    num_classes = y_shape[1]
    if strategy == 'one_hot':
        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            # logOut = 20*np.log10(thisOutputs)
            max_index = thisOutputs.argsort()[-1:][::-1]  # For one hot encoding we need the best one
            y[i,:] = 0
            y[i,max_index] = 1

    elif strategy == 'reg':
        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            # logOut = 20*np.log10(thisOutputs)   # old version
            logOut = thisOutputs
            y[i,:] = logOut
    else:
        print('Invalid strategy')
    return y,num_classes


def over_k(true,pred):
    dicti = {}
    for kth in range(100):
        kth_accuracy = metrics.top_k_categorical_accuracy(true,pred,k=kth)
        with tf.Session() as sess: this = kth_accuracy.eval()
        dicti[kth] =this
    return dicti


def precison_recall_F1(model,Xtest,Ytest):
    #####For recall and precison
    y_pred = model.predict(Xtest)
    y_pred_bool = np.argmax(y_pred, axis=1)
    y_true_bool = np.argmax(Ytest, axis=1)
    return precision_recall_fscore(y_true_bool, y_pred_bool,average='weighted')


def detecting_related_file_paths(path,categories,episodes):
    find_all_paths =['/'.join(a.split('/')[:-1]) for a in show_all_files_in_directory(path,'rf.npz')]     # rf for example
    # print('find_all_paths',find_all_paths)
    selected = []
    for Cat in categories:   # specify categories as input
        for ep in episodes:
            selected = selected + [s for s in find_all_paths if Cat in s.split('/') and 'episode_'+str(ep) in s.split('/')]
    print('Getting {} data out of {}'.format(len(selected),len(find_all_paths)))

    return selected


def get_data(data_paths,modality,key):   # per cat for now, need to add per epside for FL part
    ###generate local data
    for l in tqdm(data_paths):
        randperm = np.load(l+'/ranperm.npy')
        open_file = open_npz(l+'/'+modality+'.npz',key)
        try:
            test_data = np.concatenate((test_data, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)

        except NameError:
            test_data = open_file[randperm[int(0.9*len(randperm)):]]


        ###Get data per category
        if 'Cat1' in l.split('/'):
            try:
                test_data_cat1 = np.concatenate((test_data_cat1, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)
            except NameError:
                test_data_cat1 = open_file[randperm[int(0.9*len(randperm)):]]

        elif 'Cat2' in l.split('/'):
            try:
                test_data_cat2 = np.concatenate((test_data_cat2, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)
            except NameError:
                test_data_cat2 = open_file[randperm[int(0.9*len(randperm)):]]

        elif 'Cat3' in l.split('/'):
            try:
                test_data_cat3 = np.concatenate((test_data_cat3, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)
            except NameError:
                test_data_cat3 = open_file[randperm[int(0.9*len(randperm)):]]

        elif 'Cat4' in l.split('/'):
            try:
                test_data_cat4 = np.concatenate((test_data_cat4, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)
            except NameError:
                test_data_cat4 = open_file[randperm[int(0.9*len(randperm)):]]

    return test_data, test_data_cat1, test_data_cat2, test_data_cat3, test_data_cat4

data_path = '/home/batool/FL/data_half_half_size'
save_path = '/home/batool/FL/baseline_code/all_test/'


selected_paths = detecting_related_file_paths(data_path,['Cat1','Cat2','Cat3','Cat4'],['0','1','2','3','4','5','6','7','8','9'])



test_data, test_data_cat1, test_data_cat2, test_data_cat3, test_data_cat4 = get_data(selected_paths,'rf','rf')
print(test_data.shape, test_data_cat1.shape, test_data_cat2.shape, test_data_cat3.shape, test_data_cat4.shape)
np.savez_compressed(save_path+'rf_all.npz', rf=test_data)
np.savez_compressed(save_path+'rf_cat1.npz', rf=test_data_cat1)
np.savez_compressed(save_path+'rf_cat2.npz', rf=test_data_cat2)
np.savez_compressed(save_path+'rf_cat3.npz', rf=test_data_cat3)
np.savez_compressed(save_path+'rf_cat4.npz', rf=test_data_cat4)

test_data, test_data_cat1, test_data_cat2, test_data_cat3, test_data_cat4 = get_data(selected_paths,'gps','gps')
print(test_data.shape, test_data_cat1.shape, test_data_cat2.shape, test_data_cat3.shape, test_data_cat4.shape)
np.savez_compressed(save_path+'gps_all.npz', gps=test_data)
np.savez_compressed(save_path+'gps_cat1.npz', gps=test_data_cat1)
np.savez_compressed(save_path+'gps_cat2.npz', gps=test_data_cat2)
np.savez_compressed(save_path+'gps_cat3.npz', gps=test_data_cat3)
np.savez_compressed(save_path+'gps_cat4.npz', gps=test_data_cat4)


test_data, test_data_cat1, test_data_cat2, test_data_cat3, test_data_cat4 = get_data(selected_paths,'image','img')
print(test_data.shape, test_data_cat1.shape, test_data_cat2.shape, test_data_cat3.shape, test_data_cat4.shape)
np.savez_compressed(save_path+'image_all.npz', img=test_data)
np.savez_compressed(save_path+'image_cat1.npz', img=test_data_cat1)
np.savez_compressed(save_path+'image_cat2.npz', img=test_data_cat2)
np.savez_compressed(save_path+'image_cat3.npz', img=test_data_cat3)
np.savez_compressed(save_path+'image_cat4.npz', img=test_data_cat4)


test_data, test_data_cat1, test_data_cat2, test_data_cat3, test_data_cat4 = get_data(selected_paths,'lidar','lidar')
print(test_data.shape, test_data_cat1.shape, test_data_cat2.shape, test_data_cat3.shape, test_data_cat4.shape)
np.savez_compressed(save_path+'lidar_all.npz', lidar=test_data)
np.savez_compressed(save_path+'lidar_cat1.npz', lidar=test_data_cat1)
np.savez_compressed(save_path+'lidar_cat2.npz', lidar=test_data_cat2)
np.savez_compressed(save_path+'lidar_cat3.npz', lidar=test_data_cat3)
np.savez_compressed(save_path+'lidar_cat4.npz', lidar=test_data_cat4)
