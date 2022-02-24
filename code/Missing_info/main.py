########################################################
#Project name: FLASH Infocom 2022
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


def get_data(data_paths,modality,key,test_on_all,path_test_all):   # per cat for now, need to add per epside for FL part
    first = True
    for l in tqdm(data_paths):
        randperm = np.load(l+'/ranperm.npy')
        if first == True:
            open_file = open_npz(l+'/'+modality+'.npz',key)
            train_data = open_file[randperm[:int(0.8*len(randperm))]]
            validation_data = open_file[randperm[int(0.8*len(randperm)):int(0.9*len(randperm))]]
            test_data = open_file[randperm[int(0.9*len(randperm)):]]
            first = False
        else:
            open_file = open_npz(l+'/'+modality+'.npz',key)
            train_data = np.concatenate((train_data, open_file[randperm[:int(0.8*len(randperm))]]),axis = 0)
            validation_data = np.concatenate((validation_data, open_file[randperm[int(0.8*len(randperm)):int(0.9*len(randperm))]]),axis = 0)
            test_data = np.concatenate((test_data, open_file[randperm[int(0.9*len(randperm)):]]),axis = 0)


        ####PER CAT
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

    if test_on_all:
        test_data = open_npz(path_test_all+'/'+modality+'_'+'all.npz',key)
        test_data_cat1 = open_npz(path_test_all+'/'+modality+'_'+'cat1.npz',key)
        test_data_cat2 = open_npz(path_test_all+'/'+modality+'_'+'cat2.npz',key)
        test_data_cat3 = open_npz(path_test_all+'/'+modality+'_'+'cat3.npz',key)
        test_data_cat4 = open_npz(path_test_all+'/'+modality+'_'+'cat4.npz',key)

    print('categories shapes',test_data_cat1.shape,test_data_cat2.shape,test_data_cat3.shape,test_data_cat4.shape)
    print('tr/val/te',train_data.shape,validation_data.shape,test_data.shape)
    return train_data,validation_data,test_data, test_data_cat1, test_data_cat2, test_data_cat3, test_data_cat4


def get_past_data(data_paths,samples_back,probabilty):
    for l in tqdm(data_paths):
        randperm = np.load(l+'/ranperm.npy')
        open_file_rf = open_npz(l+'/'+'rf'+'.npz','rf')
        open_file_gps = open_npz(l+'/'+'gps'+'.npz','gps')
        open_file_img = open_npz(l+'/'+'image'+'.npz','img')
        open_file_lidar = open_npz(l+'/'+'lidar'+'.npz','lidar')

        try:
            test_data_samples_rf = np.concatenate((test_data_samples_rf, open_file_rf[randperm[int(0.9*len(randperm)):]]),axis = 0)
            test_data_samples_gps_back = np.concatenate((test_data_samples_gps_back, open_file_gps[randperm[int(0.9*len(randperm)):]]),axis = 0)
            test_data_samples_img_back = np.concatenate((test_data_samples_img_back, open_file_img[randperm[int(0.9*len(randperm)):]-samples_back]),axis = 0)
            test_data_samples_lidar_back = np.concatenate((test_data_samples_lidar_back, open_file_lidar[randperm[int(0.9*len(randperm)):]-samples_back]),axis = 0)

            test_data_samples_gps = np.concatenate((test_data_samples_gps, open_file_gps[randperm[int(0.9*len(randperm)):]]),axis = 0)
            test_data_samples_img = np.concatenate((test_data_samples_img, open_file_img[randperm[int(0.9*len(randperm)):]]),axis = 0)
            test_data_samples_lidar = np.concatenate((test_data_samples_lidar, open_file_lidar[randperm[int(0.9*len(randperm)):]]),axis = 0)

        except:
            test_data_samples_rf = open_file_rf[randperm[int(0.9*len(randperm)):]]
            test_data_samples_gps_back = open_file_gps[randperm[int(0.9*len(randperm)):]]
            test_data_samples_img_back = open_file_img[randperm[int(0.9*len(randperm)):]-samples_back]
            test_data_samples_lidar_back = open_file_lidar[randperm[int(0.9*len(randperm)):]-samples_back]

            test_data_samples_gps = open_file_gps[randperm[int(0.9*len(randperm)):]]
            test_data_samples_img = open_file_img[randperm[int(0.9*len(randperm)):]]
            test_data_samples_lidar = open_file_lidar[randperm[int(0.9*len(randperm)):]]

        print('check shapes',test_data_samples_rf.shape, test_data_samples_gps_back.shape, test_data_samples_img_back.shape, test_data_samples_lidar_back.shape)
        print('check shapes',test_data_samples_rf.shape, test_data_samples_gps.shape, test_data_samples_img.shape, test_data_samples_lidar.shape)


    #################random selection
    random_generation = np.random.rand(len(test_data_samples_rf))
    gps_test = np.empty([len(test_data_samples_rf), 2], dtype=float)
    Image_test = np.empty([len(test_data_samples_rf),90,160,3], dtype=float)
    Lidar_test = np.empty([len(test_data_samples_rf), 20, 20, 20], dtype=int)
    for c in range(len(random_generation)):
        if random_generation[c]<probabilty:
            gps_test[c] = test_data_samples_gps_back[c]
            Image_test[c] = test_data_samples_img_back[c]
            Lidar_test[c] = test_data_samples_lidar_back[c]

        else:
            gps_test[c] = test_data_samples_gps[c]
            Image_test[c] = test_data_samples_img[c]
            Lidar_test[c] = test_data_samples_lidar[c]


    test_data_samples_back_rf, _ = custom_label(test_data_samples_rf,'one_hot')

    gps_test = gps_test/ 9747
    gps_test = gps_test.reshape((gps_test.shape[0], gps_test.shape[1], 1))

    Image_test = Image_test/ 255

    return test_data_samples_rf, gps_test, Image_test, Lidar_test





parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--id_gpu', default=1, type=int, help='which gpu to use.')
parser.add_argument('--data_folder', help='Location of the data directory', type=str)
parser.add_argument('--input', nargs='*', default=['coord'],choices = ['img', 'coord', 'lidar'],
help='Which data to use as input. Select from: img, lidar or coord.')

parser.add_argument('--epochs', default=10, type = int, help='Specify the epochs to train')
parser.add_argument('--lr', default=0.001, type=float,help='learning rate for Adam optimizer',)
parser.add_argument('--bs',default=32, type=int,help='Batch size')
parser.add_argument('--shuffle', help='shuffle or not', type=str2bool, default =True)

parser.add_argument('--strategy', type=str ,default='one_hot', help='labeling strategy to use',choices=['baseline','one_hot','reg'])
parser.add_argument('--restore_models', type=str2bool, help='Load single modality trained weights', default=False)
parser.add_argument('--model_folder', help='Location of the trained models folder', type=str,default = '/home/batool/FL/baseline_code/model_folder/')
parser.add_argument('--image_feature_to_use', type=str ,default='raw', help='feature images to use',choices=['raw','custom'])

parser.add_argument('--experiment_catergories', nargs='*' ,default=['Cat1','Cat2','Cat3','Cat4'], help='categories included',choices=['Cat1','Cat2','Cat3','Cat4'])
parser.add_argument('--experiment_epiosdes', nargs='*' ,default=['0','1','2','3','4','5','6','7','8','9'], help='episodes included',choices=['0','1','2','3','4','5','6','7','8','9'])


parser.add_argument('--test_all', help='test on all data', type=str2bool, default =True)
parser.add_argument('--test_all_path', help='Location of all test', type=str,default = '/home/batool/FL/baseline_code/all_test/')


args = parser.parse_args()
print('Argumen parser inputs', args)

if args.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

check_and_create(args.model_folder)


print('******************Detecting related file paths*************************')
selected_paths = detecting_related_file_paths(args.data_folder,args.experiment_catergories,args.experiment_epiosdes)
###############################################################################
# Outputs ##needs to be changed
###############################################################################
# print('******************Getting RF data*************************')
# RF_train, RF_val, RF_test,RF_c1,RF_c2,RF_c3,RF_c4 = get_data(selected_paths,'rf','rf',args.test_all,args.test_all_path)
# print('RF data shapes on same client',RF_train.shape,RF_val.shape,RF_test.shape)

# y_train,num_classes = custom_label(RF_train,args.strategy)
# y_validation, _ = custom_label(RF_val,args.strategy)
# y_test, _ = custom_label(RF_test,args.strategy)
# y_c1, _ = custom_label(RF_c1,args.strategy)
# y_c2, _ = custom_label(RF_c2,args.strategy)
# y_c3, _ = custom_label(RF_c3,args.strategy)
# y_c4, _ = custom_label(RF_c4,args.strategy)

###############################################################################
# Inputs ##needs to be changed
###############################################################################

# if 'coord' in args.input:
#     print('******************Getting Gps data*************************')
#     X_coord_train, X_coord_validation, X_coord_test,gps_c1,gps_c2,gps_c3,gps_c4 = get_data(selected_paths,'gps','gps',args.test_all,args.test_all_path)
#     print('GPS data shapes',X_coord_train.shape,X_coord_validation.shape,X_coord_test.shape)
#     coord_train_input_shape = X_coord_train.shape

#     ### normalized
#     print('max over dataset b', np.max(abs(X_coord_train)),np.max(abs(X_coord_validation)),np.max(abs(X_coord_test)))
#     X_coord_train = X_coord_train / 9747
#     X_coord_validation = X_coord_validation / 9747
#     X_coord_test = X_coord_test / 9747
#     gps_c1 = gps_c1/ 9747
#     gps_c2 = gps_c2/ 9747
#     gps_c3 = gps_c3/ 9747
#     gps_c4 = gps_c4/ 9747
#     ## For convolutional input
#     X_coord_train = X_coord_train.reshape((X_coord_train.shape[0], X_coord_train.shape[1], 1))
#     X_coord_validation = X_coord_validation.reshape((X_coord_validation.shape[0], X_coord_validation.shape[1], 1))
#     X_coord_test = X_coord_test.reshape((X_coord_test.shape[0], X_coord_test.shape[1], 1))
#     print('shapes after re-shaping',X_coord_train.shape)

#     gps_c1 = gps_c1.reshape((gps_c1.shape[0], gps_c1.shape[1], 1))
#     gps_c2 = gps_c2.reshape((gps_c2.shape[0], gps_c2.shape[1], 1))
#     gps_c3 = gps_c3.reshape((gps_c3.shape[0], gps_c3.shape[1], 1))
#     gps_c4 = gps_c4.reshape((gps_c4.shape[0], gps_c4.shape[1], 1))


# if 'img' in args.input:
#     print('******************Getting image data*************************')
#     X_img_train, X_img_validation, X_img_test,img_c1,img_c2,img_c3,img_c4 = get_data(selected_paths,'image','img',args.test_all,args.test_all_path)
#     print('max over dataset b', np.max(X_img_train),np.max(X_img_validation),np.max(X_img_test))
#     print('image data shapes',X_img_train.shape,X_img_validation.shape,X_img_test.shape)
#     ###normalize images
#     X_img_train = X_img_train / 255
#     X_img_validation = X_img_validation / 255
#     X_img_test = X_img_test/255
#     img_c1 = img_c1/ 255
#     img_c2 = img_c2/ 255
#     img_c3 = img_c3/ 255
#     img_c4 = img_c4/ 255
#     img_train_input_shape = X_img_train.shape

# if 'lidar' in args.input:
#     print('******************Getting lidar data*************************')
#     X_lidar_train, X_lidar_validation, X_lidar_test,lid_c1,lid_c2,lid_c3,lid_c4 = get_data(selected_paths,'lidar','lidar',args.test_all,args.test_all_path)
#     print('lidar data shapes',X_lidar_train.shape,X_lidar_validation.shape,X_lidar_test.shape)
#     lidar_train_input_shape = X_lidar_train.shape
coord_train_input_shape = (2715, 2, 1)
img_train_input_shape = (2715, 90, 160, 3)
lidar_train_input_shape = (2715, 20, 20, 20)
num_classes = 64
print('******************Succesfully generated the data*************************')
##############################################################################
# Model configuration
##############################################################################
print('******************Configuring the models*************************')
#multimodal
multimodal = False if len(args.input) == 1 else len(args.input)
fusion = False if len(args.input) == 1 else True

modelHand = ModelHandler()
opt = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

if 'coord' in args.input:
    if args.restore_models:
        coord_model = load_model_structure(args.model_folder+'coord_model.json')
        coord_model.load_weights(args.model_folder + 'best_weights.coord.h5', by_name=True)
    else:
        coord_model = modelHand.createArchitecture('coord_mlp',num_classes,coord_train_input_shape[1],'complete',args.strategy, fusion)
        if not os.path.exists(args.model_folder+'coord_model.json'):
            add_model('coord',coord_model,args.model_folder)


if 'img' in args.input:
    if args.image_feature_to_use == 'raw':
        model_type = 'raw_image'
    elif args.image_feature_to_use == 'custom':
        model_type = 'custom_image'

    if args.restore_models:
        img_model = load_model_structure(args.model_folder+'image_'+args.image_feature_to_use+'_model'+'.json')
        img_model.load_weights(args.model_folder + 'best_weights.img_'+args.image_feature_to_use+'.h5', by_name=True)
        # img_model.trainable = False
    else:
        img_model = modelHand.createArchitecture(model_type,num_classes,[img_train_input_shape[1],img_train_input_shape[2],3],'complete',args.strategy,fusion)
        if not os.path.exists(args.model_folder+'image_'+args.image_feature_to_use+'_model'+'.json'):
            add_model('image_'+args.image_feature_to_use,img_model,args.model_folder)

if 'lidar' in args.input:
    if args.restore_models:
        lidar_model = load_model_structure(args.model_folder+'lidar_model.json')
        lidar_model.load_weights(args.model_folder + 'best_weights.lidar.h5', by_name=True)
    else:
        lidar_model = modelHand.createArchitecture('lidar_marcus',num_classes,[lidar_train_input_shape[1],lidar_train_input_shape[2],lidar_train_input_shape[3]],'complete',args.strategy, fusion)
        if not os.path.exists(args.model_folder+'lidar_model.json'):
            add_model('lidar',lidar_model,args.model_folder)

###############################################################################
# Fusion: Coordinate+LIDAR
###############################################################################
if multimodal == 2:
    if 'coord' in args.input and 'lidar' in args.input:
        x_train = [X_lidar_train, X_coord_train]
        x_validation = [X_lidar_validation, X_coord_validation]
        x_test = [X_lidar_test, X_coord_test]

        combined_model = concatenate([lidar_model.output, coord_model.output],name = 'cont_fusion_coord_lidar')
        reg_val=0.001
        z = Dense(600,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(combined_model)
        z = BatchNormalization()(z)
        z = Dense(600,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(z)
        z = BatchNormalization()(z)
        z = Dense(500,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(z)
        z = BatchNormalization()(z)
        z = Dense(num_classes, activation="softmax",name = 'dense3_fusion_coord_lid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)

        model = Model(inputs=[lidar_model.input, coord_model.input], outputs=z)
        add_model('coord_lidar',model,args.model_folder)    # for fusion
        model.compile(loss=categorical_crossentropy,
                      optimizer=opt,
                      metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy, top_10_accuracy, top_25_accuracy,
                                        top_50_accuracy])
        model.summary()
        hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs,batch_size=args.bs, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.coord_lidar.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=2,mode='auto')])

        print(hist.history.keys())
        print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
                ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])

        print('***************Testing the model************')
        model.load_weights(args.model_folder+'best_weights.coord_lidar.h5', by_name=True)
        scores = model.evaluate(x_test, y_test)
        print(model.metrics_names, scores)

        #####For recall and precison
        y_pred = model.predict(x_test)
        y_pred_bool = np.argmax(y_pred, axis=1)
        y_true_bool = np.argmax(y_test, axis=1)
        print(classification_report(y_true_bool, y_pred_bool))
        print('avegare presion,recall,f1',precision_recall_fscore(y_true_bool, y_pred_bool,average='weighted'))

###############################################################################
# Fusion: Coordinate+Image
###############################################################################
    elif 'coord' in args.input and 'img' in args.input:
        x_train = [X_img_train, X_coord_train]
        x_validation = [X_img_validation, X_coord_validation]
        x_test = [X_img_test, X_coord_test]
        combined_model = concatenate([img_model.output, coord_model.output],name = 'cont_fusion_coord_img')
        z = Reshape((2, 64))(combined_model)
        z = Permute((2, 1), input_shape=(2, 64))(z)
        z = Conv1D(30, kernel_size=7, strides=1, activation="relu",name = 'conv1_fusion_coord_img')(z)  # KERNEL SIZE CHANGED FROM 1 TO 2
        z = Conv1D(30, kernel_size=5, strides=1, activation="relu",name = 'conv2_fusion_coord_img')(z)  # KERNEL SIZE CHANGED FROM 1 TO 2
        z = MaxPooling1D(name='fusion_coord_img_maxpool1')(z)

        z = Conv1D(30, kernel_size=7, strides=1, activation="relu",name = 'conv3_fusion_coord_img')(z)  # KERNEL SIZE CHANGED FROM 1 TO 2
        z = Conv1D(30, kernel_size=5, strides=1, activation="relu",name = 'conv4_fusion_coord_img')(z)  # KERNEL SIZE CHANGED FROM 1 TO 2
        z = MaxPooling1D(name='fusion_coord_img_maxpool2')(z)

        z = Flatten(name = 'flat_fusion_coord_img')(z)
        z = Dense(num_classes * 3, activation="relu", use_bias=True,name = 'dense1_fusion_coord_img')(z)
        z = Dropout(0.25,name = 'drop1_fusion_coord_lidar')(z)
        z = Dense(num_classes * 2, activation="relu",name = 'dense2_fusion_coord_img', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)
        z = Dropout(0.25,name = 'drop2_fusion_coord_img')(z)
        z = Dense(num_classes, activation="softmax",name = 'dense3_fusion_coord_img', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)

        model = Model(inputs=[img_model.input, coord_model.input], outputs=z)
        add_model('coord_img_raw',model,args.model_folder)   #saving model for fusion

        model.compile(loss=categorical_crossentropy,
                      optimizer=opt,
                      metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy,top_10_accuracy, top_25_accuracy,
                                        top_50_accuracy])
        model.summary()

        hist = model.fit(x_train, y_train,validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.coord_img_'+args.image_feature_to_use+'.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=2, mode='auto')])

        print(hist.history.keys())
        print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
                ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])

        print('***************Testing the model************')
        model.load_weights(args.model_folder+'best_weights.coord_img_'+args.image_feature_to_use+'.h5', by_name=True)
        scores = model.evaluate(x_test, y_test)
        print(model.metrics_names, scores)

        #####For recall and precison
        y_pred = model.predict(x_test)
        y_pred_bool = np.argmax(y_pred, axis=1)
        y_true_bool = np.argmax(y_test, axis=1)
        print(classification_report(y_true_bool, y_pred_bool))
        print('avegare presion,recall,f1',precision_recall_fscore(y_true_bool, y_pred_bool,average='weighted'))
##############################################################################
# Fusion: Image+LIDAR
###############################################################################
    else: # img+lidar
        x_train = [X_lidar_train,X_img_train]
        x_validation = [X_lidar_validation, X_img_validation]
        x_test = [X_lidar_test, X_img_test]

        combined_model = concatenate([lidar_model.output, img_model.output],name = 'cont_fusion_img_lidar')
        reg_val=0.001
        z = Dense(600,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(combined_model)
        z = BatchNormalization()(z)
        z = Dense(600,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(z)
        z = BatchNormalization()(z)
        z = Dense(500,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(z)
        z = BatchNormalization()(z)
        z = Dense(num_classes, activation="softmax",name = 'dense3_fusion_coord_lid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)

        model = Model(inputs=[lidar_model.input, img_model.input], outputs=z)
        add_model('img_lidar',model,args.model_folder)

        model.compile(loss=categorical_crossentropy,optimizer=opt,
                      metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy ,top_10_accuracy,top_25_accuracy,top_50_accuracy])
        model.summary()

        hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs,batch_size=args.bs, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.img_lidar_'+args.image_feature_to_use+'.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=2,mode='auto')])

        print(hist.history.keys())
        print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
                ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])


        print('***************Testing the model************')
        model.load_weights(args.model_folder+'best_weights.img_lidar'+args.image_feature_to_use+'.h5', by_name=True)
        scores = model.evaluate(x_test, y_test)
        print(model.metrics_names, scores)

        #####For recall and precison
        y_pred = model.predict(x_test)
        y_pred_bool = np.argmax(y_pred, axis=1)
        y_true_bool = np.argmax(y_test, axis=1)
        print(classification_report(y_true_bool, y_pred_bool))
        print('avegare presion,recall,f1',precision_recall_fscore(y_true_bool, y_pred_bool,average='weighted'))

##############################################################################
# Fusion: Coordinate+Image+LIDAR
###############################################################################
elif multimodal == 3:
    combined_model = concatenate([lidar_model.output, img_model.output, coord_model.output])
    reg_val=0.001
    z = Dense(1024,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(combined_model)
    z = BatchNormalization()(z)
    z = Dense(512,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(z)
    z = BatchNormalization()(z)
    # z = Dense(500,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(z)
    # z = BatchNormalization()(z)
    z = Dense(256,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(z)
    z = BatchNormalization()(z)
    z = Dense(128,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(z)
    z = BatchNormalization()(z)
    z = Dense(num_classes, activation="softmax",name = 'dense3_fusion_coord_lid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)

    model = Model(inputs=[lidar_model.input, img_model.input, coord_model.input], outputs=z)
    add_model('coord_img_raw_lidar',model,args.model_folder)   #add fusion model
    model.compile(loss=categorical_crossentropy,
                  optimizer=opt,
                  metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy,top_10_accuracy,top_25_accuracy,top_50_accuracy])
    model.summary()

    # hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.coord_img_lidar_'+args.image_feature_to_use+'.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=2,mode='auto')])

    # print(hist.history.keys())
    # print('loss',hist.history['loss'],'val_loss',hist.history['val_loss'],'categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
    #             ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])


    # print('***************Testing the model************')
    # model.load_weights(args.model_folder+'best_weights.coord_img_lidar_'+args.image_feature_to_use+'.h5', by_name=True)
    # scores = model.evaluate(x_test, y_test)
    # print(model.metrics_names, scores)

    # #####For recall and precison
    # y_pred = model.predict(x_test)
    # y_pred_bool = np.argmax(y_pred, axis=1)
    # y_true_bool = np.argmax(y_test, axis=1)
    # print(classification_report(y_true_bool, y_pred_bool))
    # print('avegare presion,recall,f1',precision_recall_fscore(y_true_bool, y_pred_bool,average='weighted'))
    # ##### Per K accuracy
    # # print('per k accuracy all ',over_k(y_test,y_pred))
    # print('***************Testing per category************')
    # x_c1 = [lid_c1, img_c1, gps_c1]
    # x_c2 = [lid_c2, img_c2, gps_c2]
    # x_c3 = [lid_c3, img_c3, gps_c3]
    # x_c4 = [lid_c4, img_c4, gps_c4]
    # scores_cat1 = model.evaluate(x_c1, y_c1)
    # print('scores_cat1',scores_cat1,'PRF',precison_recall_F1(model,x_c1,y_c1))
    # scores_cat2 = model.evaluate(x_c2, y_c2)
    # print('scores_cat2',scores_cat2,'PRF',precison_recall_F1(model,x_c2,y_c2))
    # scores_cat3 = model.evaluate(x_c3, y_c3)
    # print('scores_cat3',scores_cat3,'PRF',precison_recall_F1(model,x_c3,y_c3))
    # scores_cat4 = model.evaluate(x_c4, y_c4)
    # print('scores_cat4',scores_cat4,'PRF',precison_recall_F1(model,x_c4,y_c4))



    print('***************loading past samples************')
    samples_back = 10
    prob =0.5
    test_data_samples_rf, gps_test, Image_test, Lidar_test = get_past_data(selected_paths,samples_back,prob)
    print('check shapes',test_data_samples_rf.shape, gps_test.shape, Image_test.shape, Lidar_test.shape)
    print('***************Testing the model************')
    model.load_weights(args.model_folder+'best_weights.coord_img_lidar_'+args.image_feature_to_use+'.h5', by_name=True)
    x_test = [Lidar_test, Image_test, gps_test]
    y_test = test_data_samples_rf
    scores = model.evaluate(x_test, y_test)
    print(model.metrics_names, scores)

    #####For recall and precison
    y_pred = model.predict(x_test)
    y_pred_bool = np.argmax(y_pred, axis=1)
    y_true_bool = np.argmax(y_test, axis=1)
    print(classification_report(y_true_bool, y_pred_bool))
    print('avegare presion,recall,f1',precision_recall_fscore(y_true_bool, y_pred_bool,average='weighted'))
    print('samples_back',samples_back)

###############################################################################
# Single modalities
###############################################################################
else:
    if 'coord' in args.input:
        if args.strategy == 'reg':
            model = coord_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_1_accuracy,top_2_accuracy,top_10_accuracy,top_50_accuracy,R2_metric])
            model.summary()

            hist = model.fit(X_coord_train,y_train,validation_data=(X_coord_validation, y_validation),
            epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
            print('losses in train:', hist.history['loss'])

            print('*****************Testing***********************')
            scores = model.evaluate(X_coord_test, y_test)
            pprint('scores while testing:', model.metrics_names,scores)


        if args.strategy == 'one_hot':
            model = coord_model
            model.compile(loss=categorical_crossentropy,
                                optimizer=opt,
                                metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy,top_10_accuracy, top_25_accuracy,
                                        top_50_accuracy])
            model.summary()


            call_backs = []
            # hist = model.fit(X_coord_train,y_train, validation_data=(X_coord_validation, y_validation),
            # epochs=args.epochs,batch_size=args.bs, shuffle=args.shuffle, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.coord.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=2, mode='auto')])

            # print(hist.history.keys())
            # print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
            #         ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])

            print('***************Testing model************')
            model.load_weights(args.model_folder + 'best_weights.coord.h5', by_name=True)   ## Restoring best weight for testing
            scores = model.evaluate(X_coord_test, y_test)
            print(model.metrics_names,scores)

            #####For recall and precison
            y_pred = model.predict(X_coord_test)
            y_pred_bool = np.argmax(y_pred, axis=1)
            y_true_bool = np.argmax(y_test, axis=1)
            print(classification_report(y_true_bool, y_pred_bool))
            print('avegare presion,recall,f1',precision_recall_fscore(y_true_bool, y_pred_bool,average='weighted'))
            ####Per k accuracy
            # print('per k accuracy',over_k(y_test,y_pred))
            print('***************Testing per category************')
            scores_cat1 = model.evaluate(gps_c1, y_c1)
            print('scores_cat1',scores_cat1,'PRF',precison_recall_F1(model,gps_c1,y_c1))
            scores_cat2 = model.evaluate(gps_c2, y_c2)
            print('scores_cat2',scores_cat2,'PRF',precison_recall_F1(model,gps_c2,y_c2))
            scores_cat3 = model.evaluate(gps_c3, y_c3)
            print('scores_cat3',scores_cat3,'PRF',precison_recall_F1(model,gps_c3,y_c3))
            scores_cat4 = model.evaluate(gps_c4, y_c4)
            print('scores_cat4',scores_cat4,'PRF',precison_recall_F1(model,gps_c4,y_c4))

    elif 'img' in args.input:

        if args.strategy == 'reg':
            model = img_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_1_accuracy,top_2_accuracy,top_10_accuracy,top_50_accuracy,R2_metric])
            model.summary()

            hist = model.fit(X_img_train,y_train, validation_data=(X_coord_validation, y_validation),
            epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
            print('losses in train:', hist.history['loss'])

            print('*****************Testing***********************')
            scores = model.evaluate(X_img_test, y_test)
            print('scores while testing:', model.metrics_names,scores)


        if args.strategy == 'one_hot':
            model = img_model
            model.compile(loss=categorical_crossentropy,
                                optimizer=opt,
                                metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy, top_10_accuracy, top_25_accuracy,
                                        top_50_accuracy])
            model.summary()
            # hist = model.fit(X_img_train,y_train, validation_data=(X_img_validation, y_validation),
            # epochs=args.epochs,batch_size=args.bs, shuffle=args.shuffle, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.img_'+args.image_feature_to_use+'.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=2, mode='auto')])


            # print(hist.history.keys())
            # print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
            #         ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])

            print('***************Testing model************')
            model.load_weights(args.model_folder + 'best_weights.img_'+args.image_feature_to_use+'.h5', by_name=True)
            scores = model.evaluate(X_img_test, y_test)
            print(model.metrics_names,scores)

            #####For recall and precison
            y_pred = model.predict(X_img_test)
            y_pred_bool = np.argmax(y_pred, axis=1)
            y_true_bool = np.argmax(y_test, axis=1)
            print(classification_report(y_true_bool, y_pred_bool))
            print('avegare presion,recall,f1',precision_recall_fscore(y_true_bool, y_pred_bool,average='weighted'))
            #####Per K accuracy
            # print('per k accuracy',over_k(y_test,y_pred))
            print('***************Testing per category************')
            scores_cat1 = model.evaluate(img_c1, y_c1)
            print('scores_cat1',scores_cat1,'PRF',precison_recall_F1(model,img_c1,y_c1))
            scores_cat2 = model.evaluate(img_c2, y_c2)
            print('scores_cat2',scores_cat2,'PRF',precison_recall_F1(model,img_c2,y_c2))
            scores_cat3 = model.evaluate(img_c3, y_c3)
            print('scores_cat3',scores_cat3,'PRF',precison_recall_F1(model,img_c3,y_c3))
            scores_cat4 = model.evaluate(img_c4, y_c4)
            print('scores_cat4',scores_cat4,'PRF',precison_recall_F1(model,img_c4,y_c4))
    else: #LIDAR
        if args.strategy == 'reg':
            model = lidar_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_1_accuracy,top_2_accuracy,top_10_accuracy,top_50_accuracy,R2_metric])
            model.summary()

            hist = model.fit(X_lidar_train,y_train,validation_data=(X_lidar_validation, y_validation),
            epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
            print('*****************Testing***********************')
            scores = model.evaluate(X_lidar_test, y_test)
            print('scores while testing:', model.metrics_names,scores)

        if args.strategy == 'one_hot':
            print('All shapes',X_lidar_train.shape,y_train.shape,X_lidar_validation.shape,y_validation.shape,X_lidar_test.shape,y_test.shape)
            model = lidar_model
            model.compile(loss=categorical_crossentropy,
                          optimizer=opt,
                          metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy,top_10_accuracy, top_25_accuracy,
                                        top_50_accuracy])
            model.summary()
            # hist = model.fit(X_lidar_train,y_train, validation_data=(X_lidar_validation, y_validation),epochs=args.epochs,batch_size=args.bs, shuffle=args.shuffle,callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.lidar.h5', monitor='val_loss', verbose=2, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=2,mode='auto')])

            # print(hist.history.keys())
            # print('val_loss',hist.history['val_loss'],'categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
            #         ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])

            print('***************Testing model************')
            model.load_weights(args.model_folder + 'best_weights.lidar.h5', by_name=True)   # to be added
            scores = model.evaluate(X_lidar_test, y_test)
            print(model.metrics_names,scores)

            #####For recall and precison
            y_pred = model.predict(X_lidar_test)
            y_pred_bool = np.argmax(y_pred, axis=1)
            y_true_bool = np.argmax(y_test, axis=1)
            print(classification_report(y_true_bool, y_pred_bool))
            print('average presion,recall,f1',precision_recall_fscore(y_true_bool, y_pred_bool,average='weighted'))
            ###Per k accuracy
            # print('per k accuracy',over_k(y_test,y_pred))
            print('***************Testing per category************')
            scores_cat1 = model.evaluate(lid_c1, y_c1)
            print('scores_cat1',scores_cat1,'PRF',precison_recall_F1(model,lid_c1,y_c1))
            scores_cat2 = model.evaluate(lid_c2, y_c2)
            print('scores_cat2',scores_cat2,'PRF',precison_recall_F1(model,lid_c2,y_c2))
            scores_cat3 = model.evaluate(lid_c3, y_c3)
            print('scores_cat3',scores_cat3,'PRF',precison_recall_F1(model,lid_c3,y_c3))
            scores_cat4 = model.evaluate(lid_c4, y_c4)
            print('scores_cat4',scores_cat4,'PRF',precison_recall_F1(model,lid_c4,y_c4))
