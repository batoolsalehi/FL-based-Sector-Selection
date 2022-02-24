########################################################
#Project name: FLASH: Infocom 2020
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
from tensorflow.keras.layers import Dense,concatenate, Dropout, Conv1D, Flatten, Reshape, Activation,multiply,MaxPooling1D,Add,AveragePooling1D,Lambda,Permute,Input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adadelta,Adam, SGD, Nadam,Adamax, Adagrad
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from keras.regularizers import l2
from tensorflow import keras

import sklearn
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import normalize

from models import add_model,load_model_structure, ModelHandler
from custom_metrics import *
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as precision_recall_fscore
from tensorflow.keras import backend as K

############################
# Fix the seed
############################
seed = 0
os.environ['PYTHONHASHSEED']=str(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)
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
    selected = []
    for Cat in categories:   # specify categories as input
        for ep in episodes:
            selected = selected + [s for s in find_all_paths if Cat in s.split('/') and 'episode_'+str(ep) in s.split('/')]
    print('Getting {} data out of {}'.format(len(selected),len(find_all_paths)))

    return selected


def get_data(data_paths,modality,key):   # per cat for now, need to add per epside for FL part
    first = True
    for l in data_paths:
        randperm = np.load(l+'/ranperm.npy')
        if first == True:
            open_file = open_npz(l+'/'+modality+'.npz',key)
            train_data = open_file[randperm[:int(0.8*len(randperm))]]
            validation_data = open_file[randperm[int(0.8*len(randperm)):int(0.9*len(randperm))]]
            first = False
        else:
            open_file = open_npz(l+'/'+modality+'.npz',key)
            train_data = np.concatenate((train_data, open_file[randperm[:int(0.8*len(randperm))]]),axis = 0)
            validation_data = np.concatenate((validation_data, open_file[randperm[int(0.8*len(randperm)):int(0.9*len(randperm))]]),axis = 0)

    print('tr/val/te',train_data.shape,validation_data.shape)

    return train_data,validation_data




def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    print('scaled_weight_list',len(scaled_weight_list))
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)/len(scaled_weight_list)
        avg_grad.append(layer_mean)
    return avg_grad


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def select_branch(policy):
    print('policy',policy)
    update_coord_branch, update_img_branch,update_lidar_branch,update_fusion_branch = False,False,False,False
    random_number  = np.random.rand(1)
    if random_number< policy[0]:
        update_coord_branch = True
    elif policy[0]<random_number< policy[0]+policy[1]:
        update_img_branch = True
    elif policy[0]+policy[1]<random_number< policy[0]+policy[1]+policy[2]:
        update_lidar_branch = True
    elif policy[0]+policy[1]+policy[2]<random_number< policy[0]+policy[1]+policy[2]+policy[3]:
        update_fusion_branch = True
    return update_coord_branch, update_img_branch,update_lidar_branch,update_fusion_branch

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
parser.add_argument('--model_folder', help='Location of the trained models folder', type=str,default = '/home/batool/FL/federated/models_all_test/')
parser.add_argument('--image_feature_to_use', type=str ,default='raw', help='feature images to use',choices=['raw','custom'])

parser.add_argument('--experiment_catergories', nargs='*' ,default=['Cat1','Cat2','Cat3','Cat4'], help='categories included',choices=['Cat1','Cat2','Cat3','Cat4'])
parser.add_argument('--experiment_epiosdes', nargs='*' ,default=['0','1','2','3','4','5','6','7','8','9'], help='episodes included',choices=['0','1','2','3','4','5','6','7','8','9'])

parser.add_argument('--test_all', help='test on all data', type=str2bool, default =True)
parser.add_argument('--test_all_path', help='Location of all test', type=str,default = '/home/batool/FL/baseline_code/all_test/')
parser.add_argument('--policy',default=[0.1,0.1,0.4,0.4], help='policy in the downlink')
parser.add_argument('--aggregatin_on_all', help='aggreagte entire mode', type=str2bool, default =False)

parser.add_argument('--latest_step',default=0, type=int,help='for resuming')




args = parser.parse_args()
print('Argumen parser inputs', args)

if args.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

check_and_create(args.model_folder)
print('start_time:',time())

print('**********genrating train/validation data************')
clients_data = {}
for clients in tqdm(args.experiment_epiosdes):
    selected_paths = detecting_related_file_paths(args.data_folder,args.experiment_catergories,clients)
    ########################RF
    RF_train, RF_val = get_data(selected_paths,'rf','rf')
    y_train,num_classes = custom_label(RF_train,args.strategy)
    y_validation, _ = custom_label(RF_val,args.strategy)
    ################GPS
    X_coord_train, X_coord_validation = get_data(selected_paths,'gps','gps')
    X_coord_train = X_coord_train / 9747
    X_coord_validation = X_coord_validation / 9747
    ## For convolutional input
    X_coord_train = X_coord_train.reshape((X_coord_train.shape[0], X_coord_train.shape[1], 1))
    X_coord_validation = X_coord_validation.reshape((X_coord_validation.shape[0], X_coord_validation.shape[1], 1))

    ####Image
    X_img_train, X_img_validation = get_data(selected_paths,'image','img')
    X_img_train = X_img_train / 255
    X_img_validation = X_img_validation / 255

    ####Lidar
    X_lidar_train, X_lidar_validation = get_data(selected_paths,'lidar','lidar')

    clients_data[clients] = {'X_coord_train':X_coord_train, 'X_coord_validation':X_coord_validation,
                            'X_img_train':X_img_train, 'X_img_validation':X_img_validation,
                            'X_lidar_train':X_lidar_train, 'X_lidar_validation':X_lidar_validation,
                            'RF_train':RF_train,'RF_val': RF_val
                            }

print('****************Loading test set*****************')
RF_test = open_npz(args.test_all_path+'/'+'rf'+'_'+'all.npz','rf')
X_coord_test =  open_npz(args.test_all_path+'/'+'gps'+'_'+'all.npz','gps')
X_img_test =  open_npz(args.test_all_path+'/'+'image'+'_'+'all.npz','img')
X_lidar_test =  open_npz(args.test_all_path+'/'+'lidar'+'_'+'all.npz','lidar')

y_test, _ = custom_label(RF_test,args.strategy)
X_coord_test = X_coord_test / 9747
X_coord_test = X_coord_test.reshape((X_coord_test.shape[0], X_coord_test.shape[1], 1))
X_img_test = X_img_test/255

coord_train_input_shape = (2715, 2, 1)
img_train_input_shape = (2715, 90, 160, 3)
lidar_train_input_shape = (2715, 20, 20, 20)
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

coord_model.summary()
img_model.summary()
lidar_model.summary()

if multimodal == 3:
    reg_val=0.001
    fusion_input = Input(shape=(832,), name='fusion_input')
    z = Dense(1024,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val),name = 'first_fusion')(fusion_input)
    z = BatchNormalization(name = 'bn1_fusion')(z)
    z = Dense(512,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val), name='dense1_fusion')(z)
    z = BatchNormalization(name = 'bn2_fusion')(z)
    # z = Dense(500,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(z)
    # z = BatchNormalization()(z)
    z = Dense(256,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val), name='dense2_fusion')(z)
    z = BatchNormalization(name = 'bn3_fusion')(z)
    z = Dense(128,activation='relu',kernel_regularizer=l2(reg_val), bias_regularizer=l2(reg_val), name='dense3_fusion')(z)
    z = BatchNormalization(name = 'bn4_fusion')(z)
    z = Dense(num_classes, activation="softmax",name = 'fusion_output', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)
    fusion_model = Model(inputs=fusion_input, outputs=z)

fusion_model.summary()
########################################################
########################################################
combined_model = concatenate([lidar_model.output, img_model.output, coord_model.output])
for layer in fusion_model.layers[1:]:
    if layer.name == 'first_fusion':
        z = layer(combined_model)
    else:
        z = layer(z)

complete_model = Model(inputs=[lidar_model.input, img_model.input, coord_model.input], outputs=z)
complete_model.summary()


print('**********Models are configured***********')
print('**********Start federated learning************')


comms_round = 200
loss='categorical_crossentropy'
metrics = [metrics.categorical_accuracy,top_2_accuracy, top_5_accuracy,top_10_accuracy,top_25_accuracy,top_50_accuracy]
opt = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
global_model = complete_model


for comm_round in range(args.latest_step,comms_round):
    print('iteration_start_time:',time())
    # get the global model's weights - will serve as the initial weights for all local models
    if comm_round==0:
        latest_global_model = global_model
    elif comm_round>0:
        #set local model weight to the weight of the latest local model
        latest_global_model = tf.keras.models.load_model(args.model_folder+'global_model.h5',custom_objects={'top_2_accuracy':top_2_accuracy,'top_5_accuracy':top_5_accuracy,'top_10_accuracy':top_10_accuracy,'top_25_accuracy':top_25_accuracy,'top_50_accuracy':top_50_accuracy},compile=False)

    #initial list to collect local model weights after scalling
    local_weight_list = list()

    #loop through each client and create new local model
    for client in args.experiment_epiosdes:
        print('*************trainigclient'+client+'**************')
        local_model = complete_model
        local_model.compile(loss=loss,
                      optimizer=opt,
                      metrics=metrics)

        ###set local model weight to the weight of the global model
        if comm_round==0:
            local_model.set_weights(latest_global_model.get_weights())
        elif comm_round>0:
            #set local model weight to the weight of the latest local model
            local_model.load_weights(args.model_folder+'latest_local_model_client_'+client+'.h5')


        update_coord_branch, update_img_branch,update_lidar_branch,update_fusion_branch = select_branch(args.policy)
        print('selected branches:',update_coord_branch, update_img_branch,update_lidar_branch,update_fusion_branch)
        if args.aggregatin_on_all:
            update_coord_branch, update_img_branch,update_lidar_branch,update_fusion_branch = True,True,True,True


        local_model_layer = local_model.layers
        latest_global_model_layer = latest_global_model.layers

        for match_layers in range(len(local_model.layers)):

            if update_coord_branch and "coord" in local_model_layer[match_layers].name.split('_'):
                print('coord')
                print('matching layers',local_model_layer[match_layers].name,latest_global_model_layer[match_layers].name)
                local_model_layer[match_layers].set_weights(latest_global_model_layer[match_layers].get_weights())

            if update_img_branch and "img" in local_model_layer[match_layers].name.split('_'):
                print('image')
                print('matching layers',local_model_layer[match_layers].name,latest_global_model_layer[match_layers].name)
                local_model_layer[match_layers].set_weights(latest_global_model_layer[match_layers].get_weights())

            if update_lidar_branch and "lidar" in local_model_layer[match_layers].name.split('_'):
                print('lidar')
                print('matching layers',local_model_layer[match_layers].name,latest_global_model_layer[match_layers].name)
                local_model_layer[match_layers].set_weights(latest_global_model_layer[match_layers].get_weights())

            if update_fusion_branch and "fusion" in local_model_layer[match_layers].name.split('_'):  # need to do layer by layer
                print('integration')
                print('matching layers',local_model_layer[match_layers].name,latest_global_model_layer[match_layers].name)
                local_model_layer[match_layers].set_weights(latest_global_model_layer[match_layers].get_weights())


        #fit local model with client's data
        x_train = [clients_data[client]['X_lidar_train'],clients_data[client]['X_img_train'],clients_data[client]['X_coord_train']]
        x_validation = [clients_data[client]['X_lidar_validation'],clients_data[client]['X_img_validation'],clients_data[client]['X_coord_validation']]
        y_train = clients_data[client]['RF_train']
        y_validation = clients_data[client]['RF_val']

        local_model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs, verbose=1,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2,mode='auto',restore_best_weights=True)])
        local_model.save(args.model_folder+'latest_local_model_client_'+client+'.h5')
        local_weight_list.append(local_model.get_weights())

    print('*********averaging weights*************')
    average_weights = sum_scaled_weights(local_weight_list)

    global_model.set_weights(average_weights)
    global_model.save(args.model_folder+'global_model.h5')

    print('*********Test per vehicle*************')
    top_1 = []
    top_2 =[]
    top_5 = []

    x_test = [X_lidar_test,X_img_test,X_coord_test]
    scores = global_model.evaluate(x_test,y_test)
    print('preds',global_model.predict(x_test)[0])
    top_1.append(scores[1])
    top_2.append(scores[2])
    top_5.append(scores[3])
    print('comm_round: {} | global_acc: {:.3%}'.format(comm_round, sum(top_1)/len(top_1),sum(top_2)/len(top_2),sum(top_5)/len(top_5)))

    print('iteration_end_time:',time())
