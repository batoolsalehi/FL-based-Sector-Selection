from __future__ import division
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Conv2D, add,\
    Flatten, MaxPooling2D, Dense, Reshape, Input, Dropout, Concatenate, Conv1D, MaxPooling1D,Add,Lambda

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.models import model_from_json

import tensorflow.keras.utils
import numpy as np
import copy
import os
from tensorflow.keras import backend as K




def add_model(model_flag, model,save_path):

    # save the model structure first
    model_json = model.to_json()
    print('\n*************** Saving New Model Structure ***************')
    with open(os.path.join(save_path, "%s_model.json" % model_flag), "w") as json_file:
        json_file.write(model_json)
        print("json file written")
        print(os.path.join(save_path, "%s_model.json" % model_flag))


# loading the model structure from json file
def load_model_structure(model_path='/scratch/model.json'):

    # reading model from json file
    json_file = open(model_path, 'r')
    model = model_from_json(json_file.read())
    json_file.close()
    return model


def load_weights(model, weight_path = '/scratch/weights.02-3.05.hdf5'):
    model.load_weights(weight_path)



def custom_function(input):
    # mm = K.max(input)
    # return input/mm
    return K.l2_normalize(input, axis=-1)




class ModelHandler:

    def createArchitecture(self,model_type,num_classes,input_shape,chain,strategy,fusion):
        '''
        Returns a NN model.
        modelType: a string which defines the structure of the model
        numClasses: a scalar which denotes the number of classes to be predicted
        input_shape: a tuple with the dimensions of the input of the model
        chain: a string which indicates if must be returned the complete model
        up to prediction layer, or a segment of the model.
        '''

        if(model_type == 'coord_mlp'):

            input_coord = Input(shape=(input_shape, 1), name='coord_input')
            layer = Conv1D(20, 2, padding="SAME", activation='relu', name='coord_conv1')(input_coord)
            layer = Conv1D(20, 2, padding="SAME", activation='relu', name='coord_conv2')(layer)
            layer = MaxPooling1D(pool_size=2, padding="same", name='coord_maxpool1')(layer)

            layer = Conv1D(20, 2, padding="SAME", activation='relu', name='coord_conv3')(layer)
            layer = Conv1D(20, 2, padding="SAME", activation='relu', name='coord_conv4')(layer)
            layer = MaxPooling1D(pool_size=2, padding="same", name='coord_maxpool2')(layer)

            layer = Flatten(name='coord_flatten')(layer)
            layer = Dense(1024, activation='relu', name='coord_dense1')(layer)
            layer = Dropout(0.25, name='coord_dropout1')(layer)
            layer = Dense(512, activation='relu', name='coord_dense2')(layer)
            layer = Dropout(0.25, name='coord_dropout2')(layer)
            layer = Dense(256, activation='relu', name='coord_dense22')(layer)
            layer = Dropout(0.25, name='coord_dropout22')(layer)
            if fusion:
                out = Dense(64,activation='tanh',name ='coord_output')(layer)
                # out = Lambda(custom_function)(layer)
                architecture = Model(inputs = input_coord, outputs = out)
            else:
                if strategy == 'one_hot':
                    out = Dense(num_classes,activation='softmax',name ='coord_output')(layer)
                elif strategy == 'reg':
                    out = Dense(num_classes)(layer)
                architecture = Model(inputs = input_coord, outputs = out)


        elif(model_type == 'raw_image'):
            print('************You are using Tongnet model************')
            dropProb=0.25
            channel = 32
            input_img = Input(shape = input_shape, name='img_input')
            layer1 = Conv2D(channel,kernel_size=(7,7),
                           activation='relu',padding="SAME",input_shape=input_shape, name='img_conv11')(input_img)
            b = layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='img_conv3')(layer1)
            layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='img_conv4')(layer)
            layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='img_conv5')(layer)  # + b
            layer = Add(name='img_add2')([layer, b])  # DR
            layer = MaxPooling2D(pool_size=(2, 2), name='img_maxpool2')(layer)
            c = layer = Dropout(dropProb, name='img_dropout2')(layer)

            layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='img_conv6')(layer)
            layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='img_conv7')(layer)  # + c
            layer = Add(name='img_add3')([layer, c])  # DR
            layer = MaxPooling2D(pool_size=(3, 3), name='img_maxpool3')(layer)
            layer = Dropout(dropProb, name='img_dropout3')(layer)

            layer = Flatten( name='img_flatten')(layer)
            layer = Dense(512, activation='relu', name='img_dense1')(layer)
            layer = Dropout(0.25, name='img_dropout4')(layer)

            layer = Dense(256, activation='relu', name='img_dense2')(layer)
            layer = Dropout(0.25, name='img_dropout5')(layer)

            if fusion:
                out = Dense(256, activation='tanh', name='coord_tanh')(layer)
                # out = Lambda(custom_function)(layer)
                architecture = Model(inputs=input_img, outputs=out)
            else:
                if strategy == 'one_hot':
                    out = Dense(num_classes,activation='softmax',name = 'image_custom_output')(layer)
                elif strategy == 'reg':
                    out = Dense(num_classes)(layer)
                architecture = Model(inputs = input_img, outputs = out)


        elif(model_type == 'lidar_marcus'):
            print('************You are using Reslike model************')
            dropProb = 0.3
            channel = 32  # 32 now is the best, better than 64, 16
            input_lid = Input(shape=input_shape, name='lidar_input')
            a = layer = Conv2D(channel, kernel_size=(3, 3),
                               activation='relu', padding="SAME", input_shape=input_shape, name='lidar_conv1')(input_lid)
            layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv2')(layer)
            layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv3')(layer)  # + a
            layer = Add(name='lidar_add1')([layer, a])  # DR
            layer = MaxPooling2D(pool_size=(2, 2), name='lidar_maxpool1')(layer)
            b = layer = Dropout(dropProb, name='lidar_dropout1')(layer)

            layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv4')(layer)
            layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv5')(layer)  # + b
            layer = Add(name='lidar_add2')([layer, b])  # DR
            layer = MaxPooling2D(pool_size=(2, 2), name='lidar_maxpool2')(layer)
            c = layer = Dropout(dropProb, name='lidar_dropout2')(layer)

            layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv6')(layer)
            layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv7')(layer)  # + c
            layer = Add(name='lidar_add3')([layer, c])  # DR
            layer = MaxPooling2D(pool_size=(1, 2), name='lidar_maxpool3')(layer)
            d = layer = Dropout(dropProb, name='lidar_dropout3')(layer)

            # # if add this layer, need 35 epochs to converge
            # layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu',name='lidar1_added')(layer)
            # layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu',name='lidar1_added2')(layer) #+ d
            # layer = Add(name='lidar_addbetween')([layer, d])  # DR
            # layer = MaxPooling2D(pool_size=(1, 2),name='lidar_maxpool3between')(layer)
            # e = layer = Dropout(dropProb, name='lidar_dropout3between')(layer)

            layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv8')(layer)
            layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv9')(layer)  # + d
            layer = Add(name='lidar_add4')([layer, d])  # DR

            layer = Flatten(name='lidar_flatten')(layer)
            layer = Dense(1024, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), name="lidar_dense1")(layer)  #was 512
            layer = Dropout(0.2, name='lidar_dropout4')(layer)  # 0.25 is similar ... could try more values
            # layer = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), name="lidar_dense2")(layer)
            # layer = Dropout(0.2, name='lidar_dropout5')(layer)  # 0.25 is similar ... could try more values
            if fusion :
                layer = Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), name="lidar_dense2")(layer) #was 512
                out = Dropout(0.2, name='lidar_dropout5')(layer)  # 0.25 is similar ... could try more values
                # out = Dense(64, activation='tanh', name="lidar_dense_out")(layer)
                # print('out shape',out.shape)
                # out = Lambda(custom_function)(layer)
                # out = Dropout(0.2, name='lidar_dropout5')(layer)
                architecture = Model(inputs=input_lid, outputs=out)
            else:
                if strategy == 'one_hot':
                    layer = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), name="lidar_dense2")(layer)
                    layer = Dropout(0.2, name='lidar_dropout5')(layer)  # 0.25 is similar ... could try more values
                    out = Dense(num_classes, activation='softmax',name = 'lidar_output')(layer)
                elif strategy == 'reg':
                    out = Dense(num_classes)(layer)
                architecture = Model(inputs=input_lid, outputs=out)


        return architecture
