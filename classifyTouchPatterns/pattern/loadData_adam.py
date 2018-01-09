#keras version 2.1.2
import numpy as np
from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.convolutional import Conv3D,MaxPooling3D
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.models import model_from_json

import os
import time
import csv
import h5py

def make3Ddata(filename,label,timesteps):
    #the function converts raw data to spatial format(nb_sensors,rows,cols)
    
    rows = 9
    cols = 5
    nb_sensors = 4
    num_data = 52
    
    with open(filename) as f:
        lines = (line for line in f if not line.startswith('#'))
        data = np.loadtxt(lines,delimiter = ',')
    #data = np.loadtxt(filename,delimiter=',')
    #print('raw data shape: ',np.shape(data))    
    #size = int(len(data)*0.5)
    #data = data[:size]
    #print('trimed raw data shape: ',np.shape(data))    
    final = []
    for i in range(len(data)):
        result =np.zeros((nb_sensors,rows,cols))
        for ch in range(1,nb_sensors):
            for k in range(13):
                if(k<5):
                    if(ch == 1): 
                        result[ch-1][k*2][2] = data[i][k*2]
                        result[ch][k*2][2] = data[i][k*2+1]
                    else:
                        result[ch][k*2][2] = data[i][ch*13+k]
                elif(k<9):
                    row = (k-5)*2+1
                    if(ch==1):
                        result[ch-1][row][0] = data[i][k*2]
                        result[ch][row][0] = data[i][k*2+1]
                    else:
                        result[ch][row][0] = data[i][ch*13+k]
                else:
                    row = (k-9)*2+1
                    if(ch==1):
                        result[ch-1][row][4] = data[i][k*2]
                        result[ch][row][4] = data[i][k*2+1]
                    else:
                        result[ch][row][4] = data[i][ch*13+k]
        final.append(result)
    #print('3d data shape: ',np.shape(final))
    
    # transpose final(len(data),channels,rows,cols) data
    # to (channels,len(data),rows,cols)
    final= np.transpose(final,[1,0,2,3])
    #print('transpose final shape,data: ',np.shape(a),a)
    
    ret = [[] for i in range(nb_sensors)]
    for ch in range(nb_sensors):
        for i in range(len(data)-timesteps+1):
            ret[ch].append(final[ch][i:i+timesteps])
   
    y = label* np.ones(len(ret))
    ret = np.asarray(ret)
    y = np.asarray(y)
    print('4d data,label shape: ',np.shape(ret),np.shape(y),'\n')
    #print('return prs: ',ret[2])
    return ret,y

a= make3Ddata('simple3.csv',1,3)

def load_data():
    classes = ['petcheat','petting']
    people = ['jang','lee','do','choi']
    
    num_data = 52
    timesteps= 10  #0.5s data with collection freq 20hz
    x=[]
    y=[]
    tot_train = 0
    tot_test = 0
    X_train=[]
    X_test=[]
    Y_train=[]
    Y_test =[]
    for i in range(len(classes)):
        for person in range(len(people)):
            #print("> Load 3d data " + classes[i] +people[person] + " with label " + str(i))
            data,label_array = make3Ddata(classes[i]+people[person]+'.csv',i,timesteps)
            
            train_size = int(len(data)*0.7)
            test_size = len(data)-train_size
            tot_train += train_size
            tot_test += test_size

            if (i==0 and person == 0):
                X_train,X_test = np.array(data[0:train_size]),np.array(data[train_size:]) 
                Y_train,Y_test = np.array(label_array[0:train_size]),np.array(label_array[train_size:]) 
            else:
                X_train= np.concatenate((X_train, data[0:train_size]),axis=0)
                X_test= np.concatenate((X_test, data[train_size:]),axis=0)
                Y_train = np.concatenate((Y_train,label_array[0:train_size]),axis=0)
                Y_test = np.concatenate((Y_test,label_array[train_size:]),axis=0)
            
    #print("total data, train, test length : ", tot_train + tot_test, tot_train, tot_test ) 
    #split to train and testing
    print('trainx,testx,trainy,test y shape',np.shape(X_train),np.shape(X_test),np.shape(Y_train),np.shape(Y_test)) 
    #print('Y_train: ' ,Y_train)
    #print('Y_test: ' ,Y_test)
    return X_train,X_test,Y_train,Y_test

def build_lstm(nb_classes):
    timesteps = 10
    input_shape = (timesteps,4,9,5)
    
    model = Sequential()
    model.add(LSTM(64,return_sequences = False,data_format =  'channels_first',
        input_shape = input_shape,
                dropout = 0.5))
    model.add(Dense(256,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes,activation = 'softmax'))
    return model

def build_cnn(nb_classes):
    #kernal_size = (depth,row,col)
    kernal_size = (4,3,3)
    pool_size = (1,2,2)
    timesteps =10
    #input_shape = (timesteps,nb_sensors,rows,cols)
    input_shape = (timesteps,4,9,5)
    
    model = Sequential()
    #1st layer group
    model.add(Conv3D(32,kernal_size, name = 'conv1',
                activation = 'relu', input_shape = input_shape,
                strides = (1,1,1), data_format = 'channels_first',
                padding = 'same'))
    model.add(MaxPooling3D(padding = 'valid',pool_size=pool_size,
              strides = (1,2,2),data_format = 'channels_first',name='pool1'))
    
    #model.add(Dropout(0.25))
    #2nd layer group
    model.add(Conv3D(64,kernal_size,strides = (1,1,1),
        data_format = 'channels_first',input_shape=input_shape,
        activation='relu', padding = 'same', name = 'conv2'))
    model.add(MaxPooling3D(pool_size=pool_size,strides = (1,2,2),
              padding = 'valid',data_format = 'channels_first',name='pool2'))
    
    #model.add(Dropout(0.25))
    #3rd layer group
    model.add(Conv3D(128,kernal_size,strides = (1,1,1),
        data_format = 'channels_first',input_shape=input_shape,
        activation='relu', padding = 'same', name = 'conv3a'))
    
    model.add(Conv3D(128,kernal_size,strides = (1,1,1),
        data_format = 'channels_first',input_shape=input_shape,
        activation='relu', padding = 'same', name = 'conv3b'))
    
    model.add(MaxPooling3D(pool_size=pool_size,strides = (1,2,2),
              padding = 'same',name='pool3'))
    
    #model.add(Dropout(0.25))
    #FC layers
    model.add(Flatten())
    
    model.add(Dense(256,activation='relu',name='fc1'))
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes,activation='relu',name='fc2'))
    model.add(Dropout(0.5))
    
    return model

def save_model(model,f_output):
    model_json = model.to_json()
    with open(f_output + ".json","w") as json_file:
        json_file.write(model_json)

    #serialize weights to HDF5
    model.save_weights(f_output + ".h5")
    print(" saved model at disk")


def load_model(filename):
    #load json and create model
    json_file = open(filename + 'json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    #load weights into a new model
    loaded_model.load_weights(filename + '.h5')
    print("Loaded model from disk")

    loded_model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    
#a = make3Ddata("simple3.csv",1)

def run():
    nb_classes = 2
    
    trainX,testX,trainY,testY = load_data()
    
    trainX = trainX.astype('int16')
    testX = testX.astype('int16')
    trainY = np_utils.to_categorical(trainY,nb_classes)
    testY = np_utils.to_categorical(testY,nb_classes)

    model = build_cnn(nb_classes)
    #model = build_lstm(nb_classes)
    print(model.summary())
        
    start = time.time()

    model.compile(loss='categorical_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])
    
    print("> Compile time :",time.time()-start)
    
    start = time.time()
    #model.fit(trainX, trainY, epochs = 100,verbose=1,
    #       validation_data=(testX,testY))
    
    print(">>Fitting takes time: ",time.time()-start)
    score = model.evaluate (testX, testY, verbose = 0)
    print('Test score: ', score[0])
    print('Test accuracy: ',score[1])

    save_model(model,"outputCNN2")

#run()

