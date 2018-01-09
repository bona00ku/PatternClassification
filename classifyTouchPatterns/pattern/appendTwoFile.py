# keras version 2.1.2
# test file
import numpy as np
from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.convolutional import Conv3D,MaxPooling3D

from keras.utils import np_utils


import csv
import os


def make3Ddata(filename,label):
    #the function converts raw data to spatial format(nb_sensors,rows,cols)
    
    rows = 9
    cols = 5
    nb_sensors = 4
    timesteps = 2
    num_data = 52
    
    with open(filename) as f:
        lines = (line for line in f if not line.startswith('#'))
        data = np.loadtxt(lines,delimiter = ',')
    #print('raw data shape: ',np.shape(data))    
   
    final=[]
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
    print('3d data shape: ',np.shape(final))
    print('final',final)
    ret =[] 
    for i in range((len(final)-timesteps+1)):
        ret.append(final[i:i+timesteps+1])
        
    print('return shape,data ',np.shape(ret),ret)
    y = label* np.ones(len(ret))
    ret = np.asarray(ret)
    y = np.asarray(y)
    #print('4d data,label shape: ',np.shape(ret),np.shape(y))
    return ret,y


def load_single_data(filename,timesteps,num_data,label):
    data = np.loadtxt(filename,delimiter=',')
   
    #split the data to (samples, timesteps , num_data) shape 
    result = []
    for index in range(len(data)-timesteps+1):
        result.append(data[index: index+ timesteps,:num_data])
    result = np.array(result)
    
    #add label
    label_array = label * np.ones(len(result))
    
    print(label,"data ,label size",np.shape(result),np.shape(label_array))
    
    return (result,label_array)

def load_data_3d(f_save):
    classes = ['petcheat','petting']
    people = ['choi','lee']
    
    nb_classes = len(classes)
    num_data = 52
    timesteps= 10  #0.5s data with collection freq 20hz
    x=[]
    y=[]
    
    for i in range(len(classes)):
        for person in range(len(people)):
            data,label_array = make3Ddata(classes[i]+people[person]+'.csv',i)
            
            train_size = int(len(data)*0.7)
            test_size = len(data)-train_size
        
            if (i==0):
                X_train,X_test = np.array(data[0:train_size]),np.array(data[train_size:]) 
                Y_train,Y_test = np.array(label_array[0:train_size]),np.array(label_array[train_size:]) 
            else:
                X_train= np.concatenate((X_train, data[0:train_size]),axis=0)
                X_test= np.concatenate((X_test, data[train_size:]),axis=0)
                Y_train = np.concatenate((Y_train,label_array[0:train_size]),axis=0)
                Y_test = np.concatenate((Y_test,label_array[train_size:]),axis=0)
       
    
    #split to train and testing
    print('trainx,testx,trainy,test y shape',np.shape(X_train),np.shape(X_test),np.shape(Y_train),np.shape(Y_test)) 
    print('Y_train: ' ,Y_train)
    print('Y_test: ' ,Y_test)
    return X_train,X_test,Y_train,Y_test
#

def load_data_1d(f_save):
    nb_classes = 2
    num_data = 3
    timesteps= 3  #0.5s data with collection freq 20hz
    
    x=[]
    y=[]
    for i in range(nb_classes):
        data,label_array = load_single_data('simple'+str(i+1)+'.csv',timesteps,num_data,i+1)
        train_size = int(len(data)*0.7)
        test_size = len(data)-train_size
    
        if (i==0):
            X_train,X_test = np.array(data[0:train_size]),np.array(data[train_size:]) 
            Y_train,Y_test = np.array(label_array[0:train_size]),np.array(label_array[train_size:]) 
        else:
            X_train= np.concatenate((X_train, data[0:train_size]),axis=0)
            X_test= np.concatenate((X_test, data[train_size:]),axis=0)
            Y_train = np.concatenate((Y_train,label_array[0:train_size]),axis=0)
            Y_test = np.concatenate((Y_test,label_array[train_size:]),axis=0)
   
    
    #split to train and testing
    print('trainx,testx,trainy,test y shape',np.shape(X_train),np.shape(X_test),np.shape(Y_train),np.shape(Y_test)) 
    print('Y_train: ' ,Y_train)
    print('Y_test: ' ,Y_test)
    return X_train,X_test,Y_train,Y_test
#    with open(f_save,"wb") as csv_file:
#        writer = csv.writer(csv_file,delimiter=',')
#        for line1,line2 in x,y:
#            writer.writerow([line1,line2])
def build_3d_model():
    nb_classes = 2
    nb_filters = 64
    kernal_size = (4,3,3)
    pool_size = (1,2,2)
    #input_shape = (timesteps,nb_sensors,rows,cols)
    input_shape = (3,4,9,5)
    
    model = Sequential()
    #1st layer group
    model.add(Conv3D(nb_filters,kernal_size, name = 'conv1',
                activation = 'relu', input_shape = input_shape,
                strides = (1,1,1), data_format = 'channels_first',
                padding = 'same'))
    model.add(MaxPooling3D(padding = 'valid',pool_size=pool_size,
              strides = (1,2,2),data_format = 'channels_first',name='pool1'))
    
    #2nd layer group
    model.add(Conv3D(nb_filters,kernal_size,strides = (1,1,1),
        data_format = 'channels_first',input_shape=input_shape,
        activation='relu', padding = 'same', name = 'conv2'))
    model.add(MaxPooling3D(pool_size=pool_size,strides = (1,2,2),
              padding = 'valid',data_format = 'channels_first',name='pool2'))
    
    #3rd layer group
    model.add(Conv3D(nb_filters,kernal_size,strides = (1,1,1),
        data_format = 'channels_first',input_shape=input_shape,
        activation='relu', padding = 'same', name = 'conv3'))
    
    #model.add(MaxPooling3D(pool_size=pool_size,strides = (1,2,2),
    #          padding = 'same',name='pool3'))
    
    #FC layers
    model.add(Flatten())
    
    model.add(Dense(256,activation='relu',name='fc1'))
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes,activation='relu',name='fc2'))
    model.add(Dropout(0.5))
    
    print(model.summary())
    return model
        
def build_model(timesteps,num_data,depth):
    nb_classes = 2
    nb_filters = 64
    kernal_size = (3,3)
    pool_size = (2,2)
    stride = 2
    input_shape = (timesteps ,num_data,depth) 
    
    #cnn model
    model = Sequential()
    model.add(Conv2D(nb_filters,kernal_size,stride = (2,1),padding='valid',
                    input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters,kernal_size))
    model.add(MaxPooling2D(pool_size = pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))


    model.compile(loss='categorical_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])
    
    model.fit(X_train,Y_train,epochs = 100,verbose=1,
              validation_data=(X_test,Y_test))

a = make3Ddata("simple3.csv",1)
#trainX,testX,trainY,testY=load_data_3d('output.csv')
#build_3d_model()
