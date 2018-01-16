import time
import csv
import struct
import numpy as np

def getData(ser):
    k =0
    ret = []
    try:
        while True:
            data = ser.readline()
            single=[]    
            if(len(data)!=82): 
                continue
            k+=1
            print(k)
            data=data[2:-2]
            for j in range(4):
                for i in range(13):
                    if(j ==0 or j ==1):
                        single.append(struct.unpack(">b",data[13*j+i])[0])
                    else:
                        single.append( struct.unpack("<h",
                                data[13*j+ 2*i]+data[13*j + 2*i +1])[0])
            ret.append(single)
	    time.sleep(0.01)
            if(len(ret)==10):
                return ret,data
    except KeyboardInterrupt:
        pass
                
def shape(data):
    #data: 1*52 raw data
    rows = 9
    cols = 5
    nb_sensors = 4
    num_data = 52
    
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
    print('3d data shape: ',np.shape(final))
    print('final:',final)
    ret=[final] 
    # transpose final(len(data),channels,rows,cols) data
    # to (channels,len(data),rows,cols)
    #print('transpose final shape,data: ',np.shape(a),a)
    ret = np.transpose(ret,[0,2,1,3,4])
    ret = np.asarray(ret)
    return ret

