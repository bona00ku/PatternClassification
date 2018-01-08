import numpy as np
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Activation, LSTM
from keras.utils import np_utils
from keras.models import model_from_json
import os

# brew install graphviz
# pip3 install graphviz
# pip3 install pydot-ng
from keras.utils.vis_utils import plot_model

# sample text
sample = "im hyun jin"

char_set = list(set(sample))  # id -> char ['i', 'l', 'e', 'o', 'h']
char_dic = {w: i for i, w in enumerate(char_set)}

x_str = sample[:-1]
y_str = sample[1:]

data_dim = len(char_set)
timesteps = len(y_str)
num_classes = len(char_set)

print(x_str, y_str)

x = [char_dic[c] for c in x_str]  # char to index
y = [char_dic[c] for c in y_str]  # char to index

# One-hot encoding
x = np_utils.to_categorical(x, num_classes=num_classes)
# reshape X to be [samples, time steps, features]
x = np.reshape(x, (-1, len(x), data_dim))
print(x.shape)

# One-hot encoding
y = np_utils.to_categorical(y, num_classes=num_classes)
# time steps
y = np.reshape(y, (-1, len(y), data_dim))
print(y.shape)

model = Sequential()
model.add(LSTM(num_classes, input_shape=(
    timesteps, data_dim), return_sequences=True))
model.add(TimeDistributed(Dense(num_classes)))
model.add(Activation('softmax'))
model.summary()
# Store model graph in png
# (Error occurs on in python interactive shell)
plot_model(model, to_file=os.path.basename(__file__) + '.png', show_shapes=True)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x, y, batch_size=100,epochs=1,verbose=1 )

predictions = model.predict(x, verbose=0)
for i, prediction in enumerate(predictions):
    print(prediction)
    x_index = np.argmax(x[i], axis=1)
    x_str = [char_set[j] for j in x_index]
    print(x_index, ''.join(x_str))

    index = np.argmax(prediction, axis=1)
    result = [char_set[j] for j in index]
    print(index, ''.join(result))

#evaluate the model
a=model.evaluate(x,y,verbose=0)
print(model.metrics_names)
print(a[0],a[1])

# serialize the  mdoel to JSON
model_json = model.to_json()
with open("model_hello.json","w") as json_file:
    json_file.write(model_json)

#serialize weights to HDF5
model.save_weights("model_hello.h5")
print("saved model at disk")


#load json and create model
json_file = open("model_hello.json","r")
loaded_model_json=json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#load weights into new model
loaded_model.load_weights("model_hello.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
b=loaded_model.evaluate(x,y,verbose=0)
print(b[0],b[1])
