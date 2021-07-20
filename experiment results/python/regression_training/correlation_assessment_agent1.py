import tensorflow as tf
import csv
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)

#get data

outputs=[]
inputs=[]

with open('C:\\Users\\owenb\\Desktop\\experiment results\\agent1_data\\actions.csv',newline='') as csvfile:
    reader=csv.reader(csvfile,dialect='excel')
    for x in reader:
        outputs.append(x)

with open('C:\\Users\\owenb\\Desktop\\experiment results\\agent1_data\\messages_1_3.csv',newline='') as csvfile:
    reader=csv.reader(csvfile,dialect='excel')
    for x in reader:
        inputs.append(x)

dataset=[[inputs[x],outputs[x]] for x in range(len(inputs))]

del dataset[8500:]

#process data

length=int(len(dataset)*0.8)
train_dataset=random.sample(dataset,length)
test_dataset=[y for y in dataset if y not in train_dataset]

train_features=[x[0] for x in train_dataset]
train_labels=[x[1] for x in train_dataset]

test_features=[x[0] for x in test_dataset]
test_labels=[x[1] for x in test_dataset]

for x in range(len(train_features)):
    train_features[x]=np.array(np.expand_dims([float(y) for y in train_features[x]], axis=0))
    train_labels[x]=np.array(np.expand_dims([float(y) for y in train_labels[x]], axis=0))

train_features=np.array(train_features)
train_labels=np.array(train_labels)

for x in range(len(test_features)):
    test_features[x]=np.array(np.expand_dims([float(y) for y in test_features[x]], axis=0))
    test_labels[x]=np.array(np.expand_dims([float(y) for y in test_labels[x]], axis=0))

test_features=np.array(test_features)
test_labels=np.array(test_labels)

#make model

normalizer=preprocessing.Normalization(input_shape=(1, 100))
normalizer.adapt(train_features)
linear_model=tf.keras.Sequential([normalizer,tf.keras.layers.Dense(6)])

linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),loss='mean_absolute_error')

#train model



history = linear_model.fit(x=train_features, y=train_labels, epochs=100,verbose=1,validation_split = 0.2)

plot_loss(history)

#test model

test_results = linear_model.evaluate(test_features, test_labels, verbose=0)

print("agent 1 post train error="+str(test_results))
linear_model.save('agent1.h5')
