import tensorflow as tf
import csv
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import statistics

outputs=[]
inputs=[]

with open('C:\\Users\\owenb\\Desktop\\experiment results\\agent3_data\\actions.csv',newline='') as csvfile:
    reader=csv.reader(csvfile,dialect='excel')
    for x in reader:
        outputs.append(x)

with open('C:\\Users\\owenb\\Desktop\\experiment results\\agent3_data\\messages_3_2.csv',newline='') as csvfile:
    reader=csv.reader(csvfile,dialect='excel')
    for x in reader:
        inputs.append(x)

dataset=[[inputs[x],outputs[x]] for x in range(len(inputs))]

del dataset[8500:]

avg_error=[]

linear_model=tf.keras.models.load_model('C:\\Users\\owenb\\Desktop\\experiment results\\trained regression models\\agent3-2.h5')

test_features=[x[0] for x in dataset]
test_labels=[x[1] for x in dataset]

for x in range(len(test_features)):
    test_features[x]=np.array(np.expand_dims([float(y) for y in test_features[x]], axis=0))
    test_labels[x]=np.array(np.expand_dims([float(y) for y in test_labels[x]], axis=0))

test_features=np.array(test_features)
test_labels=np.array(test_labels)

for x in range(8500):
    avg_error.append(linear_model.evaluate(test_features[x], test_labels[x], verbose=0))

print("agent 3-2 posttrain standard deviation="+str(statistics.pstdev(avg_error))+" and mean="+str(statistics.mean(avg_error)))
