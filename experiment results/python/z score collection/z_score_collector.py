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

with open('C:\\Users\\owenb\\Desktop\\experiment results\\agent3_data\\messages_3_1.csv',newline='') as csvfile:
    reader=csv.reader(csvfile,dialect='excel')
    for x in reader:
        inputs.append(x)

dataset=[[inputs[x],outputs[x]] for x in range(len(inputs))]

del dataset[8500:]

avg_error=[]

linear_model_1=tf.keras.models.load_model('C:\\Users\\owenb\\Desktop\\experiment results\\trained regression models\\agent3-1.h5')

test_features=[x[0] for x in dataset]
test_labels=[x[1] for x in dataset]

for x in range(len(test_features)):
    test_features[x]=np.array(np.expand_dims([float(y) for y in test_features[x]], axis=0))
    test_labels[x]=np.array(np.expand_dims([float(y) for y in test_labels[x]], axis=0))

test_features_1=np.array(test_features)
test_labels_1=np.array(test_labels)

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

linear_model_2=tf.keras.models.load_model('C:\\Users\\owenb\\Desktop\\experiment results\\trained regression models\\agent3-2.h5')

test_features=[x[0] for x in dataset]
test_labels=[x[1] for x in dataset]

for x in range(len(test_features)):
    test_features[x]=np.array(np.expand_dims([float(y) for y in test_features[x]], axis=0))
    test_labels[x]=np.array(np.expand_dims([float(y) for y in test_labels[x]], axis=0))

test_features_2=np.array(test_features)
test_labels_2=np.array(test_labels)

z_score=[]

for x in range(8500):
    out1=linear_model_1.evaluate(test_features_1[x],test_labels_1[x],verbose=0)
    out2=linear_model_2.evaluate(test_features_2[x],test_labels_2[x],verbose=0)

    t1=(out1-1.6090001418441533)/(26.479717810642885)
    t2=(out2-0.850237799019498)/(0.5489453711734984)
    z_score.append([t1,t2,abs(t1-t2)])

with open('z_scores.csv','w',newline='') as csvfile:
    writer=csv.writer(csvfile,dialect='excel')
    for x in z_score:
        writer.writerow(x)
