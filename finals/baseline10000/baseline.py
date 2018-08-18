# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 21:42:02 2018

@author: Elad-PC
"""

# coding: utf-8
#imports
import numpy as np
import pandas as pd
import warnings
import numpy as np
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[2]:


def load(fname='test.ft.txt',NROWS=None):
    from keras.preprocessing.text import text_to_word_sequence
    df = pd.read_csv(fname,header=None, delimiter = "\n",nrows=NROWS)
    df['Y'] = df[0].str[0:10]
    df['X'] = df[0].str[11:]
    df.drop(0,axis=1, inplace=True)
    df.Y = df.Y.apply(lambda label : 0 if label=="__label__1" else 1)

    for _,row in df.iterrows():
        row.X = text_to_word_sequence(row.X)


    docs =df.X.astype(str)
    lengths=[]

    for index,sentence in enumerate(docs):
        docs[index] = text_to_word_sequence(sentence)
        lengths.append(len(docs[index]))
    df.X = docs
    df['lengths']=lengths

    return df





def create_embeddings_index():
    embeddings_index = {}
    f = open('glove.6B.100d.txt',encoding='UTF-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def padding(df,emb):
    zeros = np.zeros(100)
    from keras.preprocessing.sequence import pad_sequences
    df['vector']= df.X.apply(lambda x: pad_sequences([[emb[word] if word in emb else zeros  for word in x]],dtype='float32',maxlen=200,value=zeros)[0])




def RNN(InputTrain, LabelsTrain ,NUM_EPOCHS=2,BATCH_SIZE = 4096, input_dim = 100,Hidden_Layer_Size= 64,timesteps=200, RLAYER=None,
        ACTIVATION="sigmoid",LOSS="binary_crossentropy",OPTIMIZER='nadam',METRICS=["accuracy"]):
    from keras.layers.core import Activation, Dense
    from keras.layers.recurrent import LSTM
    from keras.models import Sequential 
    RLAYER = LSTM if RLAYER is None else RLAYER
    print(InputTrain.shape)
    print(LabelsTrain.shape)
    model = Sequential()
    model.add(RLAYER(Hidden_Layer_Size,input_shape=(timesteps,input_dim)))
    model.add(Dense(1))
    model.add(Activation(ACTIVATION))
    model.compile(loss=LOSS, optimizer=OPTIMIZER,metrics=METRICS)
    history = model.fit(InputTrain, LabelsTrain, batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,validation_split=0.2)
    return model


def flow(filename,numOfRows):
    
    #poses=[np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]), array([0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0,1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1,0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1]), array([1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0,1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0,1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]), array([1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0,1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1,1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1,0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1]), array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1,0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0,0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1,1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0,0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1,0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1,1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1,1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1])]
    
    df = load(filename,numOfRows)
    emb = create_embeddings_index()
    padding(df,emb)
    X = np.array(df.vector.tolist())
    Y = df.Y.values
    model =  RNN(X,Y,NUM_EPOCHS=20,input_dim=len(X[0][0]),timesteps=len(X[0]))    
    model.save('baseline.h5')



import sys
filename = sys.argv[1]#'test.ft.txt'
numOfRows = (int)(sys.argv[2])#50000
flow(filename,numOfRows)





