# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 21:42:02 2018

@author: Elad-PC
"""

#imports
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model
from ast import literal_eval
import numpy as np
import os
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
    
    
    
def compare(real,predicted):
    TP=0
    FP=0
    FN=0
    TN=0
    correct = 0
    for i in range(len(real)):
        if i >= len(predicted):
            break
        if round(real[i])==round(predicted[i]):
            correct+=1
            if round(predicted[i])==1:
                TP+=1
            else:
                TN+=1
        else:
            if round(predicted[i])==1:
                FP+=1
            else:
                FN+=1
    recall = TP/(TP+FN)
    precision=TP/(TP+FP)
    F1=2*(recall*precision)/(recall+precision)
    return correct/len(real),TP,FP,TN,FN,recall,precision,F1

    
def flow(filename,numOfRows):
    print('getting df')
    df = load(filename,numOfRows)
    print('getting embeding')
    emb = create_embeddings_index()
    print('embeding and padding (this will take long)')
    padding(df,emb)
    #df.head()
    
    X = np.array(df.vector.tolist())
    Y = df.Y.values
    model = load_model('baseline.h5')
    print('evaluating')
    ev = model.evaluate(X,Y)
    print(ev)
    pred = model.predict(X)
    pred = pred.reshape(1,numOfRows)[0]
    print(compare(df.Y,pred))

#flow()

import sys
filename = sys.argv[1]#'test.ft.txt'
numOfRows = (int)(sys.argv[2])#100000
flow(filename,numOfRows)





"""
print('getting df')
df = load(filename,numOfRows)
print('getting embeding')
emb = create_embeddings_index()
print('embeding and padding (this will take long)')
padding(df,emb)
model = load_model('finals\\baseline50000\\baseline.h5')
predictions = model.predict(np.array(df.vector.tolist()))
df['predictions']= predictions

def compare(real,predicted):
    correct = 0
    for i in range(len(real)):
        if i >= len(predicted):
            break
        if round(real[i])==round(predicted[i]):
            correct+=1
    return correct/len(real)



bin0 = [row for _,row in df.iterrows() if row.lengths <= 39]
predicted0 = [row.predictions for row in bin0]
labels0 = [row.Y for row in bin0]
comp0 = compare(labels0,predicted0)



bin1 = [row for _,row in df.iterrows() if row.lengths <= 61 and row.lengths > 39]
predicted1 = [row.predictions for row in bin1]
labels1 = [row.Y for row in bin1]
comp1 = compare(labels1,predicted1)


bin2 = [row for _,row in df.iterrows() if row.lengths <= 88 and row.lengths > 61]
predicted2 = [row.predictions for row in bin2]
labels2 = [row.Y for row in bin2]
comp2 = compare(labels2,predicted2)

bin3 = [row for _,row in df.iterrows() if row.lengths <= 125 and row.lengths > 88]
predicted3 = [row.predictions for row in bin3]
labels3 = [row.Y for row in bin3]
comp3 = compare(labels3,predicted3)

bin4 = [row for _,row in df.iterrows() if row.lengths > 125]
predicted4 = [row.predictions for row in bin4]
labels4 = [row.Y for row in bin4]
comp4 = compare(labels4,predicted4)


print(comp0,comp1,comp2,comp3,comp4)
"""






"""
50000
val_acc: 0.8380
100000/100000 [==============================] - 264s 3ms/step
[0.37204062418937683, 0.83808]

0.8571293673276676 0.8473244968090329 0.8358890377648752 0.8296534653465346 0.8172274711310017

TP      ,FP  ,TN    ,FN       ,recall               ,precision       ,F1
41051, 6664, 42757, 9528, 0.8116214239111094, 0.8603374200985016, 0.8352697011007792
"""


"""
10000
val_acc: 0.64
100000/100000 [==============================] - 302s 3ms/step
[0.5902305526256562, 0.68659]

0.7694050991501417 0.7123711340206186 0.6740608406530694 0.6440594059405941 0.6220785678766783

TP      ,FP  ,TN    ,FN       ,recall               ,precision       ,F1
0.68659, 46000, 26762, 22659, 4579, 0.9094683564325116, 0.6321981253951239, 0.7458995792153462)
"""







