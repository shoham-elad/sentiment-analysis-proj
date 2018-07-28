"""
# coding: utf-8

# In[1]:
import time
start_time = time.time()
#imports
import pickle
from gensim import corpora, models, similarities
import numpy as np
from gensim.models import KeyedVectors
import pandas as pd
import re
#imports
import warnings
warnings.filterwarnings('ignore')
from ast import literal_eval
from keras.preprocessing.text import Tokenizer , text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import gc

from keras.backend import clear_session
import pyswarms as ps
from sklearn.metrics import accuracy_score
from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[2]:


def load():
    df = pd.read_csv('test.ft.txt',header=None, delimiter = "\n")
    df['Y'] = df[0].str[0:10]
    df['X'] = df[0].str[11:]
    #df['length'] = df.X.str.len()
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

df = load()


# In[3]:



# In[6]:


zeros = np.zeros(100)
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
emb = create_embeddings_index()

df['vector']= df.X.apply(lambda x: [emb[word] if word in emb else zeros  for word in x])
df


# In[7]:


def shrink(data,hit):
    shrinkedList = []
    for i in range(0,len(hit)):
        if i>len(data):
            return shrinkedList
        if hit[i]!=0:
            shrinkedList.append(data[i])
    return shrinkedList

# In[ ]:



"""
# In[1]:

def predictForModels(df,alls):
    predicPerMode=[]
    bin =0    
    for cost,pos,model,origMax in alls:
        bin+=1
        print('starting bin unnmber: '+str(bin))
        newData = []
        for i,row in df.iterrows():
            padded = sequence.pad_sequences([row.vector], maxlen=origMax,value = np.zeros(100))[0]
            masked = shrink(padded,pos)
            newData.append(masked)
        predictions = model.predict(np.array(newData))
        predictions = [p[0] for p in predictions]
        predicPerMode.append(predictions)
    return predicPerMode


def reArrange(predicPerMode):
    predicionts = []
    sent = 0
    for i in range(len(predicPerMode[0])):
        print('rearanging sentence number:' + str(sent))
        rowPred = []
        for j in range(len(predicPerMode)):
            rowPred.append(predicPerMode[j][i])
        predicionts.append(rowPred)
    return predicionts

df['predictions']= reArrange(predictForModels(df,alls))


def preproccess(df,alls):
    duckingAll = []
    for i,row in df.iterrows():
        #if i%10000==0:
        print(i)
        predictions=[] #~
        for cost,pos,model,origMax in alls:
            padded = sequence.pad_sequences([row.vector], maxlen=origMax,value = np.zeros(100))[0]
            masked = shrink(padded,pos)
            prediction =model.predict(np.array([masked]))[0][0]
            predictions.append(prediction)#~
            #predictions = [p[0] for p in predictions]
        duckingAll.append(predictions)
        #df.at[i,'predictions']=pd.Series(predictions)#~
        #predictions = [p[0] for p in predictions]
        #print(predictions)
        #df.iat[i,'predictions']=predictions
        
    duckingAll.extend([0]*(len(df)-2))
    df['predictions']= duckingAll






ITER = [0]

def predictionColumn(vector,alls,ITER):
    ITER[0] +=1
    print(ITER[0])
    predictions=[] #~
    for cost,pos,model,origMax in alls:
        padded = sequence.pad_sequences([vector], maxlen=origMax,value = np.zeros(100), dtype='float32')[0]
        masked = shrink(padded,pos)
        prediction =model.predict(np.array([masked]))[0]
        predictions.append(prediction)#~
    predictions = [p[0] for p in predictions]
    return predictions






dirtAlls = pickle.load(open( "alls.pkl", "rb" ))
alls = []

for i in range(len(dirtAlls)):
    cos,pos,maxL = dirtAlls[i]
    mod = load_model('modelOfBin'+str(i)+'.h5')
    alls.append((cos,pos,mod,maxL))





#df['predictions'] =df.vector.apply(predictionColumn,args=(alls,ITER,))
#preproccess(df,alls)






perc = load_model('perceptron.h5')


X = np.array([row.predictions+[row.lengths] for _,row in df.iterrows()])



ev =  perc.evaluate(X,df.Y,BATCH_SIZE=1)

print(ev)

#model.fit(np.array([[1,2,3,4,5,6]]),[0])







#load all models
#save out put of each RNN
#give perceptron its thing





"""
perModel.save('perceptron.h5')
allsSaved = []
for i in range(len(alls)):
    cos,pos,model,max = alls[i]
    model.save('modelOfBin'+str(i)+'.h5')
    allsSaved.append((cos,pos,max))

pickle.dump( allsSaved, open( "alls.pkl", "wb" ) )
"""

print(time.time()-start_time)


"""
def preproccess(df,alls):
    duckingAll = []
    for i,row in df[:2].iterrows():
        predictions=[] #~
        for cost,pos,model,origMax in alls:
            padded = sequence.pad_sequences([row.vector], maxlen=origMax,value = np.zeros(100))[0]
            masked = shrink(padded,pos)
            prediction =model.predict(np.array([masked]))[0]
            predictions.append(prediction)#~
        duckingAll.append(predictions)
        #df.at[i,'predictions']=pd.Series(predictions)#~
        predictions = [p[0] for p in predictions]
        #print(predictions)
        #df.iat[i,'predictions']=predictions
        
    duckingAll.extend([0]*(len(df)-2))
    #df['predictions']= duckingAll

preproccess(df,alls)
"""








#df = df.drop(['predictions'],axis=1)
#df['predictions']

#df['classified']=[ alls[row.bin][2].predict(sequence.pad_sequences([row.vector], maxlen=alls[row.bin][3])) for _,row in df[:2].iterrows()]

#save all max
#pad to all max
#add rnn to input




"""
relevent = df[df.bin==0]
vec = relevent.vector
labe= relevent.Y
maxSize = max(relevent.lengths)

X = sequence.pad_sequences(vec, maxlen=maxSize)
#Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, labe, test_size=0.2,random_state=42)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X[:2], labe[:2], test_size=0.2,random_state=42)

cost,pos =singlePSO(Xtrain, Ytrain,Xtest, Ytest, maxSize)

shrinked =np.array([shrink(x,pos) for x in X[:2]])
model = RNNCustom(timesteps=maxSize)#RNNCustom(VOCAB_SIZE=vocab_size,EMBEDDING_SIZE=EMBEDDING_SIZE,MAX_SENTENCE_LENGTH=m.sum())
#model.fit(X, labe,epochs=10)
#model.fit([shrink(x,pos) for x in X, labe,epochs=10)

model.fit(shrinked , labe[:2],epochs=1)
"""




# In[ ]:


"""
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
"""
