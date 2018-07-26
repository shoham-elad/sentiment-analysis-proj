
# coding: utf-8

# In[1]:

#imports
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
from keras.models import Sequential
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
    df = pd.read_csv('train.ft.txt',header=None, delimiter = "\n", nrows=50000)
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



lengths = df.lengths
maxLength = max(lengths)
minLength = min(lengths)

def histMapAndIndexes(lengths):
    HitMap ={}
    IndexMap ={}
    for i in range(0,len(lengths)):
        value = lengths[i]
        if value not in HitMap:
            IndexMap[value]=[]
            HitMap[value]=0
        HitMap[value]+=1
        IndexMap[value].append(i)
    return HitMap,IndexMap
hit,ind = histMapAndIndexes(lengths)



maxi = 0
for leng in hit:
    if hit[leng]>maxi:
        maxi = hit[leng]



mini = 50000000
for leng in hit:
    if hit[leng]<mini:
        mini = hit[leng]



X=df.X
uniqueWords ={}
for sen in X:
    arraied= sen
    for word in arraied:
        if word not in uniqueWords:
            uniqueWords[word]=0
        uniqueWords[word]+=1

        
print('max length', 'min length')        
print(maxLength,minLength)

print('hits (length:times seen)')
print(len(hit))        

print('unique')
print(len(uniqueWords))

print('min uccor','maxi uccor')
print(mini,maxi)

#print(hit)
#print(lengths)
hit


# In[4]:


bins= []
binInsex=[[]]
count=0
nums = list(hit)
nums.sort()
i=0
while i < len(nums):
    if count >=10000:
        bins.append(count)
        count=0
        binInsex.append([])
    else:
        if count >=20000 and hit[nums[i]]>=15000:
            bins.append(count)
            bins.append(hit[nums[i]])
            binInsex.append([nums[i]])
            i+=1
            count=0
            binInsex.append([])
    count += hit[nums[i]]
    binInsex[-1].append(nums[i])
    i+=1
    #print(i)





len(bins),min(bins),max(bins),bins,binInsex


# In[5]:


lengthToBin = {}
for i in range(len(binInsex)):
    for num in binInsex[i]:
        lengthToBin[num]=i

df['bin']= [lengthToBin[row.lengths] for _,row in df.iterrows()]
df


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


def RNN(VOCAB_SIZE,EMBEDDING_SIZE,MAX_SENTENCE_LENGTH,Hidden_Layer_Size= 64, RLAYER=LSTM,
        ACTIVATION=Activation("sigmoid"),LOSS="binary_crossentropy",OPTIMIZER='nadam',METRICS=["accuracy"]):
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
    #model.add(SpatialDropout1D(0.2))
    #model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=float(0.2), recurrent_dropout=float(0.2)))
    model.add(RLAYER(Hidden_Layer_Size))
    model.add(Dense(1))
    model.add(ACTIVATION)
    model.compile(loss=LOSS, optimizer=OPTIMIZER,metrics=METRICS)
#     history = model.fit(InputTrain, LabelsTrain,,,,,,,,,,,,,,,,validation_data=(InputTest, LabelsTest))
    return model



def RNNCustom(BATCH_SIZE = 32, input_dim = 100,Hidden_Layer_Size= 64,timesteps=8, RLAYER=LSTM,
        ACTIVATION=Activation("sigmoid"),LOSS="binary_crossentropy",OPTIMIZER='nadam',METRICS=["accuracy"]):
    model = Sequential()
    #model.add(Embedding(VOCAB_SIZE, INPUT_SIZE,input_length=MAX_SENTENCE_LENGTH))
    #model.add(SpatialDropout1D(0.2))
    #model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=float(0.2), recurrent_dropout=float(0.2)))
    model.add(RLAYER(Hidden_Layer_Size,input_shape=(timesteps,input_dim)))
    model.add(Dense(1))
    model.add(ACTIVATION)
    model.compile(loss=LOSS, optimizer=OPTIMIZER,metrics=METRICS)
    #history = model.fit(InputTrain, LabelsTrain, batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,validation_data=(InputTest, LabelsTest))
    return model


def binaryPSO(f,dimensions,n_particles,print_step,iters,epochs,options= {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 10, 'p':2},verbose=2):
    optimizer = ps.discrete.BinaryPSO(n_particles=20, dimensions=dimensions, options=options)
    cost, pos = optimizer.optimize(f, print_step=print_step, iters=iters, verbose=verbose)
    return cost,pos



def create_fit(X,Y,X_test,Y_test,dimensions,alpha=0.5,NUM_EPOCHS = 10):
    
    def fit_per_particle(m,alpha):
        print("m:")
        print(m)
        print(len(m))
        if np.count_nonzero(m) == 0:
            X_subset = X
            X_test_subset = X_test
        else:
            X_subset = X[:,m==1]
            X_test_subset = X_test[:,m==1]
        model = RNNCustom(timesteps=m.sum())#RNNCustom(VOCAB_SIZE=vocab_size,EMBEDDING_SIZE=EMBEDDING_SIZE,MAX_SENTENCE_LENGTH=m.sum())
        # Perform classification and store performance in P
        print(X[0])
        print(X_subset[0])
        model.fit(X_subset, Y,epochs=NUM_EPOCHS,validation_data=(X_test_subset, Y_test))
        P = accuracy_score(model.predict_classes(X_subset),Y)
        # Compute for the objective function
        j = (alpha * (1.0 - P)
            + (1.0 - alpha) * (1 - (X_subset.shape[1] / dimensions)))
        #clear_session()
        return j
    
    def fit(x,alpha=0.5):
        fit.counter+=1
        print("current Iteration: " + str(fit.counter))
        n_particles = x.shape[0]
        j = [fit_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)
    fit.counter = 0



    
    #def fit(x,alpha=0.5):
    #    n_particles = x.shape[0]
    #    j = [fit_per_particle(x[i], alpha) for i in range(n_particles)]
    #    return np.array(j)
    return fit




def singlePSO(X,Y,Xtest,Ytest,MAX_SENTENCE_LENGTH):
    fit = create_fit(X=Xtrain,Y=Ytrain,X_test=Xtest,Y_test=Ytest,dimensions=MAX_SENTENCE_LENGTH,alpha=0.5)
    print(fit)
    cost,pos = binaryPSO(f=fit,dimensions=MAX_SENTENCE_LENGTH,n_particles=3,print_step=100,iters=5,epochs=3)
    print(cost)
    #return cost,pos


# In[ ]:


relevent = df[df.bin==0]
vec = relevent.vector
labe= relevent.Y
maxSize = max(relevent.lengths)


X = sequence.pad_sequences(vec, maxlen=maxSize)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, labe, test_size=0.2,random_state=42)



cost,pos =singlePSO(Xtrain, Ytrain,Xtest, Ytest, maxSize)


# In[1]:





# In[ ]:


"""
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
"""
