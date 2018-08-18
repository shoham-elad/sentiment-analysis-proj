# In[1]:
import time
start_time = time.time()
#imports
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from keras.preprocessing.text import text_to_word_sequence
from keras.models import load_model
from keras.preprocessing import sequence
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# In[2]:


def load(fileName='test.ft.txt',NROWS=None):
    df = pd.read_csv(fileName,header=None, delimiter = "\n", nrows=NROWS)#100000
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


def shrink(data,hit):
    shrinkedList = []
    for i in range(0,len(hit)):
        if i>len(data):
            return shrinkedList
        if hit[i]!=0:
            shrinkedList.append(data[i])
    return shrinkedList


def embed(df,emb):
    df['vector']= df.X.apply(lambda x: [emb[word] if word in emb else np.zeros(100)  for word in x])

# In[1]:

I = [0]
def predictionColumn(vector,alls):
    predictions=[] 
    I[0]+=1
    print(I[0])
    for cost,pos,model,origMax in alls:
        padded = sequence.pad_sequences([vector], maxlen=origMax,value = np.zeros(100), dtype='float32')[0]
        masked = shrink(padded,pos)
        prediction =model.predict(np.array([masked]))[0]
        predictions.append(prediction)#~
    predictions = [p[0] for p in predictions]
    return predictions



def loadModels():
    dirtAlls = pickle.load(open( "releventPOSOutputs.pkl", "rb" ))
    releventPOSOutputs = []
    
    for i in range(len(dirtAlls)):
        cos,pos,maxL = dirtAlls[i]
        mod = load_model('modelOfBin'+str(i)+'.h5')
        releventPOSOutputs.append((cos,pos,mod,maxL))
    perc = load_model('classifier.h5')
    return releventPOSOutputs,perc

def addPredictions(df,releventPOSOutputs):
    df['predictions'] =df.vector.apply(predictionColumn,args=(releventPOSOutputs,))



def flow(filename,numOfRows):    
    df2 = load(filename,numOfRows)
    emb = create_embeddings_index()
    embed(df2,emb)
    releventPOSOutputs2,classifier2 =loadModels()
    addPredictions(df2,releventPOSOutputs2)
    X2 = np.array([row.predictions+[row.lengths] for _,row in df2.iterrows()])
    ev2 =  classifier2.evaluate(X2,df2.Y,batch_size=1)
    print(ev2)
    print(time.time()-start_time)

    
    
import sys
filename = sys.argv[1]#'test.ft.txt'
numOfRows = (int)(sys.argv[2])#100000
flow(filename,numOfRows)




# In[666]:
"""
#For analysis only
df = load(NROWS=100000)
print('loaded data')
emb = create_embeddings_index()
print('created embeding')
embed(df,emb)
print('embeded the data')
releventPOSOutputs,perc =loadModels()
print('load modekls')
addPredictions(df,releventPOSOutputs)
print('get predictions')
store = pd.HDFStore('testdf.h5')
store['testdf'] = df  # save it
print("stored dfs")
X = np.array([row.predictions+[row.lengths] for _,row in df.iterrows()])
print('got relevet data for classifier')
ev =  perc.evaluate(X,df.Y,batch_size=1)
print(ev)
print(time.time()-start_time)


def compare(real,predicted):
    correct = 0
    for i in range(len(real)):
        if i >= len(predicted):
            break
        if round(real[i])==round(predicted[i]):
            correct+=1
    return correct/len(real)
    
    
    
    
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


bin0 = [row for _,row in df.iterrows() if row.lengths <= 39]
predicted0 = [row.predictions[0] for row in bin0]
labels0 = [row.Y for row in bin0]
comp0 = compare(labels0,predicted0)



bin1 = [row for _,row in df.iterrows() if row.lengths <= 61 and row.lengths > 39]
predicted1 = [row.predictions[1] for row in bin1]
labels1 = [row.Y for row in bin1]
comp1 = compare(labels1,predicted1)


bin2 = [row for _,row in df.iterrows() if row.lengths <= 88 and row.lengths > 61]
predicted2 = [row.predictions[2] for row in bin2]
labels2 = [row.Y for row in bin2]
comp2 = compare(labels2,predicted2)

bin3 = [row for _,row in df.iterrows() if row.lengths <= 125 and row.lengths > 88]
predicted3 = [row.predictions[3] for row in bin3]
labels3 = [row.Y for row in bin3]
comp3 = compare(labels3,predicted3)

bin4 = [row for _,row in df.iterrows() if row.lengths > 125]
predicted4 = [row.predictions[4] for row in bin4]
labels4 = [row.Y for row in bin4]
comp4 = compare(labels4,predicted4)


print(comp0,comp1,comp2,comp3,comp4)
"""






"""
first
[1.2512380799002962, 0.80887]
20631.242270946503
0.8341831916902739 0.8046637211585665 0.7945511389012953 0.7772277227722773 0.7913144372617271

TP,           FP,             TN,              FN,            recall,          precision,       F1
42289,     10823,      38598,        8290,           0.83609798532988,   0.7962230757644223,   0.8156734914312718

"""

"""
second
[1.3246909548826873, 0.82212]
14268.735230922699
0.8343720491029273 0.829111438389789 0.7912262418738524 0.8083663366336634 0.7993811812807338


TP,    FP,   TN,     FN,    recall,          precision,       F1
41712, 8921, 40500,  8867,0.82469008877202,0.823810558331523, 0.8242500889222621

"""

