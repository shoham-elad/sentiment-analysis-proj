# In[1]:

import time
import numpy as np
import pandas as pd
import warnings
import pyswarms as ps
from sklearn.model_selection import train_test_split
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# In[2]:


def load(fileName='train.ft.txt',NROWS=None):
    from keras.preprocessing.text import text_to_word_sequence 
    df = pd.read_csv(fileName,header=None, delimiter = "\n", nrows=NROWS)
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



# In[3]:




def maxAndMin(lengths):
    maxLength = max(lengths)
    minLength = min(lengths)
    return maxLength , minLength

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




def maxAndMinSeenLengths(hit):
    maxi = 0
    maxLength=0
    for leng in hit:
        if hit[leng]>maxi:
            maxi = hit[leng]
            maxLength = leng
    
    mini = 50000000
    minLength = 50000000
    for leng in hit:
        if hit[leng]<mini:
            mini = hit[leng]
            minLength = leng
    return [(maxi,maxLength),(mini,minLength)]

def getUniqueWords(df):
    X=df.X
    uniqueWords ={}
    for sen in X:
        arraied= sen
        for word in arraied:
            if word not in uniqueWords:
                uniqueWords[word]=0
            uniqueWords[word]+=1
    return uniqueWords
        


# In[4]:

def binify(hit,inf):
    bins= []
    binIndex=[[]]
    count=0
    nums = list(hit)
    nums.sort()
    for i in range(len(nums)):#for all the lengths of sentences seen
        if count >=inf: # create new bin
            bins.append(count)
            count=0
            binIndex.append([])
        #update current bin
        count += hit[nums[i]]
        binIndex[-1].append(nums[i])
    #end case for var that is never used so dont ask to many questions
    bins.append(count)
    print(bin)
    return binIndex 






# In[5]:

def addBinsInDF(df,binIndex):
    lengthToBin = {}
    for i in range(len(binIndex)):
        for num in binIndex[i]:
            lengthToBin[num]=i
    df['bin']= [lengthToBin[row.lengths] for _,row in df.iterrows()]


# In[6]:

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

def embed(df):
    zeros = np.zeros(100)
    emb = create_embeddings_index()
    df['vector']= df.X.apply(lambda x: [emb[word] if word in emb else zeros  for word in x])



# In[7]:





def RNNCustom(BATCH_SIZE = 32, input_dim = 100,Hidden_Layer_Size= 64,timesteps=8, RLAYER=None,
        ACTIVATION="sigmoid",LOSS="binary_crossentropy",OPTIMIZER='nadam',METRICS=["accuracy"]):
    from keras.layers.core import Activation, Dense
    from keras.layers.recurrent import LSTM
    from keras.models import Sequential
    RLAYER = LSTM if RLAYER is None else RLAYER
    model = Sequential()
    model.add(RLAYER(Hidden_Layer_Size,input_shape=(timesteps,input_dim)))
    model.add(Dense(1))
    model.add(Activation(ACTIVATION))
    model.compile(loss=LOSS, optimizer=OPTIMIZER,metrics=METRICS)
    return model


def binaryPSO(f,dimensions,n_particles,print_step,iters,options= {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 2, 'p':2},verbose=2):
    optimizer = ps.discrete.BinaryPSO(n_particles, dimensions=dimensions, options=options)
    cost, pos = optimizer.optimize(f, print_step=print_step, iters=iters, verbose=verbose)
    return cost,pos


def create_fit(X,Y,X_test,Y_test,dimensions,alpha=0.5,NUM_EPOCHS = 10):
    
    def fit_per_particle(m,alpha):
        from sklearn.metrics import accuracy_score
        if np.count_nonzero(m) == 0:
            X_subset = X
            X_test_subset = X_test
        else:
            X_subset = X[:,m==1]
            X_test_subset = X_test[:,m==1]
        model = RNNCustom(timesteps=m.sum())
        model.fit(X_subset, Y,epochs=NUM_EPOCHS,validation_data=(X_test_subset, Y_test))
        P=accuracy_score(model.predict_classes(X_test_subset),Y_test)
        j = (alpha * (1.0 - P)
            + (1.0 - alpha) * (1 - (X_subset.shape[1] / dimensions)))
        return j
    
    def fit(x,alpha=alpha):
        print("alpha:")
        print(alpha)
        fit.counter+=1
        print("current Iteration: " + str(fit.counter))
        n_particles = x.shape[0]
        j = [fit_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)
    fit.counter = 0

    return fit


#due to an empiric observation that bins with longer sentences had lower accuracy
#we chose to change alpha with the bin length in order to choose accuracy over dimensionality reduction
def computeAlpha(binIndex,numOfBins):
    return 0.5*(1+binIndex*(1/numOfBins))

def singlePSO(X,Y,Xtest,Ytest,MAX_SENTENCE_LENGTH,bin,numOfBins):
    fit = create_fit(X=X,Y=Y,X_test=Xtest,Y_test=Ytest,dimensions=MAX_SENTENCE_LENGTH,alpha=computeAlpha(bin,numOfBins),NUM_EPOCHS=15)
    print(fit)
    cost,pos = binaryPSO(f=fit,dimensions=MAX_SENTENCE_LENGTH,n_particles=3,print_step=1,iters=3)
    print(cost)
    return cost,pos



def shrink(data,hit):
    shrinkedList = []
    for i in range(0,len(hit)):
        if i>len(data):
            return shrinkedList
        if hit[i]!=0:
            shrinkedList.append(data[i])
    return shrinkedList

# In[ ]:

def PSOAndModel(df,bin,numOfBins):
    from keras.preprocessing.sequence import pad_sequences
    relevent = df[df.bin==bin]
    vec = relevent.vector
    labe= relevent.Y
    maxSize = max(relevent.lengths)
    
    X = pad_sequences(vec, maxlen=maxSize, value = np.zeros(100), dtype='float32')
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, labe, test_size=0.2,random_state=42)

    cost,pos =singlePSO(Xtrain, Ytrain,Xtest, Ytest, maxSize,bin,numOfBins)
    shrinked =np.array([shrink(x,pos) for x in X])

    model = RNNCustom(timesteps=len(shrinked[0]))
    print(shrinked)
    model.fit(shrinked , labe,epochs=20)#3
    
    return cost,pos,model,maxSize


# In[1]:


def allPSO(df,binInsex):
    all=[]
    for i in range (len(binInsex)):
        print('starting bin number'+str(i))
        all.append(PSOAndModel(df,i,len(binInsex)))
        print('ended bin number'+str(i))
    return all





def predictionColumn(vector,alls):
    from keras.preprocessing.sequence import pad_sequences
    predictions=[] #~
    for cost,pos,model,origMax in alls:
        padded = pad_sequences([vector], maxlen=origMax,value = np.zeros(100), dtype='float32')[0]
        masked = shrink(padded,pos)
        prediction =model.predict(np.array([masked]))[0]
        predictions.append(prediction)#~
    predictions = [p[0] for p in predictions]
    return predictions



def trainClassifier(InputTrain, LabelsTrain,
        FLayerSize = 128,BATCH_SIZE = 64,NUM_EPOCHS = 100,
        ACTIVATION="sigmoid",LOSS="binary_crossentropy",OPTIMIZER='nadam',METRICS=["accuracy"]):    
    from keras.layers.core import Activation, Dense
    from keras.models import Sequential
    print(FLayerSize)
    model = Sequential([Dense(128,input_shape=(FLayerSize,),activation='relu'),Dense(64,activation='relu'),Dense(1),Activation(ACTIVATION)])
    model.compile(loss=LOSS, optimizer=OPTIMIZER,metrics=METRICS)
    print(model.summary())
    model.fit(InputTrain, LabelsTrain, batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,validation_split=0.2)
    return model





def createClassifier(df,allsLength):
    X = np.array([row.predictions+[row.lengths] for _,row in df.iterrows()])
    inputLength = allsLength+1
    classifier =  trainClassifier(X, df.Y,inputLength)
    return classifier

def save(releventPOSOutputs,classifier,df):
    import pickle
    store = pd.HDFStore('df.h5')
    store['df'] = df  # save it
    allsSaved = []
    for i in range(len(releventPOSOutputs)):
        cos,pos,model,max = releventPOSOutputs[i]
        model.save('modelOfBin'+str(i)+'.h5')
        allsSaved.append((cos,pos,max))    
    pickle.dump( allsSaved, open( "releventPOSOutputs.pkl", "wb" ) )
    classifier.save('classifier.h5')

def addPredictions(df,releventPOSOutputs):
    df['predictions'] =df.vector.apply(predictionColumn,args=(releventPOSOutputs,))

def flow(filename,numOfRows):
    start_time = time.time()
    df = load(filename,NROWS=numOfRows)
    lengths = df.lengths #the lengths of sentence
    hit,ind = histMapAndIndexes(lengths)# hit := sentence length frequencies 
    binIndex = binify(hit,10000)#wanted bin size
    addBinsInDF(df,binIndex)
    embed(df)
    releventPOSOutputs =allPSO(df,binIndex)# [(cost, pos , model, maxOfBin),.....]
    addPredictions(df,releventPOSOutputs)
    classifier = createClassifier(df,len(releventPOSOutputs))
    save(releventPOSOutputs,classifier,df)
    print(time.time()-start_time)


import sys
filename =sys.argv[1] # 'train.ft.txt'
numOfRows = (int)(sys.argv[2])#50000
flow(filename,numOfRows )


