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
    
    def fit(x,alpha=0.5):
        fit.counter+=1
        print("current Iteration: " + str(fit.counter))
        n_particles = x.shape[0]
        j = [fit_per_particle(x[i], alpha) for i in range(n_particles)]
        return np.array(j)
    fit.counter = 0

    return fit




def singlePSO(X,Y,Xtest,Ytest,MAX_SENTENCE_LENGTH):
    fit = create_fit(X=X,Y=Y,X_test=Xtest,Y_test=Ytest,dimensions=MAX_SENTENCE_LENGTH,alpha=0.5,NUM_EPOCHS=15)
    print(fit)
    cost,pos = binaryPSO(f=fit,dimensions=MAX_SENTENCE_LENGTH,n_particles=3,print_step=1,iters=3)
    print(cost)
    return cost,pos



def shrink(data,hit):
    shrinkedList = []
    for i in range(0,len(hit)):
        if i>=len(data):
            return shrinkedList
        if hit[i]!=0:
            shrinkedList.append(data[i])
    return shrinkedList

# In[ ]:

def PSOAndModel(df,bin):
    from keras.preprocessing.sequence import pad_sequences
    relevent = df[df.bin==bin]
    vec = relevent.vector
    labe= relevent.Y
    maxSize = max(relevent.lengths)
    
    X = pad_sequences(vec, maxlen=maxSize, value = np.zeros(100), dtype='float32')
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, labe, test_size=0.2,random_state=42)

    cost,pos =singlePSO(Xtrain, Ytrain,Xtest, Ytest, maxSize)
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
        all.append(PSOAndModel(df,i))
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
filename = sys.argv[1]#'test.ft.txt'
numOfRows = (int)(sys.argv[2])#50000
flow(filename,numOfRows)




#val_acc: 0.9842
#45522.34945893288 sexonds



"""
Layer (type)                 Output Shape              Param #   
=================================================================
dense_96 (Dense)             (None, 128)               896       
_________________________________________________________________
dense_97 (Dense)             (None, 64)                8256      
_________________________________________________________________
dense_98 (Dense)             (None, 1)                 65        
_________________________________________________________________
activation_96 (Activation)   (None, 1)                 0         
=================================================================
Total params: 9,217
Trainable params: 9,217
Non-trainable params: 0
_________________________________________________________________
None
Train on 40000 samples, validate on 10000 samples
Epoch 1/100
40000/40000 [==============================] - 16s 399us/step - loss: 0.4327 - acc: 0.8181 - val_loss: 0.2199 - val_acc: 0.9085
Epoch 2/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.2389 - acc: 0.9013 - val_loss: 0.1918 - val_acc: 0.9170
Epoch 3/100
40000/40000 [==============================] - 4s 98us/step - loss: 0.2039 - acc: 0.9133 - val_loss: 0.1737 - val_acc: 0.9220
Epoch 4/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.1693 - acc: 0.9284 - val_loss: 0.1447 - val_acc: 0.9363
Epoch 5/100
40000/40000 [==============================] - 4s 98us/step - loss: 0.1556 - acc: 0.9323 - val_loss: 0.1354 - val_acc: 0.9407
Epoch 6/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.1434 - acc: 0.9376 - val_loss: 0.1317 - val_acc: 0.9400
Epoch 7/100
40000/40000 [==============================] - 4s 98us/step - loss: 0.1358 - acc: 0.9425 - val_loss: 0.1711 - val_acc: 0.9212
Epoch 8/100
40000/40000 [==============================] - 4s 98us/step - loss: 0.1274 - acc: 0.9453 - val_loss: 0.1208 - val_acc: 0.9444
Epoch 9/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.1192 - acc: 0.9491 - val_loss: 0.1041 - val_acc: 0.9521
Epoch 10/100
40000/40000 [==============================] - 4s 98us/step - loss: 0.1018 - acc: 0.9584 - val_loss: 0.1113 - val_acc: 0.9549
Epoch 11/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0912 - acc: 0.9640 - val_loss: 0.0805 - val_acc: 0.9714
Epoch 12/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0835 - acc: 0.9688 - val_loss: 0.0784 - val_acc: 0.9693
Epoch 13/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0773 - acc: 0.9709 - val_loss: 0.0833 - val_acc: 0.9683
Epoch 14/100
40000/40000 [==============================] - 4s 98us/step - loss: 0.0725 - acc: 0.9739 - val_loss: 0.0712 - val_acc: 0.9734
Epoch 15/100
40000/40000 [==============================] - 4s 98us/step - loss: 0.0687 - acc: 0.9752 - val_loss: 0.0619 - val_acc: 0.9778
Epoch 16/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0675 - acc: 0.9757 - val_loss: 0.0597 - val_acc: 0.9780
Epoch 17/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0653 - acc: 0.9766 - val_loss: 0.0559 - val_acc: 0.9798
Epoch 18/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0605 - acc: 0.9791 - val_loss: 0.0714 - val_acc: 0.9717
Epoch 19/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0599 - acc: 0.9789 - val_loss: 0.0614 - val_acc: 0.9741
Epoch 20/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0573 - acc: 0.9799 - val_loss: 0.0575 - val_acc: 0.9790
Epoch 21/100
40000/40000 [==============================] - 4s 98us/step - loss: 0.0559 - acc: 0.9808 - val_loss: 0.0553 - val_acc: 0.9798
Epoch 22/100
40000/40000 [==============================] - 4s 98us/step - loss: 0.0540 - acc: 0.9810 - val_loss: 0.0470 - val_acc: 0.9861
Epoch 23/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0538 - acc: 0.9811 - val_loss: 0.0473 - val_acc: 0.9827
Epoch 24/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0527 - acc: 0.9819 - val_loss: 0.0511 - val_acc: 0.9829
Epoch 25/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0524 - acc: 0.9819 - val_loss: 0.0472 - val_acc: 0.9831
Epoch 26/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0494 - acc: 0.9837 - val_loss: 0.0440 - val_acc: 0.9866
Epoch 27/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0500 - acc: 0.9829 - val_loss: 0.0495 - val_acc: 0.9815
Epoch 28/100
40000/40000 [==============================] - 4s 98us/step - loss: 0.0477 - acc: 0.9832 - val_loss: 0.0554 - val_acc: 0.9780
Epoch 29/100
40000/40000 [==============================] - 4s 98us/step - loss: 0.0468 - acc: 0.9841 - val_loss: 0.0488 - val_acc: 0.9812
Epoch 30/100
40000/40000 [==============================] - 4s 98us/step - loss: 0.0460 - acc: 0.9843 - val_loss: 0.0374 - val_acc: 0.9882
Epoch 31/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0448 - acc: 0.9845 - val_loss: 0.0441 - val_acc: 0.9837
Epoch 32/100
40000/40000 [==============================] - 4s 98us/step - loss: 0.0458 - acc: 0.9842 - val_loss: 0.0705 - val_acc: 0.9705
Epoch 33/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0440 - acc: 0.9843 - val_loss: 0.0357 - val_acc: 0.9876
Epoch 34/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0434 - acc: 0.9854 - val_loss: 0.0406 - val_acc: 0.9867
Epoch 35/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0433 - acc: 0.9846 - val_loss: 0.0375 - val_acc: 0.9883
Epoch 36/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0431 - acc: 0.9851 - val_loss: 0.0374 - val_acc: 0.9878
Epoch 37/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0430 - acc: 0.9851 - val_loss: 0.0371 - val_acc: 0.9877
Epoch 38/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0423 - acc: 0.9861 - val_loss: 0.0545 - val_acc: 0.9786
Epoch 39/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0410 - acc: 0.9859 - val_loss: 0.0377 - val_acc: 0.9868
Epoch 40/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0409 - acc: 0.9869 - val_loss: 0.0372 - val_acc: 0.9893
Epoch 41/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0390 - acc: 0.9866 - val_loss: 0.0375 - val_acc: 0.9870
Epoch 42/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0396 - acc: 0.9864 - val_loss: 0.0336 - val_acc: 0.9897
Epoch 43/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0396 - acc: 0.9866 - val_loss: 0.0376 - val_acc: 0.9885
Epoch 44/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0405 - acc: 0.9855 - val_loss: 0.0409 - val_acc: 0.9856
Epoch 45/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0392 - acc: 0.9865 - val_loss: 0.0322 - val_acc: 0.9899
Epoch 46/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0378 - acc: 0.9864 - val_loss: 0.0363 - val_acc: 0.9878
Epoch 47/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0373 - acc: 0.9871 - val_loss: 0.0324 - val_acc: 0.9905
Epoch 48/100
40000/40000 [==============================] - 4s 102us/step - loss: 0.0386 - acc: 0.9864 - val_loss: 0.0351 - val_acc: 0.9884
Epoch 49/100
40000/40000 [==============================] - 4s 104us/step - loss: 0.0377 - acc: 0.9874 - val_loss: 0.0315 - val_acc: 0.9896
Epoch 50/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0371 - acc: 0.9874 - val_loss: 0.0398 - val_acc: 0.9861
Epoch 51/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0370 - acc: 0.9872 - val_loss: 0.0335 - val_acc: 0.9887
Epoch 52/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0370 - acc: 0.9874 - val_loss: 0.0362 - val_acc: 0.9879
Epoch 53/100
40000/40000 [==============================] - 4s 101us/step - loss: 0.0369 - acc: 0.9871 - val_loss: 0.0337 - val_acc: 0.9883
Epoch 54/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0351 - acc: 0.9883 - val_loss: 0.0360 - val_acc: 0.9867
Epoch 55/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0367 - acc: 0.9870 - val_loss: 0.0370 - val_acc: 0.9879
Epoch 56/100
40000/40000 [==============================] - 4s 101us/step - loss: 0.0357 - acc: 0.9874 - val_loss: 0.0322 - val_acc: 0.9889
Epoch 57/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0345 - acc: 0.9883 - val_loss: 0.0532 - val_acc: 0.9840
Epoch 58/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0346 - acc: 0.9883 - val_loss: 0.0425 - val_acc: 0.9846
Epoch 59/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0361 - acc: 0.9873 - val_loss: 0.0325 - val_acc: 0.9908
Epoch 60/100
40000/40000 [==============================] - 4s 101us/step - loss: 0.0344 - acc: 0.9880 - val_loss: 0.0395 - val_acc: 0.9865
Epoch 61/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0363 - acc: 0.9877 - val_loss: 0.0343 - val_acc: 0.9875
Epoch 62/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0343 - acc: 0.9882 - val_loss: 0.0357 - val_acc: 0.9879
Epoch 63/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0349 - acc: 0.9883 - val_loss: 0.0462 - val_acc: 0.9857
Epoch 64/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0341 - acc: 0.9889 - val_loss: 0.0557 - val_acc: 0.9806
Epoch 65/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0346 - acc: 0.9881 - val_loss: 0.0486 - val_acc: 0.9817
Epoch 66/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0341 - acc: 0.9882 - val_loss: 0.0381 - val_acc: 0.9870
Epoch 67/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0332 - acc: 0.9890 - val_loss: 0.0358 - val_acc: 0.9876
Epoch 68/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0336 - acc: 0.9882 - val_loss: 0.0380 - val_acc: 0.9852
Epoch 69/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0328 - acc: 0.9883 - val_loss: 0.0356 - val_acc: 0.9872
Epoch 70/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0333 - acc: 0.9885 - val_loss: 0.0281 - val_acc: 0.9916
Epoch 71/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0330 - acc: 0.9888 - val_loss: 0.0312 - val_acc: 0.9896
Epoch 72/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0331 - acc: 0.9885 - val_loss: 0.0449 - val_acc: 0.9852
Epoch 73/100
40000/40000 [==============================] - 4s 101us/step - loss: 0.0334 - acc: 0.9888 - val_loss: 0.0369 - val_acc: 0.9898
Epoch 74/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0322 - acc: 0.9893 - val_loss: 0.0325 - val_acc: 0.9899
Epoch 75/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0330 - acc: 0.9884 - val_loss: 0.0330 - val_acc: 0.9885
Epoch 76/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0325 - acc: 0.9894 - val_loss: 0.0360 - val_acc: 0.9872
Epoch 77/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0325 - acc: 0.9889 - val_loss: 0.0424 - val_acc: 0.9841
Epoch 78/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0330 - acc: 0.9886 - val_loss: 0.0273 - val_acc: 0.9918
Epoch 79/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0328 - acc: 0.9886 - val_loss: 0.0297 - val_acc: 0.9902
Epoch 80/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0315 - acc: 0.9891 - val_loss: 0.0364 - val_acc: 0.9887
Epoch 81/100
40000/40000 [==============================] - 4s 101us/step - loss: 0.0328 - acc: 0.9886 - val_loss: 0.0289 - val_acc: 0.9915
Epoch 82/100
40000/40000 [==============================] - 4s 103us/step - loss: 0.0323 - acc: 0.9889 - val_loss: 0.0350 - val_acc: 0.9873
Epoch 83/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0314 - acc: 0.9893 - val_loss: 0.0396 - val_acc: 0.9856
Epoch 84/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0334 - acc: 0.9882 - val_loss: 0.0404 - val_acc: 0.9857
Epoch 85/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0322 - acc: 0.9888 - val_loss: 0.0354 - val_acc: 0.9887
Epoch 86/100
40000/40000 [==============================] - 4s 102us/step - loss: 0.0317 - acc: 0.9886 - val_loss: 0.0326 - val_acc: 0.9889
Epoch 87/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0323 - acc: 0.9887 - val_loss: 0.0406 - val_acc: 0.9846
Epoch 88/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0314 - acc: 0.9887 - val_loss: 0.0293 - val_acc: 0.9912
Epoch 89/100
40000/40000 [==============================] - 4s 103us/step - loss: 0.0323 - acc: 0.9889 - val_loss: 0.0331 - val_acc: 0.9889
Epoch 90/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0318 - acc: 0.9888 - val_loss: 0.0427 - val_acc: 0.9846
Epoch 91/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0315 - acc: 0.9889 - val_loss: 0.0284 - val_acc: 0.9928
Epoch 92/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0312 - acc: 0.9894 - val_loss: 0.0308 - val_acc: 0.9909
Epoch 93/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0306 - acc: 0.9893 - val_loss: 0.0368 - val_acc: 0.9871
Epoch 94/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0308 - acc: 0.9898 - val_loss: 0.0302 - val_acc: 0.9906
Epoch 95/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0319 - acc: 0.9894 - val_loss: 0.0294 - val_acc: 0.9910
Epoch 96/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0329 - acc: 0.9887 - val_loss: 0.0446 - val_acc: 0.9838
Epoch 97/100
40000/40000 [==============================] - 4s 100us/step - loss: 0.0309 - acc: 0.9891 - val_loss: 0.0310 - val_acc: 0.9906
Epoch 98/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0309 - acc: 0.9901 - val_loss: 0.0352 - val_acc: 0.9874
Epoch 99/100
40000/40000 [==============================] - 4s 99us/step - loss: 0.0308 - acc: 0.9890 - val_loss: 0.0324 - val_acc: 0.9890
Epoch 100/100
40000/40000 [==============================] - 4s 101us/step - loss: 0.0299 - acc: 0.9895 - val_loss: 0.0421 - val_acc: 0.9842
45522.34945893288

runfile('C:/Users/Elad-PC/Desktop/sentiment-analysis-proj/testFlow.py', wdir='C:/Users/Elad-PC/Desktop/sentiment-analysis-proj')
100000/100000 [==============================] - 585s 6ms/step  
[1.2512380799002962, 0.80887]
20631.242270946503
0.8341831916902739 0.8046637211585665 0.7945511389012953 0.7772277227722773 0.7913144372617271
"""