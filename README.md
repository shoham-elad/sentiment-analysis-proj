# sentiment-analysis-proj
NLP project
This project is for the course 22933 introduction to NLP in the open university of israel
This project was created by Elad Beran and Elad Shoham

##how to run. 

To run the train please run the following:

./project.sh train \<file\> \<number of rows to take\>

./project.sh test \<file\> \<number of rows to take\>


##data

This project uses glove's database

please download it from https://nlp.stanford.edu/projects/glove/   (we used this version http://nlp.stanford.edu/data/glove.6B.zip) 

make sure the training or test files are in the same folder as the python script

##returns

note that our test script will only output the scalar test loss (less relevant), and the accuracy

There will not be an output file with the predictions of each sentence, and there will be no prints of the recall, f1, precision and such.

In the code there are functions that show these parameters. If needed, one can use them to see the scores.



##Installation/Prerequisites

We use many imports that need installation before hand.

We used anaconda to install and run the imports.

here is the list.

pickle 

numpy
```
conda install -c anaconda numpy 
```

pandas
```
conda install -c anaconda pandas 
```


tensorFlow
```
pip install tensorflow-gpu
or
conda install -c conda-forge tensorflow 
```

keras
```
conda install -c conda-forge keras 
or
sudo pip install keras
```

sklearn
```
pip install -U scikit-learn
or
conda install scikit-learn
```

pyswarms
```
conda install -c auto pyswarm 
```

please note that this is a heavy project for the computer. You might need a small corpus, a lot of running time, and a 64bit computer (for tensorFlow).


## about the folders
in finals there are 4 folders. 
old: our old model with a static alpha
new: our new model with dynamic alpha
baseline50000: the baseline trained on 50,000 sentences
baseline10000: the baseline trained on 10,000 sentences

each folder has other files in it. they are the models and other data needed. This means you can run the test without training.
