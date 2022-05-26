#Deep Learning has rapidly evolved and is now routinely being used in prediction-based studies in crops improvement. In this study, an effort was made to develop the DeepMap, a unique deep learning enabled python package for genotype to phenotype prediction. It can be easily extended and used in other crops such as Maize, Wheat, Soybean and even Cattle, Loblolly pine and Humans. It uses epistatic interactions for data augmentation and outperforms existing state-of-the-art machine/deep learning models such as Bayesian LASSO, GBLUP, DeepGS, and dualCNN. It is hosted on Python Package Index for ease-of-use to encourage reproducibility with four-line code execution which import libraries, takes genotypic and phenotypic data, invoking model for training/testing data and storing results. The DeepMap can be further improve by adding environmental interactions, incorporating nascent architecture of deep learning such as GANs, autoencoders, transformers etc., and development of graphical user interface would increase the user community with ease-of-use.
"""
Created on Thu Sep  9 11:47:12 2021
@author: Ajay Kumar at International Rice Research Institute, South Asia Hub, Hyderabad, India
"""
#Importing Libraries
import pandas as pd
import math
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.models import Model,Sequential
from keras.utils import to_categorical
from matplotlib import pyplot
from keras.layers import LeakyReLU, BatchNormalization,MaxPool1D, Conv1D, Flatten
from sklearn.model_selection import KFold

from scipy.stats import pearsonr

import sklearn

from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import r2_score
from keras import Input
import matplotlib.pyplot as plt 
import os
os.chdir("D:/_IRRI-SOUTH ASIA/Researches/Research-1 DeepPredict/Datasets/5k_SNPs/PL")

# define and fit model on a training dataset
def DeepGS(trainX, trainy, testX, testy):
    inputs = Input(shape=(7290,3))
    x = Conv1D(18,8,activation='relu') (inputs)
    x= MaxPool1D(4,strides=4) (x)
    x= Dropout(0.1) (x) #Here you can change the drop out values accordingly general, 0.20, 0.10, 0.05
    x= Flatten()(x)
    x= Dense(32,activation='linear')(x)
    x= Dense(1,activation='linear')(x)
    model = Model(inputs=inputs,outputs=x)
    model.compile(loss='mse', optimizer='adam',metrics='mse')
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_mse',patience=2,verbose=1,factor=0.5,min_lr=0.01)
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=200, verbose=1,batch_size=100,callbacks=[learning_rate_reduction])
    model.summary()
    return model,history

geno_pheno = pd.read_csv('PL.csv')
geno = geno_pheno.iloc[:,3:]

geno = geno.to_numpy()

#One-hot encoding function for genotypic dataset
def callme(data):
    nb_classes=3
    def indices_to_one_hot(g,nb_classes):
        targets = np.array(g).reshape(-1)
        return np.eye(nb_classes)[targets]
    arr = np.empty(shape=(data.shape[0],data.shape[1] , nb_classes))
    for i in range(0,data.shape[0]):
        arr[i] = indices_to_one_hot(pd.to_numeric(data[i],downcast='signed'), nb_classes)
    return arr

for i in range(1,2): #Use for loop for more than one trait
    kf = KFold(n_splits=10,shuffle=True)
    result = pd.DataFrame(data=None,columns=('Corr_train','Corr_test','RMSE_train','RMSE_test','r2_train','r2_test'))
    count = 0
    phenotypic_info = geno_pheno.iloc[:,i]
    phenotypic_info = phenotypic_info.to_numpy()
    for train_index, test_index in kf.split(phenotypic_info):
        print("Train:",train_index,"Test:",test_index)
        trainX, testX = geno[train_index,],geno[test_index,]
        trainy,testy = phenotypic_info[train_index],phenotypic_info[test_index,]
        trainX = trainX + 1
        testX = testX + 1
        trainX = callme(trainX)
        testX = callme(testX)
        print("[INFO] training started model...")
        model,history = DeepGS(trainX, trainy, testX, testy)
        #Evaluating model performance
        predicted_train = model.predict([trainX])
        predicted_test = model.predict([testX])
        corr_training = pearsonr(trainy, predicted_train[:,0])[0]
        corr_testing = pearsonr(testy, predicted_test[:,0])[0]
        mse_train = sklearn.metrics.mean_squared_error(trainy, predicted_train)
        rmse_train = math.sqrt(mse_train)
        mse_test = sklearn.metrics.mean_squared_error(testy, predicted_test)
        rmse_test = math.sqrt(mse_test)
        print("This is",count,"Fold")
        print("Training_Correlation:\n",corr_training,"\n\nTesting_Correlation:\n",corr_testing,"\n\nTraining_RMSE\n",rmse_train,"\n\nTesting_RMSE:\n",rmse_test)
        r2_train = r2_score(trainy,predicted_train)
        r2_test = r2_score(testy,predicted_test)
        result.loc[count] = [corr_training,corr_testing,rmse_train,rmse_test,r2_train,r2_test]
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        count = count + 1
        
    mean_result = pd.DataFrame(np.mean(result)) 
    print(mean_result)
    result.to_csv('results_%s.csv'%i)


