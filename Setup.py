#Deep Learning has rapidly evolved and is now routinely being used in prediction-based studies in crops improvement. In this study, an effort was made to develop the DeepMap, a unique deep learning enabled python package for genotype to phenotype prediction. It can be easily extended and used in other crops such as Maize, Wheat, Soybean and even Cattle, Loblolly pine and Humans. It uses epistatic interactions for data augmentation and outperforms existing state-of-the-art machine/deep learning models such as Bayesian LASSO, GBLUP, DeepGS, and dualCNN. It is hosted on Python Package Index for ease-of-use to encourage reproducibility with four-line code execution which import libraries, takes genotypic and phenotypic data, invoking model for training/testing data and storing results. The DeepMap can be further improve by adding environmental interactions, incorporating nascent architecture of deep learning such as GANs, autoencoders, transformers etc., and development of graphical user interface would increase the user community with ease-of-use.
#Instruction to use
"""
Created on Thu Sep  9 11:47:12 2021
@author: Ajay Kumar, IRRI South Asia Hub
"""
from DeepMap import main;import pandas as pd;import os #Importing libraries and gDeepPredict package
os.chdir('D:/_IRRI-SOUTH ASIA/Paper-1_DeepMap/Paper_writing/R 11/Github_DeepMap/Case study 1 and 2 IRRI-SAH and ISARC/Dataset/Days to Flowering (DTF)/HY') #Set your current working directory and also, make sure that your preprocessed data is there.
geno = pd.read_csv('SNP.csv');genotypic_additive = pd.read_csv('additive.csv');genotypic_dominance = pd.read_csv('dominance_new.csv') #Import your genotypic, phenotypic, additive and dominanc information.
main.main(geno,genotypic_additive,genotypic_dominance,epochs=10000,batch_size=2143,n_splits=10,learning_rate=0.1) #Calling the gDeepPredict function

#Note: You may change the number of epochs, batch-size, number of splits and learning rate in above line to optimize the model.

