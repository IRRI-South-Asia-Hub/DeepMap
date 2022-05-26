#                                    DeepMap: A model with four-line code for prediction-based breeding in crops                                             
#Deep Learning has rapidly evolved and is now routinely being used in prediction-based studies in crops improvement. In this study, an effort was made to develop the DeepMap, a unique deep learning enabled python package for genotype to phenotype prediction. It can be easily extended and used in other crops such as Maize, Wheat, Soybean and even Cattle, Loblolly pine and Humans. It uses epistatic interactions for data augmentation and outperforms existing state-of-the-art machine/deep learning models such as Bayesian LASSO, GBLUP, DeepGS, and dualCNN. It is hosted on Python Package Index for ease-of-use to encourage reproducibility with four-line code execution which import libraries, takes genotypic and phenotypic data, invoking model for training/testing data and storing results. The DeepMap can be further improve by adding environmental interactions, incorporating nascent architecture of deep learning such as GANs, autoencoders, transformers etc., and development of graphical user interface would increase the user community with ease-of-use.
#Instruction to use
"""
Authors:

Ajay Kumar1, Krishna T. Sundaram1, Niranjani Gnanapragasam1, Uma Maheshwar Singh2, K.J. Pranesh2, Challa Venkateshwarlu1, Pronob J. Paul1, Waseem Hussain3, Sankalp Bhosale3,
Ajay Kohli3, Vikas Kumar Singh1*, Pallavi Sinha1* 

1. International Rice Research Institute, South‐Asia Hub (IRRI-SAH), ICRISAT Campus, Patancheru- 502324, Hyderabad, Telangana State, India

2. IRRI South Asia Regional Center (ISARC), NSRTC Campus, G.T. Road, Collectry Farm, P.O. Industrial Estate, Varanasi-221006, Uttar Pradesh State, India

3. International Rice Research Institute (IRRI), Los Banos, Laguna, Philippines.

"""
from DeepMap import main;import pandas as pd;import os #Importing libraries and gDeepPredict package
os.chdir('Enter your path here') #Set your current working directory and also, make sure that your preprocessed data is there.
geno = pd.read_csv('SNP.csv');genotypic_additive = pd.read_csv('additive.csv');genotypic_dominance = pd.read_csv('dominance.csv') #Import your genotypic, phenotypic, additive and dominanc information.
main.main(geno,genotypic_additive,genotypic_dominance,epochs=10000,batch_size=2143,n_splits=10,learning_rate=0.1) #Calling the gDeepPredict function

#Note: You may change the number of epochs, batch-size, number of splits and learning rate in above line to optimize the model.

