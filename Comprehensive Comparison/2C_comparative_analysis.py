#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 


# ## Import Data and Preprocess

# In[2]:


df = pd.read_csv("dataset_comb.csv")


# In[3]:


df


# In[4]:


df = df.drop(['id'],axis=1)
df['Class'] = [0 if x == 'jasmine' else 1 for x in df['Class']]


# In[5]:


X = df.iloc[:,0:-1]
Y = df.iloc[:,-1]


# ## Normalize data

# In[6]:


for col in X.columns:
    if col!='Class':
        max_val = X[col].max()
        min_val = X[col].min()
        for val in X[col]:
            norm_val = (max_val - val)/(max_val - min_val)
            X[col] = X[col].replace(val, norm_val)


# In[7]:


X = X.values
Y = Y.values


# ## Import Classifiers from Sklearn

# In[8]:


from sklearn.model_selection import train_test_split,KFold 


# In[9]:


cv = KFold(n_splits = 7,random_state = 3,shuffle = True)


# In[10]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression 
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ## Initializing Classifier Objects

# In[18]:


fc = LinearDiscriminantAnalysis()
pc = Perceptron()
nb = GaussianNB()
lr = LogisticRegression()
ann = MLPClassifier()
svm = SVC()


fc_score_train = []
pc_score_train = []
nb_score_train = []
lr_score_train = []
ann_score_train = []
svm_score_train = []

fc_score_test = []
pc_score_test = []
nb_score_test = []
lr_score_test = []
ann_score_test = []
svm_score_test = []


# ## Train, Test and Calculate Accuracy

# In[19]:


for train_index,test_index in cv.split(X):
   
    X_train,X_test = X[train_index], X[test_index]
    Y_train,Y_test = Y[train_index], Y[test_index]
    
    #train all the classifiers with train data
    fc.fit(X_train,Y_train)
    pc.fit(X_train,Y_train)
    nb.fit(X_train,Y_train)
    lr.fit(X_train,Y_train)
    ann.fit(X_train,Y_train)
    svm.fit(X_train,Y_train)
    
    #preditct train data 
    Y_pred_fc_train = fc.predict(X_train)
    Y_pred_pc_train = pc.predict(X_train)
    Y_pred_nb_train = nb.predict(X_train)
    Y_pred_lr_train = lr.predict(X_train)
    Y_pred_ann_train = ann.predict(X_train)
    Y_pred_svm_train = svm.predict(X_train)
    
    #calculate train accuraciies and appento train_accuracy
    fc_score_train.append(accuracy_score(Y_train,Y_pred_fc_train))
    pc_score_train.append(accuracy_score(Y_train,Y_pred_pc_train))
    nb_score_train.append(accuracy_score(Y_train,Y_pred_nb_train))
    lr_score_train.append(accuracy_score(Y_train,Y_pred_lr_train))
    ann_score_train.append(accuracy_score(Y_train,Y_pred_ann_train))
    svm_score_train.append(accuracy_score(Y_train,Y_pred_svm_train))
    
    #predict test data
    Y_pred_fc_test = fc.predict(X_test)
    Y_pred_pc_test = pc.predict(X_test)
    Y_pred_nb_test = nb.predict(X_test)
    Y_pred_lr_test = lr.predict(X_test)
    Y_pred_ann_test = ann.predict(X_test)
    Y_pred_svm_test = svm.predict(X_test)

    #calculate test accuraciies and appento test_accuracy
    fc_score_test.append(accuracy_score(Y_test,Y_pred_fc_test))
    pc_score_test.append(accuracy_score(Y_test,Y_pred_pc_test))
    nb_score_test.append(accuracy_score(Y_test,Y_pred_nb_test))
    lr_score_test.append(accuracy_score(Y_test,Y_pred_lr_test))
    ann_score_test.append(accuracy_score(Y_test,Y_pred_ann_test))
    svm_score_test.append(accuracy_score(Y_test,Y_pred_svm_test))


# ## Print Accuracies

# In[20]:


names = ['Linear Discriminant']
test_dict = {'Linear Discriminant': fc_score_test , 'Perceptron' : pc_score_test, 'Naive-Bayes' : nb_score_test, 'Logistic Regression' : lr_score_test,'ANN' : ann_score_test,'SVM' : svm_score_test}
test_acc = pd.DataFrame(test_dict)
train_dict = {'Linear Discriminant': fc_score_train , 'Perceptron' : pc_score_train, 'Naive-Bayes' : nb_score_train, 'Logistic Regression' : lr_score_train,'ANN' : ann_score_train,'SVM' : svm_score_train}
train_acc = pd.DataFrame(train_dict)


# In[21]:


train_acc


# In[22]:


test_acc


# ## Box Plots
# 

# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[24]:


accuracy = list([fc_score_test,pc_score_test,nb_score_test,lr_score_test,ann_score_test,svm_score_test])
fig,ax = plt.subplots(figsize=(20,10))
ax.boxplot(accuracy)
ax.set_ylim([0.94,1.0])
ax.set_title('Box Plots')
ax.set_xlabel('Classifcation Models')
ax.set_ylabel('Accuracy')
xticklabels = ["Fischer Linear Discriminant","Perceptron","Naive-Bayes","Logistic Regression","ANN","SVM"]
ax.set_xticklabels(xticklabels)
ax.yaxis.grid(True)
plt.show()


# In[ ]:





# In[ ]:




