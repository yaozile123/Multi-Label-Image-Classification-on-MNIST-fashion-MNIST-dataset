#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd

import torch
import torchvision.datasets as datasets
print(torch.__version__)


# In[8]:


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)


# In[3]:


print(f"train={len(mnist_trainset)},test={len(mnist_testset)}")


# In[4]:


print(type(mnist_trainset))


# In[5]:


#[(img1, label1), (img2,label2),...]

import matplotlib.pyplot as plt
image_sample=[]
label_sample=[]
row=2
col=4
for i in range(0,row*col):
    img,label=mnist_trainset[i]
    #print(item)
    #print(label)
    image_sample.append(img)
    label_sample.append(label)

fig, axes = plt.subplots(nrows=row, ncols=col, figsize=(8, 4))

#for ax, image, label in zip(axes, image_sample, label_sample):
for i in range(0,row):
    for j in range(0,col):
        ax= axes[i][j]
        ax.set_axis_off()
        image=image_sample[i*col+j]
        label=label_sample[i*col+j]
        #ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title('Training: %i' % label)
plt.show()


# In[10]:


a =np.asarray(image_sample[0])#.reshape(-1)
print(a.shape)
print(a)

#[
#[0,0,0,0],
#[(0),(255,255,255),0,0],
#[0,0,0,0]
#]
#3x4x3->3x4x1
#r*w1+g*w2+b*w3->(0,255)
#(20,40,60) * (0.1, 0.5, 0.4) -> (2+20+24)=46


# In[16]:


import random
train_X=[]
train_y=[]
test_X=[]
test_y=[]
train_size=10000
test_size=2000
idx_train = [random.randint(0,len(mnist_trainset)-1) for _ in range(train_size)]
idx_test = [random.randint(0,len(mnist_testset)-1) for _ in range(test_size)]

for i in idx_train:
    img,label=mnist_trainset[i]
    train_X.append(np.asarray(img).reshape(-1))
    train_y.append(label)

for i in idx_test:
    img,label=mnist_testset[i]
    test_X.append(np.asarray(img).ravel())
    test_y.append(label)



train_X = np.asarray(train_X)
test_X=np.asarray(test_X)
print(train_X.shape)
print(train_X[0])


# In[ ]:


#PIL
#error analysis

#[x,y,(r,g,b)]


# In[25]:


from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


#clf = tree.DecisionTreeClassifier()
clf = LogisticRegression()
#clf=make_pipeline(tfidftransformer(), StanrdardScaler(), svm.SVC(kernel='rbf'))

'''
clf = svm.SVC()
scl = StanrdardScaler()
train_X = scl.fit_transform(train_X)
clf.fit(train_X,train_y)
#---------training above

test_X = scl.fit(test_X)
clf.predict(test_X,test_y)
#--------testing above

'''
clf.fit(train_X,train_y)

y_pred = clf.predict(test_X)
y_pred_train = clf.predict(train_X[:1000])


result = classification_report(train_y[:1000], y_pred_train)
print(result)

result = classification_report(test_y, y_pred)
print(result)


# In[26]:


pred_probas=clf.predict_proba(test_X)
print(pred_probas.shape)


# In[27]:


import plotly.graph_objects as go
yactual = np.asarray(test_y)
print()
fig=go.Figure()
for i in range(10):
    items = y_pred[yactual==i]
    correct = items[items==i]
    err = items[items!=i]
#     fig.add_trace(go.Histogram(x=correct,
#                            nbinsx=10,name=f"c{i}",marker_color='blue',opacity=0.5,
#                               ))
    fig.add_trace(go.Histogram(x=err,
                           nbinsx=10,name=f"e{i}",marker_color=i,opacity=0.5))

fig.show()


# In[ ]:




