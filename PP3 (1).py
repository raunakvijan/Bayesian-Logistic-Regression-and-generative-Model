


# coding: utf-8

# In[1461]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal
from scipy.special import expit
from sklearn.metrics import accuracy_score
import math
from numpy import linalg as LA
from numpy.linalg import inv
import math
import copy
import time


# In[1462]:


A_X = pd.read_csv(r"pp3data/A.csv",header = None)
A_y = pd.read_csv(r"pp3data/labels-A.csv",header = None)

B_X = pd.read_csv(r"pp3data/B.csv",header = None)
B_y = pd.read_csv(r"pp3data/labels-B.csv",header = None)

irls_X = pd.read_csv(r"pp3data/irlstest.csv",header = None)
irls_y = pd.read_csv(r"pp3data/labels-irlstest.csv",header = None)

usps_X = pd.read_csv(r"pp3data/usps.csv",header = None)
usps_y = pd.read_csv(r"pp3data/labels-usps.csv",header = None)


# In[1463]:


def generative_train(X_train, y_train):
    class_0  = X_train[y_train[0] == 0]
    class_1  = X_train[y_train[0] == 1]
    
    mean_0 = np.mean(class_0, axis = 0)
    prior_0 = len(class_0)/len(X_train)

    mean_1 = np.mean(class_1, axis = 0)
    prior_1 = len(class_1)/len(X_train)
    
    cov = X_train.cov()

    return mean_0, mean_1, prior_0, prior_1, cov


# In[1464]:


def generative_test(X_test, mean_0, mean_1, prior_0, prior_1, cov):
    posterior_0 = prior_0*multivariate_normal.pdf(X_test, mean_0, cov)
    posterior_1 = prior_1*multivariate_normal.pdf(X_test, mean_1, cov)

    b = np.log((posterior_1/posterior_0))
    prob = [1 if a > 0.5 else 0 for a in expit(b)]
    return prob


# In[1465]:


def bayasian_logistic_reg(X, t):
    time_vector = []
    start_time = time.time()
    alpha = 0.130
    X.reset_index(inplace = True, drop = True)
    col = 30
    
    X['bias'] = col

    w_old = np.zeros((np.shape(X)[1],1))
    time_vector.append((w_old, time.time() - start_time))

    b = np.dot(X, w_old)
    y = expit(b)
    a =   y*(1 - y)
    R = np.diagflat(a)
    mat = np.dot(np.dot((X.T),R),(X))
    w_new = w_old - (np.dot(inv(alpha*np.identity(np.shape(mat)[0]) + mat), (np.dot(X.T, y - t) + alpha*w_old)))
    time_vector.append((w_new, time.time() - start_time))
    n = 1
    while(((LA.norm(w_new - w_old) / LA.norm(w_old)) > 10**(-3)) and (n <= 100)):
        w_old = w_new
        b = np.dot(X, w_old)
        y = expit(b)
        a =   y*(1 - y)
        R = np.diagflat(a)
        mat = np.dot(np.dot((X.T),R),X)
        w_new = w_old - (np.dot(inv(alpha*np.identity(np.shape(mat)[0]) + mat), (np.dot(X.T, y - t) + alpha*w_old)))
        time_vector.append((w_new, time.time() - start_time))
        n+=1

    return w_new, time_vector


# In[1466]:


for i in range(30):
    A_X_train, A_X_test, A_y_train, A_y_test = train_test_split(A_X, A_y, test_size=(1/3))
    mean_0, mean_1, prior_0, prior_1, cov = generative_train(A_X_train, A_y_train)
    res = generative_test(A_X_test, mean_0, mean_1, prior_0, prior_1, cov)
    #print(accuracy_score(res, A_y_test))

    B_X_train, B_X_test, B_y_train, B_y_test  = train_test_split(B_X, B_y, test_size=(1/3))
    mean_0, mean_1, prior_0, prior_1, cov = generative_train(B_X_train, B_y_train)
    res = generative_test(B_X_test, mean_0, mean_1, prior_0, prior_1, cov)
    #print(accuracy_score(res, B_y_test))

# In[1467]:


w, _ = bayasian_logistic_reg(irls_X, irls_y)
#print(w)



# In[1468]:


def gradient_ascent(X, t):
    time_vector = []
    start_time = time.time()
    alpha = 0.1
    X.reset_index(inplace = True, drop = True)
    col = pd.DataFrame(np.ones((np.shape(X)[0],1)))
    X['bias'] = col
    
    w_old = np.zeros((np.shape(X)[1],1))
    time_vector.append((w_old, time.time() - start_time))
    b = np.dot(X, w_old)
    y = expit(b)
    w_new = w_old - (10**(-3)*(np.dot(X.T, y - t) + alpha*w_old))
    time_vector.append((w_new, time.time() - start_time))
    n = 1
    while(((LA.norm(w_new - w_old) / LA.norm(w_old)) > 10**(-3)) and (n <= 6000)):
        w_old = w_new
        b = np.dot(X, w_old)
        y = expit(b)
        w_new = w_old - (10**(-3)*(np.dot(X.T, y - t) + alpha*w_old))
        if(n%10 == 0):
            time_vector.append((w_new, time.time() - start_time))
        n+=1

    return w_new, time_vector


# In[1469]:


index = math.ceil(len(A_X)/3)

w, time_vector1 = gradient_ascent(copy.deepcopy(A_X[index::]),copy.deepcopy( A_y[index::]))
w, time_vector2 = gradient_ascent(copy.deepcopy(A_X[index::]),copy.deepcopy( A_y[index::]))
w, time_vector3 = gradient_ascent(copy.deepcopy(A_X[index::]),copy.deepcopy( A_y[index::]))
#print(time_vector1[0][0])


A_test = copy.deepcopy(A_X[0:index])
A_test.reset_index(inplace = True, drop = True)
col = pd.DataFrame(np.ones((np.shape(A_test)[0],1)))
A_test['bias'] = col

time_vals = []
error = []

for i in range(len(time_vector1)):
    y = expit(np.dot(A_test, time_vector1[i][0]))
    prob = [1 if a > 0.5 else 0 for a in y]
    time_vals.append((time_vector1[i][1] + time_vector2[i][1] + time_vector3[i][1])/3) 
    error.append(1 - accuracy_score(prob, A_y[0:index]))


fig, ax = plt.subplots(nrows=1, ncols= 1)

fig.tight_layout(pad = 0.5, w_pad = 3, h_pad = 3)
ax.plot(time_vals, error)
ax.set_title("MSE")
#plt.rcParams['figure.dpi'] = 75 # default for me was 75
ax.set_ylabel('Error')
ax.set_xlabel('Time')
ax.grid(True )



    
w, time_vector1 = bayasian_logistic_reg(copy.deepcopy(A_X[index::]),copy.deepcopy( A_y[index::]))
w, time_vector2 = bayasian_logistic_reg(copy.deepcopy(A_X[index::]),copy.deepcopy( A_y[index::]))
w, time_vector3 = bayasian_logistic_reg(copy.deepcopy(A_X[index::]),copy.deepcopy( A_y[index::]))

time_vals = []
error = []

for i in range(len(time_vector1)):
    y = expit(np.dot(A_test, time_vector1[i][0]))
    prob = [1 if a > 0.5 else 0 for a in y]
    time_vals.append((time_vector1[i][1] + time_vector2[i][1] + time_vector3[i][1])/3) 
    error.append(1 - accuracy_score(prob, A_y[0:index]))

ax.plot(time_vals, error)



# In[ ]:


index = math.ceil(len(usps_X)/3)

w, time_vector1 = gradient_ascent(copy.deepcopy(usps_X[index::]),copy.deepcopy( usps_y[index::]))
w, time_vector2 = gradient_ascent(copy.deepcopy(usps_X[index::]),copy.deepcopy( usps_y[index::]))
w, time_vector3 = gradient_ascent(copy.deepcopy(usps_X[index::]),copy.deepcopy( usps_y[index::]))


usps_test = copy.deepcopy(usps_X[0:index])
usps_test.reset_index(inplace = True, drop = True)
col = pd.DataFrame(np.ones((np.shape(usps_test)[0],1)))
usps_test['bias'] = col

time_vals = []
error = []

for i in range(len(time_vector1)):
    y = expit(np.dot(usps_test, time_vector1[i][0]))
    prob = [1 if a > 0.5 else 0 for a in y]
    time_vals.append((time_vector1[i][1] + time_vector2[i][1] + time_vector3[i][1])/3) 
    error.append((1 - accuracy_score(prob, usps_y[0:index])))


fig, ax = plt.subplots(nrows=1, ncols= 1)

fig.tight_layout(pad = 0.5, w_pad = 3, h_pad = 3)
ax.plot(time_vals, error)
ax.set_title("MSE")
plt.rcParams['figure.dpi'] = 75 # default for me was 75
ax.set_ylabel('Error')
ax.set_xlabel('Time')
ax.grid(True )


    
w, time_vector1 = bayasian_logistic_reg(copy.deepcopy(usps_X[index::]),copy.deepcopy( usps_y[index::]))
w, time_vector2 = bayasian_logistic_reg(copy.deepcopy(usps_X[index::]),copy.deepcopy( usps_y[index::]))
w, time_vector3 = bayasian_logistic_reg(copy.deepcopy(usps_X[index::]),copy.deepcopy( usps_y[index::]))

time_vals = []
error = []

for i in range(len(time_vector1)):
    y = expit(np.dot(usps_test, time_vector1[i][0]))
    prob = [1 if a > 0.5 else 0 for a in y]
    time_vals.append((time_vector1[i][1] + time_vector2[i][1] + time_vector3[i][1])/3) 
    error.append((1 - accuracy_score(prob, usps_y[0:index])))

ax.plot(time_vals, error)