import sys

import numpy
import numpy as np
import matplotlib as plt
import pandas as pd


df= pd.read_csv("train.csv",header=0)
print(df.columns.values)
pd.set_option('display.max_columns', None)
df1 = df[['Survived', 'Pclass', 'Age', 'Sex','SibSp','Parch','Fare']]
# df1['Sex'] = df1['Sex'].astype('category')
df1['Sex']=df1['Sex'].astype('category').cat.codes
# print(df1)
x_train=df1[['Pclass', 'Age', 'Sex','SibSp','Parch','Fare']]
y_train=df1[['Survived']]
x_train=x_train.to_numpy()
y_train=y_train.to_numpy()
#np.set_printoptions(threshold = sys.maxsize)
# print(len(x_train))
m=x_train.shape[1]
w=np.arange(m)
b=5
a=0.5
x_train=x_train[~np.isnan(x_train)]
# print(w)
def cal_multi(x,w,b):
    sum=0
    sum+=np.dot(x,w)
    return sum+b

def cal_cost(x,y,w,b):
    m=x.shape[0]
    cost=0
    for i in range(m):
        f_wb=cal_multi(x[i],w,b)**2
        cost+=f_wb
    return cost//(2*m)

def cal_par(x,y,w,b):

    J_wb=cal_cost(x,y,w,b)
    m,n=x.shape
    print(m,n)
    d_dw=d_db=0
    for i in range(m):
        f_wb=cal_multi(x[i],w,b)
        d_db+=f_wb
        for j in range(n):
            d_dw+=(f_wb*x[i,j])
    return d_dw,d_db

def cal_final(x,y,w,b,a):
    d_dw,d_db=cal_par(x,y,w,b)
    m=x.shape[0]
    w=w
    b=b
    for i in range(m):
        w=w-(a*d_dw)
        b=b-(a*d_db)
    return w,b

print(cal_final(x_train,y_train,w,b,a))
# print(cal_cost(x_train,y_train,w,b))