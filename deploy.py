#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st
from pickle import load
import pickle


# In[2]:


st.write('Titanic survuval Prediction')
st.sidebar.header('Parameter Input')


# In[3]:


def input_feature():
    Sex_male = st.sidebar.selectbox('sex',('0','1'))
    Sex_female = st.sidebar.selectbox('sex',('1','0'))
    data ={'Sex_female':Sex_female,'Sex_male':Sex_male}
    feature = pd.DataFrame(data,index=[0])
    return feature
df = input_feature()
st.subheader('user inputs parameters')
st.write(df)


# In[4]:


Lmodel=load(open("C:/Users/Nitro V 15/Desktop/DATA SCIENCE assignments/Model.pkl",'rb'))
Prediction = Lmodel.predict(df)
predict_prob = Lmodel.predict_proba(df)
st.subheader('predicted results')
st.write('Not Survived' if Prediction[0]==0 else
         'Survived')
st.subheader('prediction probability')
st.write(predict_prob)


# In[ ]:




