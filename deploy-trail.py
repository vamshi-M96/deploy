#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import streamlit as st
import pickle
from pickle import load




# In[2]:


st.write('Titanic survuval Prediction')
st.sidebar.header('Parameter Input')


# In[3]:


def input_feature():
    Age = st.sidebar.number_input('Insert Age')
    Parch =st.sidebar.number_input('Insert Parch')
    Sex_male = st.sidebar.selectbox('sex',('0','1'))
    Sex_female = st.sidebar.selectbox('sex',('1','0'))
    Embarked_C = st.sidebar.selectbox('Embarked_c',('0','1'))
    Embarked_Q = st.sidebar.selectbox('Embarked_Q',('0','1'))
    Embarked_S = st.sidebar.selectbox('Embarked_S',('0','1'))
    data ={'Age':Age,'Parch':Parch,'Sex_female':Sex_female,'Sex_male':Sex_male,'Embarked_C':Embarked_C,'Embarked_Q':Embarked_Q,'Embarked_S':Embarked_S}
    feature = pd.DataFrame(data,index=[0])
    return feature
df = input_feature()
st.subheader('user inputs parameters')
st.write(df)


# In[5]:


lmodel=load(open("survived.pkl",'rt'))
Prediction = lmodel.predict(df)
predict_prob = lmodel.predict_proba(df)
st.subheader('predicted results')
st.write('Not Survived' if Prediction[0]==0 else
         'Survived')
st.subheader('prediction probability')
st.write(predict_prob)


# In[ ]:





# In[ ]:




