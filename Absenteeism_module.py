#!/usr/bin/env python
# coding: utf-8

# In[1]:


# load the library needed

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# define the customscaler
class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.scaler = StandardScaler()
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    def fit(self, X, y = None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y=None):
        init_col_order = X.columns
        x_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        x_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([x_scaled, x_not_scaled], axis=1)[init_col_order]
    
# create the absenteeism model that we could use later on
class absenteeism_model():
    
    def __init__(self,model_file):
        with open('final_model', 'rb') as model_file:
            self.pipe = pickle.load(model_file)
            self.data=None
    
    def load_and_clear_data(self, data_file):
        
        df = pd.read_csv(data_file, delimiter=',')
        self.df_without_predictions=df.copy()
        df.drop('ID', axis=1, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['Month'] = df['Date'].apply(lambda x: x.month)
        df['day_of_week'] = df['Date'].apply(lambda x: x.dayofweek)
        df.drop('Date', axis=1, inplace=True)
        def combine_education(x):
            if x == 1:
                return 0
            else:
                return 1
        
        df['Education']=df['Education'].apply(combine_education)
        
        def absence_reason(x):
            if x in np.arange(1,15):
                return '1'
            elif x in np.arange(15, 18):
                return '2'
            elif x in np.arange(18, 22):
                return '3'
            elif x in np.arange(22, 29):
                return '4'
        
        df['Reason for Absence'] = df['Reason for Absence'].apply(absence_reason)
        df = pd.get_dummies(df)
        df.drop(['Absenteeism Time in Hours', 'Month', 'Age'], axis=1, inplace=True)
        self.preprocessed_data=df.copy()
        
    def predict_probability(self):
        if(self.preprocessed_data is not None):
            pred_prob = self.pipe.predict_proba(self.preprocessed_data)[:,1]
            return pre_prob
        
    def predict_output(self):
        if(self.preprocessed_data is not None):
            pred_output = self.pipe.predict(self.preprocessed_data)
            return pred_output
    
    def predicted_output(self):
        if(self.preprocessed_data is not None):
            self.preprocessed_data['Probability'] = self.pipe.predict_proba(self.preprocessed_data)[:,1]
            self.preprocessed_data['prediction'] = self.pipe.predict(self.preprocessed_data.iloc[:,:-1])
            return self.preprocessed_data


# In[ ]:




