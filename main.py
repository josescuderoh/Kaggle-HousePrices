# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:07:57 2019

@author: JoseEscudero
"""

# Import required packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import LabelEncoder

#Define plotting style
plt.style.use('ggplot')

#Import dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Shape of files
train.shape
test.shape

# =============================================================================
# Data cleansing
# =============================================================================

#Remove outliers

#GarageArea
plt.scatter(x = train.GarageArea, y= train.SalePrice , color = 'blue')
plt.show()
train.drop(train[(train.GarageArea>1200) & (train.SalePrice<300000)].index, inplace=True)

#GrLivArea
plt.scatter(x = train.GrLivArea, y= train.SalePrice , color = 'blue')
plt.show()
train.drop(train[(train.GrLivArea > 4000)&(train.SalePrice < 300000)].index, inplace = True)


#Check for NAs

#Merge dataframes

all_data = pd.concat([train,test], sort = False, ignore_index=True)

null_count = all_data.isnull().sum()
null_count[(null_count>0)].sort_values(ascending = False)

#Function to clean dataframe
def clean_data(df):
    
    #Substitute LotFrontageValue for median of neighboorhood
    df['LotFrontage'] = df.groupby(['Neighborhood'])[['LotFrontage']].transform(
            lambda x: x.fillna(x.median()))
    
    #Substitue others for None according to data_description
    cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 
            'GarageFinish', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 'GarageQual',
            'GarageYrBlt', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'MasVnrType',
            'BsmtFullBath', 'BsmtHalfBath']
    for col in cols:
        df[col].fillna(value = 'None', inplace=True) 
        
    #Substitue others for 0 according to data_description
    cols = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','GarageCars',
            'GarageArea', 'TotalBsmtSF']
    for col in cols:
        df[col].fillna(value = 0, inplace=True)
    
    #Substitue others for mode according to data_description
    cols = ['Electrical', 'MSZoning', 'Utilities', 'Functional', 'Exterior2nd',
            'Exterior1st', 'SaleType', 'KitchenQual']
    for col in cols:
        df[col].fillna(value = df[col].mode()[0], inplace=True)
    

clean_data(all_data)

#Check for no missing data
null_count= all_data.isnull().sum()
null_count[(null_count>0)].sort_values(ascending = False)

# =============================================================================
# Feature engineering 
# =============================================================================

# Encoding categorical features

nominal = ["MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour",
                "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1",
                "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
                "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "BsmtFinType1",
                "BsmtFinType2", "Heating", "CentralAir", "Electrical",
                "Functional", "GarageType","GarageFinish", "PavedDrive", "Fence", 
                "MiscFeature", "SaleType", "SaleCondition"]

test_encoding = copy.copy(all_data)

#Use label encoding for the nominal columns

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# Use mapping for ordinal columns

ordinal = ["ExterCond", "ExterQual","BsmtQual", "BsmtCond", "BsmtExposure", 
           "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]

def map_ordinal(df):
    df['ExterCond'] = df['ExterCond'].map({
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1})
    df['ExterQual'] = df['ExterQual'].map({
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1})
    df['BsmtQual'] = df['BsmtQual'].map({
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "NA": 0})
    df['BsmtCond'] = df['BsmtCond'].map({
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "NA": 0})
    df['BsmtExposure'] = df['BsmtExposure'].map({
            "Gd": 4,
            "Av": 3,
            "Mn": 2,
            "No": 1,
            "NA": 0})
    df['HeatingQC'] = df['HeatingQC'].map({
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1})
    df['KitchenQual'] = df['KitchenQual'].map({
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1})
    df['FireplaceQu'] = df['FireplaceQu'].map({
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "NA": 0})
    df['GarageQual'] = df['GarageQual'].map({
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "NA": 0})
    df['GarageCond'] = df['GarageCond'].map({
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "NA": 0})
    df['PoolQC'] = df['PoolQC'].map({
            "Ex": 4,
            "Gd": 3,
            "TA": 2,
            "Fa": 1,
            "NA": 0})
    
    print("Encoding terminated")
    
    return df
    
def encode_categorical(df, nominal):
    df = MultiColumnLabelEncoder(columns=nominal).fit_transform(df)
    df = map_ordinal(df)
    
    print("Dataframe encoded")
    return df

all_data = encode_categorical(all_data, nominal)

# drop two unwanted columns -->>> WHYYY
all_data.drop("LotAreaCut",axis=1,inplace=True)
all_data.drop(['SalePrice'],axis=1,inplace=True)



# =============================================================================
# #Exploratory analysis
# =============================================================================

#Get insights into the y variable
train.SalePrice.describe()   # Summary
train.SalePrice.skew()  #->> Prices are highly skewed right
plt.hist(train.SalePrice, color = "red", bins= 50)
plt.show()

#Using a logaritmic scale
np.log(train.SalePrice).skew()  #Skewness
plt.hist(np.log(train.SalePrice), color = 'red', bins = 50)
plt.show()    # Seems normally distribuited or symmetric using log

#Find correlation matrix for numeric features
