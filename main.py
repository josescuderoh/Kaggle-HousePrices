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
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline

#Define plotting style
plt.style.use('ggplot')

#Import dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Shape of dfs
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
all_data.drop(['Id'], axis=1, inplace=True)

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

all_data.drop(['SalePrice'], axis=1, inplace = True)  ## Not needed

# =============================================================================
# Feature engineering 
# =============================================================================

# Encoding categorical features
test_df = copy.copy(all_data)

class OrdinalTransformer

# First map ordinal columns
def map_ordinal(df):
    df['LandSlope'] = df['LandSlope'].map({
            "Gtl": 4,
            "Mod": 3,
            "Sev": 2})
    df['LotShape'] = df['LotShape'].map({
            "Reg": 4,
            "IR1": 3,
            "IR2": 2,
            "IR3": 1})
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
            "None": 0})
    df['BsmtCond'] = df['BsmtCond'].map({
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "None": 0})
    df['BsmtExposure'] = df['BsmtExposure'].map({
            "Gd": 4,
            "Av": 3,
            "Mn": 2,
            "No": 1,
            "None": 0})
    df['BsmtFinType1'] = df['BsmtFinType1'].map({
            "GLQ": 6,
            "ALQ": 5,
            "BLQ": 4,
            "Rec": 3,
            "LwQ": 2,
            "Unf": 1,
            "None": 0})
    df['BsmtFinType2'] = df['BsmtFinType2'].map({
            "GLQ": 6,
            "ALQ": 5,
            "BLQ": 4,
            "Rec": 3,
            "LwQ": 2,
            "Unf": 1,
            "None": 0})
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
            "None": 0})
    df['GarageFinish'] = df['GarageFinish'].map({
            "Fin": 3,
            "RFn": 2,
            "Unf": 1,
            "None": 0})
    df['GarageQual'] = df['GarageQual'].map({
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "None": 0})
    df['GarageCond'] = df['GarageCond'].map({
            "Ex": 5,
            "Gd": 4,
            "TA": 3,
            "Fa": 2,
            "Po": 1,
            "None": 0})
    df['PoolQC'] = df['PoolQC'].map({
            "Ex": 4,
            "Gd": 3,
            "TA": 2,
            "Fa": 1,
            "None": 0})
    df['Fence'] = df['Fence'].map({
            "GdPrv": 4,
            "MnPrv": 3,
            "GdWo": 2,
            "MnWw": 1,
            "None": 0})
    
    print("Mapping terminated")
    return df

##Remember to transform the month

test_df = map_ordinal(test_df)

#Use label encoding for the mapped and non mapped ordinal columns
    
#ordinal = ["LotShape", "LandSlope", "OverallQual", "OverallCond", 
#        "ExterCond", "ExterQual","BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
#        "BsmtFinType2", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageFinish",
#        "GarageQual", "GarageCond", "PoolQC", "Fence"]

#class MultiColumnLabelEncoder:
#    def __init__(self,columns = None):
#        self.columns = columns # array of column names to encode
#
#    def fit_transform(self,X):
#        '''
#        Transforms columns of X specified in self.columns using
#        LabelEncoder(). If no columns specified, transforms all
#        columns in X.
#        '''
#        output = X.copy()
#        if self.columns is not None:
#            for col in self.columns:
#                output[col] = LabelEncoder().fit_transform(output[col])
#        else:
#            for colname,col in output.iteritems():
#                output[colname] = LabelEncoder().fit_transform(col)
#        return output
#        print("Label encoding terminated")

#test_df = MultiColumnLabelEncoder(columns=ordinal).fit_transform(test_df)
    
## Encoding nominal values

## Create class for transforming skewed columns
class NumericalTransformer(BaseEstimator, TransformerMixin):
    ##These inherit everything I need to build the class, they have to always be used
    ##for transformers
    def __init__(self, skew=0.5):
        self.skew = skew
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        skewness = X.select_dtypes(exclude=['object']).apply(lambda x: skew(x))
        ## Take note that were we have selected integers as well
        skewed_idx = skewness[abs(skewness) > self.skew].index  ##Returns the name
        X[skewed_idx] = np.log1p(X[skewed_idx])
        return X

# Select nominal
nominal = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour",
                "Utilities", "LotConfig", "Neighborhood", "Condition1",
                "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
                "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation",
                "Heating", "CentralAir", "Electrical",
                "Functional", "GarageType","PavedDrive", 
                "MiscFeature", "SaleType", "SaleCondition"]



## Transform skewed numeric values
numerical_transformer = Pipeline(steps=[('transform_numeric', TransformSkewed(skew=1)),
                                      ('transform_nominal', OneHotEncoder())])

## Should I use get_dummies or OneHotEncoding here???
## Why not include all trnasformations within the pipline, including the map function
    

# drop two unwanted columns -->>> WHYYY
all_data.drop("LotAreaCut",axis=1,inplace=True)