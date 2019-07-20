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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion

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

#Drop id and y
y= train['SalePrice']

train.drop('Id', axis=1, inplace = True)
test.drop('Id', axis=1, inplace = True)
train.drop('SalePrice', axis=1, inplace = True)

#Check for NAs

# =============================================================================
# Feature engineering 
# =============================================================================

class CategoricalCleaner(BaseEstimator, TransformerMixin):
    """Custom class for cleaning categorical features with respect to data_description"""
    def __init__(self):
        pass
    def fit(self, X, y=None):
        pass
    def transform(X, y=None):
        
        #Substitue others for None according to data_description
        cols = ['MiscFeature', 'Alley', 'GarageType','MasVnrType']
        for col in cols:
            X[col].fillna(value = 'None', inplace=True) 
                    
        #Substitue others for mode according to data_description
        cols = ['Electrical', 'MSZoning', 'Utilities', 'Functional', 'Exterior2nd',
                'Exterior1st', 'SaleType']
        for col in cols:
            X[col].fillna(value = X[col].mode()[0], inplace=True)
        return X

class NumericCleaner(BaseEstimator, TransformerMixin):
    """Custom class for cleaning numeric and ordinal features with respect to data_description"""
    def __init__(self):
        pass
    def fit(self, X, y=None):
        pass
    def transform(X, y=None):
        
        #Substitute LotFrontageValue for median of neighboorhood
        X['LotFrontage'] = X.groupby(['Neighborhood'])[['LotFrontage']].transform(
                lambda x: x.fillna(x.median()))
        
        #Substitue others for None according to data_description
        cols = ['PoolQC', 'Fence', 'FireplaceQu', 'GarageQual',
                'GarageFinish', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 
                'GarageYrBlt', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 
                'BsmtFullBath', 'BsmtHalfBath']
        for col in cols:
            X[col].fillna(value = 'None', inplace=True) 
            
        #Substitue others for 0 according to data_description
        cols = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','GarageCars',
                'GarageArea', 'TotalBsmtSF']
        for col in cols:
            X[col].fillna(value = 0, inplace=True)
        
        #Substitue others for mode according to data_description
        cols = ['KitchenQual']
        for col in cols:
            X[col].fillna(value = X[col].mode()[0], inplace=True)
            
        return X

class OrdinalTransformer(BaseEstimator, TransformerMixin):
    """Custom class for mapping ordinal fields to numbers for further processing"""
    def __init__(self):
        pass
    def fit(self,X, y=None):
        pass
    def transform(self,X, y=None):
        #Map for numerical values
        X['LandSlope'] = X['LandSlope'].map({
                "Gtl": 4,
                "Mod": 3,
                "Sev": 2})
        X['LotShape'] = X['LotShape'].map({
                "Reg": 4,
                "IR1": 3,
                "IR2": 2,
                "IR3": 1})
        X['ExterCond'] = X['ExterCond'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1})
        X['ExterQual'] = X['ExterQual'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1})
        X['BsmtQual'] = X['BsmtQual'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1,
                "None": 0})
        X['BsmtCond'] = X['BsmtCond'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1,
                "None": 0})
        X['BsmtExposure'] = X['BsmtExposure'].map({
                "Gd": 4,
                "Av": 3,
                "Mn": 2,
                "No": 1,
                "None": 0})
        X['BsmtFinType1'] = X['BsmtFinType1'].map({
                "GLQ": 6,
                "ALQ": 5,
                "BLQ": 4,
                "Rec": 3,
                "LwQ": 2,
                "Unf": 1,
                "None": 0})
        X['BsmtFinType2'] = X['BsmtFinType2'].map({
                "GLQ": 6,
                "ALQ": 5,
                "BLQ": 4,
                "Rec": 3,
                "LwQ": 2,
                "Unf": 1,
                "None": 0})
        X['HeatingQC'] = X['HeatingQC'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1})
        X['KitchenQual'] = X['KitchenQual'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1})
        X['FireplaceQu'] = X['FireplaceQu'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1,
                "None": 0})
        X['GarageFinish'] = X['GarageFinish'].map({
                "Fin": 3,
                "RFn": 2,
                "Unf": 1,
                "None": 0})
        X['GarageQual'] = X['GarageQual'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1,
                "None": 0})
        X['GarageCond'] = X['GarageCond'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1,
                "None": 0})
        X['PoolQC'] = X['PoolQC'].map({
                "Ex": 4,
                "Gd": 3,
                "TA": 2,
                "Fa": 1,
                "None": 0})
        X['Fence'] = X['Fence'].map({
                "GdPrv": 4,
                "MnPrv": 3,
                "GdWo": 2,
                "MnWw": 1,
                "None": 0})
        
        ## Mapped dataframe
        return X

class NumericalTransformer(BaseEstimator, TransformerMixin):
    """Custom class for transforming numerical features using measure of skewness
    and log1p. These classes inherit everything I need to build the class, they 
    have to always be used for transformers"""
    def __init__(self, skew=0.5, ordinal=[]):
        self.skew = skew
        self.ordinal = ordinal
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        skewness = X[X.columns.drop(self.ordinal)].select_dtypes(exclude=['object']).apply(lambda x: skew(x))
        ## Take note that were we have selected integers as well
        skewed_idx = skewness[abs(skewness) > self.skew].index  ##Returns the name
        X[skewed_idx] = np.log1p(X[skewed_idx])
        return X

#Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self._feature_names ] 

# Select nominal fields
nominal = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour",
                "Utilities", "LotConfig", "Neighborhood", "Condition1",
                "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
                "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation",
                "Heating", "CentralAir", "Electrical",
                "Functional", "GarageType","PavedDrive", 
                "MiscFeature", "SaleType", "SaleCondition"]

# Select ordinal fields
ordinal = ["LotShape", "LandSlope", "OverallQual", "OverallCond", 
        "ExterCond", "ExterQual","BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
        "BsmtFinType2", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageFinish",
        "GarageQual", "GarageCond", "PoolQC", "Fence"]

numeric = train.columns.drop(nominal)

## Build pipelines of transformers
numerical_transformer = Pipeline(steps=[('num_cleaner', NumericCleaner()),
                                        ('ord_transformer', OrdinalTransformer()),
                                        ('num_transformer', 
                                         NumericalTransformer(skew = 1, ordinal= ordinal)),
                                        ('std_scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[(('cat_cleaner', CategoricalCleaner())),
                                          ('feature_selector', FeatureSelector(feature_names=nominal)),
                                          ('one_hot_encoder', OneHotEncoder())])




## Should I use get_dummies or OneHotEncoding here???
## Why not include all trnasformations within the pipline, including the map function
    

##Remember to transform the month properly

# drop two unwanted columns -->>> WHYYY
all_data.drop("LotAreaCut",axis=1,inplace=True)