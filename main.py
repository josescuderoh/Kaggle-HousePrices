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

#Define options
plt.style.use('ggplot')
pd.options.mode.chained_assignment = None

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
        return self
    def transform(self, X, y=None):
        
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
        return self
    def transform(self,X, y=None):
        
        #Substitute LotFrontageValue for median of corresponding lot area quantile 
        X.loc[:,'LotAreaCut'] = pd.qcut(X.LotArea,10)
        X.loc[:,'LotFrontage']=X.groupby(['LotAreaCut'])['LotFrontage'].transform(
                lambda x: x.fillna(x.median()))
        
        #Drop new field
        X.drop('LotAreaCut', axis=1,inplace= True)
        
        #Substitue others for None according to data_description
        cols = ['PoolQC', 'Fence', 'FireplaceQu', 'GarageQual',
                'GarageFinish', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 
                'BsmtFinType1', 'BsmtCond', 'BsmtQual', 
                'BsmtFullBath', 'BsmtHalfBath']
        for col in cols:
            X.loc[:,col].fillna(value = 'None', inplace=True) 
            
        #Substitue others for 0 according to data_description
        cols = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','GarageCars',
                'GarageArea', 'TotalBsmtSF']
        for col in cols:
            X.loc[:,col].fillna(value = 0, inplace=True)
        
        #Substitue others for mode according to data_description
        cols = ['KitchenQual', 'GarageYrBlt']
        for col in cols:
            X.loc[:,col].fillna(value = X[col].mode()[0], inplace=True)
            
        return X

class OrdinalTransformer(BaseEstimator, TransformerMixin):
    """Custom class for mapping ordinal fields to numbers for further processing"""
    def __init__(self):
        pass
    def fit(self,X, y=None):
        return self
    def transform(self,X, y=None):
        #Map for numerical values
        X.loc[:,'LandSlope'] = X['LandSlope'].map({
                "Gtl": 3,
                "Mod": 2,
                "Sev": 1})
        X.loc[:,'LotShape'] = X['LotShape'].map({
                "Reg": 4,
                "IR1": 3,
                "IR2": 2,
                "IR3": 1})
        X.loc[:,'ExterCond'] = X['ExterCond'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1})
        X.loc[:,'ExterQual'] = X['ExterQual'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1})
        X.loc[:,'BsmtQual'] = X['BsmtQual'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1,
                "None": 0})
        X.loc[:,'BsmtCond'] = X['BsmtCond'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1,
                "None": 0})
        X.loc[:,'BsmtExposure'] = X['BsmtExposure'].map({
                "Gd": 4,
                "Av": 3,
                "Mn": 2,
                "No": 1,
                "None": 0})
        X.loc[:,'BsmtFinType1'] = X['BsmtFinType1'].map({
                "GLQ": 6,
                "ALQ": 5,
                "BLQ": 4,
                "Rec": 3,
                "LwQ": 2,
                "Unf": 1,
                "None": 0})
        X.loc[:,'BsmtFinType2'] = X['BsmtFinType2'].map({
                "GLQ": 6,
                "ALQ": 5,
                "BLQ": 4,
                "Rec": 3,
                "LwQ": 2,
                "Unf": 1,
                "None": 0})
        X.loc[:,'HeatingQC'] = X['HeatingQC'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1})
        X.loc[:,'KitchenQual'] = X['KitchenQual'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1})
        X.loc[:,'FireplaceQu'] = X['FireplaceQu'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1,
                "None": 0})
        X.loc[:,'GarageFinish'] = X['GarageFinish'].map({
                "Fin": 3,
                "RFn": 2,
                "Unf": 1,
                "None": 0})
        X.loc[:,'GarageQual'] = X['GarageQual'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1,
                "None": 0})
        X.loc[:,'GarageCond'] = X['GarageCond'].map({
                "Ex": 5,
                "Gd": 4,
                "TA": 3,
                "Fa": 2,
                "Po": 1,
                "None": 0})
        X.loc[:,'PoolQC'] = X['PoolQC'].map({
                "Ex": 4,
                "Gd": 3,
                "TA": 2,
                "Fa": 1,
                "None": 0})
        X.loc[:,'Fence'] = X['Fence'].map({
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

#Neighborhodd is needed
numeric = train.columns.drop(nominal)

## Build pipelines of transformers
numerical_pipeline = Pipeline(steps=[('num_selector', FeatureSelector(feature_names=numeric)),
                                        ('num_cleaner', NumericCleaner()),
                                        ('ord_transformer', OrdinalTransformer()),
                                        ('num_transformer', 
                                         NumericalTransformer(skew = 1, ordinal= ordinal)),
                                        ('std_scaler', StandardScaler())])

train2 = FeatureSelector(feature_names=numeric).fit_transform(train)
train2 = NumericCleaner().fit_transform(train2)
train2 = OrdinalTransformer().fit_transform(train2)
train2 = NumericalTransformer(skew = 1, ordinal= ordinal).fit_transform(train2)
train2 = StandardScaler().fit_transform(train2)   ## Removes some things

categorical_pipeline = Pipeline(steps=[('feature_selector', FeatureSelector(feature_names=nominal)),
                                       ('cat_cleaner', CategoricalCleaner()),
                                       ('one_hot_encoder', OneHotEncoder(sparse=False))])

train3 = FeatureSelector(feature_names=nominal).fit_transform(train)
train3 = CategoricalCleaner().fit_transform(train3)
train3 = OneHotEncoder(sparse=False).fit_transform(train3)

#Last steps are turning into floats, what can I do to keep the names??

#Combine both pipelines for parallel processing using feature union
preprocessing_pipeline = FeatureUnion(
        transformer_list=[('categorical_pipeline', categorical_pipeline),
                          ('numerical_pipeline', numerical_pipeline)])

#Preprocess and get the feature matrix
test_process = preprocessing_pipeline.fit_transform(train)

## Selecting important features using a LinearRegression model

