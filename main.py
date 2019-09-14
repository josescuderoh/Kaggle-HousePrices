# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:07:57 2019

@author: JoseEscudero
"""

# Import required packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

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
        self.X = X
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
                'BsmtFinType1', 'BsmtCond', 'BsmtQual']
        for col in cols:
            X.loc[:,col].fillna(value = 'None', inplace=True) 
            
        #Substitue others for 0 according to data_description
        cols = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','GarageCars',
                'GarageArea', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
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
        self.X = X
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
    def get_feature_names(self):
        return self.colnames
    def fit(self, X, y=None):
        self.colnames = X.columns.tolist()
        return self
    def transform(self, X, y=None):
        skewness = X[X.columns.drop(self.ordinal)].select_dtypes(exclude=['object']).apply(lambda x: skew(x))
        ## Take note that were we have selected integers as well
        skewed_idx = skewness[abs(skewness) > self.skew].index  ##Returns the name
        X[skewed_idx] = np.log1p(X[skewed_idx])
        return X

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

numerical_pipeline = Pipeline(steps=[('num_cleaner', NumericCleaner()),
                                        ('ord_transformer', OrdinalTransformer()),
                                        ('num_transformer', NumericalTransformer(skew = 1, ordinal= ordinal))])

nominal_pipeline = Pipeline(steps=[('cat_cleaner', CategoricalCleaner()),
                                   ('one_hot_encoder', OneHotEncoder(sparse=False))])

#Combine both pipelines for parallel processing using COlumnTransformer
preprocessor = ColumnTransformer(
    transformers=[('num', numerical_pipeline, numeric),
                  ('cat', nominal_pipeline, nominal)])
preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])


#Fit and transform the complete dataset
X = preprocessing_pipeline.fit_transform(pd.concat([train,test], axis = 0))

num_names = preprocessing_pipeline.named_steps['preprocessor'].transformers_[0][1].named_steps['num_transformer'].get_feature_names()
nom_names = list(preprocessing_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['one_hot_encoder'].get_feature_names())
num_names.extend(nom_names)

#Create dataset for visualization
X_df= pd.DataFrame(data = X, columns =  num_names)

# Break dataset
std_scaler = StandardScaler()

Xstrain = pd.DataFrame(std_scaler.fit_transform(X_df[:train.shape[0]]), columns = X_df.columns)
Xstest = pd.DataFrame(std_scaler.fit_transform(X_df[train.shape[0]:]), columns= X_df.columns )
y_log =  np.log(y)

Xstrain.shape, Xstest.shape

## Selecting important features using a LASSO model

lasso=Lasso(alpha=0.001)
lasso.fit(Xstrain, y_log)

#%matplotlib qt

# Get the significance coefficients

feat_coef = pd.DataFrame({'importance': lasso.coef_}, index=X_df.columns)

#Get relevant features

#Plot relevant features
n = feat_coef.shape[0]
fig, ax = plt.subplots(figsize=(5,n//5))

feat_coef.loc[feat_coef.importance != 0,:]\
             .sort_values(by='importance') \
             .plot(kind='barh', ax=ax)

Xftrain = Xstrain[feat_coef.loc[feat_coef.importance != 0,:].index.to_list()]
Xftrain = Xstrain
Xftest = Xstest[feat_coef.loc[feat_coef.importance != 0,:].index.to_list()]
Xftest = Xstest

#%matplotlib inline

#Apply PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=180)

X_train = pca.fit_transform(Xftrain)
X_test = pca.transform(Xftest)

X_train.shape, X_test.shape


##################################################################
######################## Fit the model ###########################
##################################################################

#Models
models = [Lasso(), RandomForestRegressor(), SVR()]
names = ['LASSO', 'RF', 'SVR']

# Define Metric Function
def rmse_cv(model, X, y):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))
    return rmse

# Evaluate all possible models
import warnings
warnings.filterwarnings("ignore")

for name, model in zip(names, models):
    score = rmse_cv(model, X_train, y_log)
    print(f'{name}: RMSE Mean: {score.mean()} RMSE STD: {score.std()}')

##################################################################
###################### Export the results ########################
##################################################################

rf_regressor = RandomForestRegressor()
rf_regressor.fit(Xftrain, y_log)

#Predic and transform
pred = np.exp(rf_regressor.predict(Xftest))

# Export the file
result=pd.DataFrame({'Id':pd.read_csv('test.csv').Id, 'SalePrice':pred})
result.to_csv("submission.csv",index=False)