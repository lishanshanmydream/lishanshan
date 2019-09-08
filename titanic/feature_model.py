# -*- coding: utf-8 -*-
__author__ = 'lishanshan'
import pandas as pd
from data_analyze import get_title
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import  train_test_split

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import  ensemble
from sklearn.linear_model import LinearRegression
from some_model import *

#弃用：Cabin 填充：Age 用对应title的age均值填充 Embarked众数填充  ok
#one hot:SEX Pclass  ； Embarked  ； title
#分级： SibSp(0  1-2  3-4  >4)	Parch (0  1-3  >3)   fare(<=10  <10-60  60-100  >=100) 先看下不分级，再分级看下效果不同处。
#Fare 只对团体票平均处理 ok


def mode_fill(data):
    if data.isnull().sum() !=0 :
        data.fillna(data.mode().iloc[0],inplace=True)

def family_fare_avg(data):
    group_tickets = data['Fare'].groupby(by=data['Ticket']).transform('count')
    data['Fare'] = data['Fare']/group_tickets

def age_fill(data): #对应index 赋值对应title的mean
    title_mean =  data.groupby('title')['Age'].mean() #求title均值
    age_na=data['Age'].isna()
    name=list(data.loc[age_na,'title']) #每个为空的位置的title->title对应的数值->title的index
    t=title_mean.loc[name]
    t.index = data.loc[age_na,'Age'].index
    data.loc[age_na,'Age']=t

def fare_fill(data):
    pclass_mean = data.groupby('Pclass')['Fare'].mean()
    fare_na=data['Fare'].isna()
    name = list(data.loc[fare_na,'Pclass'])
    t=pclass_mean.loc[name]
    t.index = data.loc[fare_na,'Fare'].index
    data.loc[fare_na,'Fare'] = t

def get_dummy(df):
    l_dummy = ['title']
    for col in df.columns:
        if len(set(df[col]))<=5:
            l_dummy.append(col)
    df = pd.get_dummies(df,prefix=l_dummy,columns=l_dummy)
    return df


def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_train.drop(['Survived'],axis=1,inplace=True)
    missing_age_test.drop(['Survived'],axis=1,inplace=True)

    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)
    #模型1
    gbm_reg = ensemble.GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [1000], 'max_depth': [3],'learning_rate': [0.01], 'max_features': [18]}
    gbm_reg_grid = GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1,  scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    #print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
    #print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
    #print('GB Train Error for "Age" Feature Regressor:'+ str(gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test['Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)

    #模型2
    lrf_reg = LinearRegression()
    lrf_reg_param_grid = {'fit_intercept': [True], 'normalize': [True]}
    lrf_reg_grid = GridSearchCV(lrf_reg, lrf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    lrf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    #print('Age feature Best LR Params:' + str(lrf_reg_grid.best_params_))
    #print('Age feature Best LR Score:' + str(lrf_reg_grid.best_score_))
    #print('LR Train Error for "Age" Feature Regressor' + str(lrf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test['Age_LRF'] = lrf_reg_grid.predict(missing_age_X_test)

    #将两个模型预测后的均值作为最终预测结果
    #print('shape1',missing_age_test['Age'].shape,missing_age_test[['Age_GB','Age_LRF']].mode(axis=1).shape)
    #missing_age_test['Age'] = missing_age_test[['Age_GB','Age_LRF']].mode(axis=1)
    missing_age_test['Age'] = np.mean([missing_age_test['Age_GB'],missing_age_test['Age_LRF']])
    return missing_age_test['Age']

def main():
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    train_test = train.append(test)

#训练集&测试集一起生成特征：
    mode_fill(train_test['Embarked'])
    train_test['title'] = train_test['Name'].apply(lambda x:get_title(x))
    family_fare_avg(train_test)
    fare_fill(train_test) #测试集里fare有缺失，用对应pclass 的fare均值填充

    train_test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
    age_fill(train_test)  #Age 用对应title的均值填充

    train_test = get_dummy(train_test)

    #通过预测得到的AGE 分数是：0.77511 先不用预测age了 就用title对应的均值吧。
    #missing_age_train=train_test[train_test['Age'].notnull()]
    #missing_age_test =train_test[train_test['Age'].isnull()]
    #train_test.loc[train_test['Age'].isnull(),'Age'] = fill_missing_age(missing_age_train,missing_age_test)

    train_model = train_test[train_test['Survived'].notnull()]
    train_model.drop(['PassengerId'],axis=1,inplace=True)
    #测试集
    x=train_test[train_test['Survived'].isnull()]
    x.pop('Survived')
    test_PassengerId = x.pop('PassengerId')
    train_model_Y = train_model.pop('Survived')

    #做一下特征选择： (这样做完效果变差了，一共也没多少特征，先不选了)
    #corrdf=train_model.corr()
    #feature_name=list(corrdf['Survived'].abs().sort_values(ascending=False)[:25].index)
    #train_model=train_model[feature_name]

    #TODO: 因为data_analyze看一些组合特征效果应该不错比如pclass sex 等，所以做下特征衍生试试

    #LR model
    #lr_model(train_model,train_model_Y,x,test_PassengerId)

    #xgb model
    run_xgb_model(train_model,train_model_Y,x,test_PassengerId)

if __name__ == '__main__':
    main()
