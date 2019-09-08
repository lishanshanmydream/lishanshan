# -*- coding: utf-8 -*-
__author__ = 'lishanshan'
import pandas as pd
from data_analyze import get_title
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import  train_test_split
import xgboost as xgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import  ensemble
from sklearn.linear_model import LinearRegression

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

def add_kfold(train_model,train_model_Y,best_params):
    skf=StratifiedKFold(n_splits=5,shuffle=True)
    print best_params['C']
    print best_params['max_iter']

    max_score = 0
    for train_index,vaild_index in skf.split(train_model,train_model_Y):
        clf = LogisticRegression(penalty='l1', C=best_params['C'], max_iter=best_params['max_iter'], solver='liblinear', n_jobs=32)
        x_train,y_train,x_valid,y_valid = train_model.iloc[train_index],train_model_Y.iloc[train_index],train_model.iloc[vaild_index],train_model_Y.iloc[vaild_index]
        #x_train,y_train,x_valid,y_valid = train_model[train_index],train_model_Y[train_index],train_model[vaild_index],train_model_Y[vaild_index]
        clf.fit(x_train,y_train)
        s = clf.score(x_valid,y_valid) #内部调的acc_score
        if s>max_score:
            max_score=s
            max_clf = clf
    return max_clf,max_score

def grid_search_lr(X,y,hyperparameter_grid,clf,gridsearch_params = {'cv':3,'scoring':'roc_auc'}):

    gridparams = {'estimator' : clf,
                  'param_grid' : hyperparameter_grid,
                 }
    gridparams.update(gridsearch_params)
    grid_result = GridSearchCV(**gridparams).fit(X,np.squeeze(y))

    print 'Best: {}'.format(grid_result.best_params_)
    return grid_result

def do_grid_search(train_model,train_model_Y):
    clf = LogisticRegression(penalty='l1', C=1, max_iter=100, solver='liblinear', n_jobs=32)
    c_list = np.arange(1,10,1)
    max_iter_list = np.arange(100,500,50)
    res = grid_search_lr(train_model,train_model_Y,{'C':c_list,'max_iter':max_iter_list},clf)
    return res.best_params_

def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_train.drop(['Survived'],axis=1,inplace=True)
    missing_age_test.drop(['Survived'],axis=1,inplace=True)

    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)
    #模型1
    gbm_reg = ensemble.GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [20], 'max_depth': [3],'learning_rate': [0.01], 'max_features': [18]}
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
    train_test = get_dummy(train_test)

    missing_age_train=train_test[train_test['Age'].notnull()]
    missing_age_test =train_test[train_test['Age'].isnull()]

    train_test.loc[train_test['Age'].isnull(),'Age'] = fill_missing_age(missing_age_train,missing_age_test)

    #age_fill(train_test)  #Age 用对应title的均值填充
    #TODO: 缺失Age通过预测填充

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


    #网格找C max_iter  C值搜出来的是2 max_inter 来回变 不过总的来看100较好
    #best_params=do_grid_search(train_model,train_model_Y)
    best_params = {'C': 2, 'max_iter': 100}
    #train_X,test_X,train_y,test_y = train_test_split(train_model,train_model_Y,train_size=0.8)
    max_clf,max_score = add_kfold(train_model,train_model_Y,best_params)  #尝试交叉验证



    print max_score
    print "***************生成提交结果*****************"
    y_pred=max_clf.predict(x)
    y_pred= y_pred.astype(int)
    result=pd.DataFrame({'PassengerId':test_PassengerId,'Survived':y_pred})
    final_result = result.sort_values(by='PassengerId')
    final_result.to_csv("./result1.csv",index=None)

if __name__ == '__main__':
    main()
