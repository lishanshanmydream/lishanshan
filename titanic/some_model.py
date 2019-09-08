# -*- coding:utf-8 -*-
__author__ = 'lishanshan'
import pandas as pd
from sklearn.model_selection import  StratifiedKFold
from sklearn.linear_model import  LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.model_selection import train_test_split
from ks import ks
from sklearn.metrics import *

def add_kfold(train_model,train_model_Y,best_params):
    skf=StratifiedKFold(n_splits=5,shuffle=True)

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

def grid_search(X,y,hyperparameter_grid,clf,gridsearch_params = {'cv':3,'scoring':'roc_auc'}):

    gridparams = {'estimator' : clf,
                  'param_grid' : hyperparameter_grid,
                 }
    gridparams.update(gridsearch_params)
    grid_result = GridSearchCV(**gridparams).fit(X,np.squeeze(y))

    print 'Best: {}'.format(grid_result.best_params_)
    return grid_result

def do_lr_grid_search(train_model,train_model_Y):
    clf = LogisticRegression(penalty='l1', C=1, max_iter=100, solver='liblinear', n_jobs=32)
    c_list = np.arange(1,10,1)
    max_iter_list = np.arange(100,500,50)
    res = grid_search(train_model,train_model_Y,{'C':c_list,'max_iter':max_iter_list},clf)
    return res.best_params_

def do_xgb_grid_search(train_model,train_model_Y):
    params = {'objective':'binary:logistic', 'nthread':32, 'silent':1,'reg_alpha':1,'reg_lambda':1}
    clf = xgb.XGBClassifier(**params)
    learn_rate = np.arange(0.05,0.2,0.03)
    subsample = np.arange(0.8,1,0.1)
    colsample = np.arange(0.8,1,0.1)
    ntree = np.arange(10,1000,50)
    deep = np.arange(3,20,3)
    min_weight = np.arange(50,200,50)

    dic_grid = {'learning_rate':learn_rate,'subsample':subsample,'colsample_bytree':colsample,'n_estimators':ntree,'max_depth':deep,'min_child_weight':min_weight}
    res = grid_search(train_model,train_model_Y,dic_grid,clf)
    best_par = res.best_params_
    params.update(best_par)
    return params

def lr_model(train_model,train_model_Y,x,test_PassengerId):
    #网格找C max_iter  C值搜出来的是2 max_inter 来回变 不过总的来看100较好
    #best_params=do_grid_search(train_model,train_model_Y)
    best_params = {'C': 2, 'max_iter': 100}

    max_clf,max_score = add_kfold(train_model,train_model_Y,best_params)  #尝试交叉验证

    print max_score
    print "***************生成提交结果*****************"
    y_pred=max_clf.predict(x)
    y_pred= y_pred.astype(int)
    result=pd.DataFrame({'PassengerId':test_PassengerId,'Survived':y_pred})
    final_result = result.sort_values(by='PassengerId')
    final_result.to_csv("./result1.csv",index=None)

def get_score_threshold(dic_ks):
    l_ks = list(dic_ks['ks_list'])
    idx = l_ks.index(max(l_ks))
    span_score = dic_ks['span_list'][idx]
    res = span_score.split(',')[-1][:-1]
    return float(res)

def run_xgb_model(train_model,train_model_y,test,test_pid): #0.755分
    xgb_test = xgb.DMatrix(test)
    param=do_xgb_grid_search(train_model,train_model_y)
    plst = param.items()
    print "*********************params**********************"
    print plst
    skf=StratifiedKFold(n_splits=5,shuffle=True)
    max_ks = 0
    final_threshold = 0
    for train_index , valid_index in skf.split(train_model,train_model_y):
        x_train,y_train,x_valid,y_valid = train_model.iloc[train_index],train_model_y.iloc[train_index],train_model.iloc[valid_index],train_model_y.iloc[valid_index]
        xgb_train = xgb.DMatrix(x_train,label=y_train)
        xgb_valid = xgb.DMatrix(x_valid,label=y_valid)
        evallist = [(xgb_valid,'eval'), (xgb_train,'train')]
        clf=xgb.train(plst,xgb_train,100,evallist)
        valid_pred = clf.predict(xgb_valid)
        dic_ks = ks(y_valid,valid_pred)
        threshold = get_score_threshold(dic_ks)
        if dic_ks['ks'] > max_ks:
            max_ks = dic_ks['ks']
            final_threshold = threshold
            max_clf = clf
            valid_survived = [ 1 if i > final_threshold else 0 for i in valid_pred]
            print "lishanshan"
            print accuracy_score(y_valid,valid_survived)

    test_pred = max_clf.predict(xgb_test)
    test_survived = [ 1 if i > final_threshold else 0 for i in test_pred]
    result=pd.DataFrame({'PassengerId':test_pid,'Survived':test_survived})
    xgb_result=result.sort_values(by='PassengerId')
    xgb_result.to_csv("result_xgb.csv",index=None)