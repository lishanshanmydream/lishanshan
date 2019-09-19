# -*- coding: utf-8 -*-
__author__ = 'lishanshan'
import pandas as pd
from feature_model import family_fare_avg,mode_fill,age_fill,fare_fill
from feature_model import get_dummy
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import *
import numpy as np
from skopt import BayesSearchCV
from feature_analyze import getKsIv
import warnings
warnings.filterwarnings("ignore")

def get_title(name):
    import re
    title_search = re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)

class Xgbmodel(object):
    def __init__(self):
        self.train_test = pd.read_csv("./data/train.csv")

        test = pd.read_csv("./data/test.csv")
        self.pid = test.pop("PassengerId")
        self.train_test = self.train_test.append(test)

        #一起做数据处理，一起做dummy
        self.data_processing()
        self.cross_feature()
        self.train_test = get_dummy(self.train_test)

        train_test = self.train_test[self.train_test['Survived'].notnull()]
        self.kaggle_test = self.train_test[self.train_test['Survived'].isnull()]

        self.train_test_y = train_test.pop("Survived")
        self.train_test = train_test
        self.kaggle_test.drop("Survived",axis=1,inplace=True)

        #衍生label必须有值，所以用train
        self.xgb_feature_more()

        self.sample_process() #特征选择只用train的做特征筛选，不动test

        self.xgb_select_feture()
        self.select_feature() #用train选完特征，再用特征列表过滤下train_test特征

        self.sample_process() #重新得到过滤特征的train 和 test

    def data_processing(self):
        family_fare_avg(self.train_test) #团体票处理
        self.train_test['title'] = self.train_test['Name'].apply(lambda x:get_title(x))
        #fare_fill(self.train_test)
        #mode_fill(self.train_test['Embarked'])  #填充embarked和age 效果都有所下降不填充了。
        #age_fill(self.train_test)
        self.train_test.drop(['Name','Ticket','Cabin','PassengerId'],axis=1,inplace=True)

    def cross_feature(self):
        self.train_test['sexpclass']=self.train_test['Sex'].str.cat(self.train_test['Pclass'].astype(str))
        self.train_test['family'] = self.train_test['SibSp']+self.train_test['Parch']

    def select_feature(self):
        df = pd.DataFrame()
        for col in self.train_test.columns:
            res = getKsIv(self.train_y,self.train[col])
            df = df.append(pd.DataFrame({"feature":col,"ks":res['ks'],'iv':res['iv']},index=[0]))

        iv_sort = df.sort_values(by='iv',ascending=False)
        self.feature_iv = iv_sort[iv_sort['iv']>0.02]
        ks_sort = df.sort_values(by='ks',ascending=False)
        self.feature_ks = ks_sort[ks_sort['ks']>0.02]

        self.feature_list = list(set(list(self.feature_iv['feature']) + list(self.feature_ks['feature']) + self.xgb_feature)) #合并iv ks 训练模型的特征
        print len(self.feature_list)
        self.train_test = self.train_test[self.feature_list]



    def trans_dummy(self):
        self.test = get_dummy(self.test)
        for train_fea in self.train_fea:
            if train_fea not in self.test.columns:
                self.test[train_fea] = np.nan
        self.test = self.test[self.train_fea]

    def sample_process(self):
        cnt = int(0.8*self.train_test.shape[0])
        self.train = self.train_test.iloc[:cnt,:]
        self.test = self.train_test.iloc[cnt:,:]

        self.train_y = self.train_test_y.iloc[:cnt]
        self.test_y = self.train_test_y.iloc[cnt:]


    def select_params_grid(self):
        model = xgb.XGBClassifier(
            learning_rate=0.0869700535740794,
            max_depth=2,
            n_estimators=81,
            min_child_weight=1,
            gamma=0.00123933465812134,
            objective='binary:logistic',
            subsample = 0.8,
            colsample_bytree = 0.8
        )
        # learning_rate = np.arange(0.08,0.12,0.01)
        # max_depth = np.arange(5,8,1)
        # #min_child_weight = np.arange(3,10,1)
        # n_estimators = np.arange(20,60,5)
        #param_grid = dict(learning_rate=learning_rate,max_depth=max_depth,n_estimators=n_estimators)
        param_grid = dict()
        grid_search = GridSearchCV(model,param_grid,n_jobs=-1,cv=10,scoring='accuracy')
        grid_result=grid_search.fit(self.train,self.train_y)
        print grid_result.best_params_,grid_result.best_score_
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    def random_params_search(self):
        model = xgb.XGBClassifier()
        params = {'n_estimators':np.arange(5,60,5),'max_depth':np.arange(3,11,1),'learning_rate':np.arange(0.01,0.2,0.01),'min_child_weight':np.arange(1,10,1),'gamma':np.arange(0.02,0.1,0.02)}
        clf=RandomizedSearchCV(model,params,cv=10)
        random_result=clf.fit(self.train,self.train_y)
        print random_result.best_params_,random_result.best_score_
        means = random_result.cv_results_['mean_test_score']
        stds = random_result.cv_results_['std_test_score']
        params = random_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        #结论： max_depth : 5左右  min_child_weight:4  树：15~20，然后网格 learning_rate：0.12-0.16

    def bayes_search(self):
        from sklearn.model_selection import StratifiedKFold
        self.bayes_cv_tuner = BayesSearchCV(
            estimator=xgb.XGBClassifier(objective='binary:logistic',n_jobs=-1,silent=1),
            search_spaces= {
                    'learning_rate': (0.01, 0.1, 'log-uniform'),
                    'min_child_weight': (1, 10),
                    'max_depth': (1, 10),
                    'gamma': (1e-9, 0.2, 'log-uniform'),
                    'min_child_weight': (0, 5),
                    'n_estimators': (2, 100)
            },
            scoring='accuracy',
            cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=42),
            n_iter = 10,
            verbose = 0,
            refit = True,
            random_state = 42
        )
        result = self.bayes_cv_tuner.fit(self.train,self.train_y)
        print result.best_score_,result.best_params_



    def train_model(self,kaggle_test=None):
        clf = xgb.XGBClassifier(
            learning_rate=  0.09,
            max_depth=  5,
            n_estimators=40,
            min_child_weight=1,
            gamma= 0.01,
            objective='binary:logistic',
            subsample = 0.8,
            colsample_bytree = 0.8
        )

        #在上传kaggel结果的时候用所有的train训练，在看效果用0.8train训练
        if kaggle_test:
            self.clf = clf.fit(self.train_test,self.train_test_y)
        else:
            self.clf = clf.fit(self.train,self.train_y)

    def xgb_select_feture(self):
        self.train_model()
        clf = self.clf.get_booster()
        weight=clf.get_score(importance_type='weight')
        gain = clf.get_score(importance_type='gain')
        self.xgb_feature = list(set(weight.keys() + gain.keys()))

    def test_pred(self):
        self.test_pred_y = self.clf.predict(self.test)

    def kaggle_pred(self):
        pred = self.clf.predict(self.kaggle_test).astype(int)
        result = pd.DataFrame({'PassengerId':self.pid,'Survived':pred})
        final_result = result.sort_values(by='PassengerId')
        final_result.to_csv("./result.csv",index=None)

    def test_acc(self):
        print accuracy_score(self.test_y,self.test_pred_y)

    def xgb_feature_more(self):
        from feature_model import xgb_tr,feature_more
        base_col = self.train_test.columns
        self.bst = xgb_tr(self.train_test.values,self.train_test_y)
        new_train,self.xgb_enc,n_col=feature_more(self.train_test.values,self.bst)
        col = [ 'f'+str(i) for i in range(n_col)]
        self.more_col = list(base_col)+col
        self.train_test = pd.DataFrame(new_train,columns=self.more_col)

    def trans_feature_more(self):
        from feature_model import test_feature_more_trans
        kaggle_test = test_feature_more_trans(self.kaggle_test.values,self.bst,self.xgb_enc)
        self.kaggle_test = pd.DataFrame(kaggle_test,columns=self.more_col)
        self.kaggle_test = self.kaggle_test[self.feature_list]

if __name__ == '__main__':
    ty = 'grid'  #1.调参 grid/random/bayes  2：测试   3：用于提交kaggle:commit_kaggle
    titanic = Xgbmodel()

    if ty=='random':
        titanic.random_params_search()

    if ty=='grid':
        titanic.select_params_grid()

    if ty=='bayes':
        titanic.bayes_search()

    if ty=='test':
        titanic.train_model()
        titanic.test_pred()
        titanic.test_acc()

    if ty=='commit_kaggle': #有比较大的提升，但依然比最好成绩差0.00478
        titanic.train_model(kaggle_test=True)
        titanic.trans_feature_more()
        titanic.kaggle_pred()






