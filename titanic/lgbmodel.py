# -*- coding: utf-8 -*-
__author__ = 'lishanshan'
import re
import pandas as pd
from feature_model import family_fare_avg,mode_fill,age_fill,fare_fill
from xgbmodle import get_title
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from skopt import BayesSearchCV
from sklearn.metrics import  *
import featuretools as ft
from feature_analyze import getKsIv


params = {'boosting_type': 'gbdt',
          'max_depth': -1, #限制树的最大深度
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 64,  #一颗树上的叶子数
          'learning_rate': 0.05,  #梯度下降步长
          'max_bin': 512,   #最大直方图数目默认255
          'subsample_for_bin': 200, #构建直方图的数据的数量，越大效果越好，时间越长，如果数据非常稀疏，可以设置更大的值
          'subsample': 1,   #每棵树随机采样比例
          'subsample_freq': 1, #bagging的频率，0代表禁止bagging,k意味着每k次迭代执行bagging
          'colsample_bytree': 0.8, #每颗树训练前随机选择80%特征
          'reg_alpha': 5, #l1正则haunted
          'reg_lambda': 10,  #l2正则化
          'min_split_gain': 0.5,  #执行切分的最小增益
          'min_child_weight': 1, #子节点最小权重
          'min_child_samples': 5, #一个叶子上数据的最小数量
          'scale_pos_weight': 1, #正值的权重
          'num_class': 1,  #对于多分类才有这个参数吧?  分类个数
          'train_metric': True, #输出度量结果
          'metric': 'binary_error'}   #度量函数

def ADSplit(s):
    match = re.match(r"([a-z]+)([0-9]+)", s, re.I)

    try:
        letter = match.group(1)
    except:
        letter = ''

    try:
        number = match.group(2)
    except:
        number = 9999

    return letter, number

def DR(s):
    # Check contents
    if isinstance(s, (int, float)):
        # If field is empty, return nothing
        letter = ''
        number = ''
        nRooms = 9999
    else:
        # If field isn't empty, split sting on space. Some strings contain
        # multiple cabins.
        s = s.split(' ')
        # Count the cabins based on number of splits
        nRooms = len(s)
        # Just take first cabin for letter/number extraction
        s = s[0]

        letter, number = ADSplit(s)

    return [letter, number, nRooms]

class LgbmModel(object):
    def __init__(self):
        self.train_test = pd.read_csv("./data/train.csv")
        test = pd.read_csv("./data/test.csv")
        self.pid = test.pop("PassengerId")
        self.train_test = self.train_test.append(test)

        #处理数据+基本特征生成
        self.data_processing()

        #类别信息转为数值信息
        self.cate_col = ['CL','CN','Embarked','Sex','title','sexpclass']
        self.categorical_feature_process()

        #训练测试集 x y
        train_test = self.train_test[self.train_test['Survived'].notnull()]
        self.kaggle_test = self.train_test[self.train_test['Survived'].isnull()]
        self.train_test_y = train_test.pop("Survived")
        self.train_test = train_test
        self.kaggle_test.drop("Survived",axis=1,inplace=True)

        #特征衍生
        self.xgb_feature_more()
        print "***********共衍生特征数量*************"
        print len(self.train_test.columns)

        #得到特征列表：主要用于lgbm dataset的feature_name参数
        self.fea_col = list(self.train_test.columns)

        #特征选择只用train的做特征筛选，不动test （所以先取样本）
        self.sample_process()
        self.select_feature()
        self.fea_col = list(self.train_test.columns) #特征选择后还要更新下fea_col

        #将有label的 train_test  train_test_y 分成训练集 和 测试集
        self.sample_process()

        #grid bayes 均使用LGBMClassifier 所以提出来弄一份了
        self.lgbm_model()

    def lgbm_feature_select(self):
        self.lgbm_train()

        try:
            feature_list = self.feature_list
        except AttributeError as e:
            feature_list = list(self.train.columns)

        self.lgb_fe_importance = pd.DataFrame({
        'column': feature_list,
        'importance': self.gbm_model.feature_importance(importance_type = "gain"),
        }).sort_values(by='importance',ascending=False)

        #self.lgb_fe_importance.to_csv("./lgb_fe_importance.csv",index=None)

        # lgb.plot_importance(self.gbm_model)
        # plt.show()

    def sample_process(self):
        self.train_test[:] = self.train_test[:].astype(float)
        cnt = int(0.8*self.train_test.shape[0])
        self.train = self.train_test.iloc[:cnt,:]
        self.test = self.train_test.iloc[cnt:,:]

        self.train_y = self.train_test_y.iloc[:cnt]
        self.test_y = self.train_test_y.iloc[cnt:]

    def select_feature(self):
        df = pd.DataFrame()
        for col in self.train_test.columns:
            res = getKsIv(self.train_y,self.train[col])
            df = df.append(pd.DataFrame({"feature":col,"ks":res['ks'],'iv':res['iv']},index=[0]))

        iv_sort = df.sort_values(by='iv',ascending=False)
        #iv_sort.to_csv("./lgb_iv_sort.csv",index=None)
        self.feature_iv = iv_sort[iv_sort['iv']>0.02]

        ks_sort = df.sort_values(by='ks',ascending=False)
        #ks_sort.to_csv("./lgb_ks_sort.csv",index=None)
        self.feature_ks = ks_sort[ks_sort['ks']>0.02]

        self.lgbm_feature_select()
        lgb_fea_list = list(self.lgb_fe_importance[self.lgb_fe_importance['importance']>0]['column'].values)
        self.feature_list = list(set(list(self.feature_iv['feature']) + list(self.feature_ks['feature']) + lgb_fea_list))  #合并iv ks 训练模型的特征
        print "**************筛选后特征*************"
        print len(self.feature_list)
        self.train_test = self.train_test[self.feature_list]

    def lgbm_model(self):
        self.mdl = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective='binary',
                             nthread=5,
                             silent=True,
                             max_depth=params['max_depth'],
                             max_bin=params['max_bin'],
                             subsample_for_bin=params['subsample_for_bin'],
                             subsample=params['subsample'],
                             subsample_freq=params['subsample_freq'],
                             min_split_gain=params['min_split_gain'],
                             min_child_weight=params['min_child_weight'],
                             min_child_samples=params['min_child_samples'],
                             scale_pos_weight=params['scale_pos_weight'])


    def lgbm_grid_search(self):
        gridParams = {
        'learning_rate': [0.01],
        'n_estimators': [8, 24],
        'num_leaves': [6, 8, 12, 16],
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
        'seed': [500],
        'colsample_bytree': [0.65, 0.75, 0.8],
        'subsample': [0.7, 0.75],
        'reg_alpha': [1, 2, 6],
        'reg_lambda': [1, 2, 6],
        }
        gridParams = {}
        grid = GridSearchCV(self.mdl, gridParams, verbose=1, cv=5, n_jobs=-1,scoring='accuracy')
        grid.fit(self.train_test.values,self.train_test_y)
        print(grid.best_params_)
        print(grid.best_score_)

    def lgbm_train(self):
        self.train[:] = self.train[:].astype(float)
        train_lgb_dateset = self.data_set(self.train,self.train_y)
        #train_lgb_dateset.data[:] = train_lgb_dateset.data[:].apply(pd.to_numeric) #经过上面函数每列的类型都变成object了，所以把类型都改回float
        self.gbm_model = lgb.train(params,train_lgb_dateset)

    def lgbm_test(self):
        test = self.test[:].apply(pd.to_numeric)
        test_pred=self.gbm_model.predict(test)
        test_pred = [ 1 if i> 0.5 else 0 for i in test_pred]
        print accuracy_score(self.test_y,test_pred)

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

    def kaggle_result(self):
        self.kaggle_test[:] = self.kaggle_test[:].astype(float)
        kaggle_result = self.gbm_model.predict(self.kaggle_test)
        kagge_pred = [ 1 if i> 0.5 else 0 for i in kaggle_result]
        result = pd.DataFrame({'PassengerId':self.pid,'Survived':kagge_pred})
        final_result = result.sort_values(by='PassengerId')
        final_result.to_csv("./result.csv",index=None)

    def bayes_search(self):
        from sklearn.model_selection import StratifiedKFold
        self.bayes_cv_tuner = BayesSearchCV(
            estimator=self.mdl,
            search_spaces= {
                    'learning_rate': (0.01, 0.1, 'log-uniform'),
                    'n_estimators': (1, 10),
                    'num_leaves': (1, 10),
                    'colsample_bytree': (0.6, 0.8, 'log-uniform'),
                    'subsample': (0.6, 0.8, 'log-uniform'),
                    'reg_alpha': (1, 6),
                    'reg_lambda' : (1,6)
            },
            scoring='accuracy',
            cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42),
            n_iter = 10,
            verbose = 0,
            refit = True,
            random_state = 42
        )
        result = self.bayes_cv_tuner.fit(self.train,self.train_y)
        print result.best_score_,result.best_params_

    def categorical_feature_process(self):
        for c in self.cate_col:
            self.train_test[c] = pd.Categorical(self.train_test[c])
            self.train_test[c] = self.train_test[c].cat.codes
            self.train_test[c] = pd.Categorical(self.train_test[c])

    def data_processing(self):
        family_fare_avg(self.train_test) #团体票处理
        self.train_test['title'] = self.train_test['Name'].apply(lambda x:get_title(x))


        self.train_test['sexpclass']=self.train_test['Sex'].str.cat(self.train_test['Pclass'].astype(str))
        self.train_test['family'] = self.train_test['SibSp']+self.train_test['Parch']
        self.train_test['parch_sibsp_ratio'] = (self.train_test['Parch']+1) / (self.train_test['SibSp'])
        self.train_test['Adult'] = self.train_test['Age']>18

        cabin_process = self.train_test['Cabin'].apply(DR)
        cabin_process = cabin_process.apply(pd.Series)
        cabin_process.columns = ['CL', 'CN', 'nC']
        self.train_test = pd.concat([self.train_test,cabin_process],axis=1)
        self.train_test.drop(['Name','Ticket','PassengerId','Cabin'],axis=1,inplace=True)

    def data_set(self,x,y):
        return lgb.Dataset(x,label=y,free_raw_data=False,feature_name=self.fea_col,categorical_feature=self.cate_col)
        # return lgb.Dataset(x,label=y,free_raw_data=False,feature_name=self.fea_col,categorical_feature='auto')

if __name__ == '__main__':
    ty = 'kaggle'

    lgbm = LgbmModel()

    if ty == 'train':
        lgbm.lgbm_train()
        lgbm.lgbm_test()

    if ty == 'grid':
        lgbm.lgbm_grid_search()

    if ty == 'bayes':
        lgbm.bayes_search()

    if ty == 'kaggle':
        lgbm.lgbm_train()
        lgbm.trans_feature_more()
        lgbm.kaggle_result()