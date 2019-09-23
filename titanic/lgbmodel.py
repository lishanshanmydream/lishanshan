# -*- coding: utf-8 -*-
__author__ = 'lishanshan'
import re
import warnings
import bayes_opt as bo
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

from feature_model import family_fare_avg
from xgbmodle import get_title
import numpy as np

warnings.filterwarnings("ignore")
#from skopt import BayesSearchCV
from sklearn.metrics import  *
from feature_analyze import getKsIv

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
    if isinstance(s, (int, float)):
        letter = ''
        number = ''
        nRooms = 9999
    else:
        s = s.split(' ')
        nRooms = len(s)
        s = s[0]
        letter, number = ADSplit(s)
    return [letter, number, nRooms]

def lightgbm_bayesian_optimization(X, y, nfold=5, params_space=None,init_points=2, n_iter=10, objective='binary',metric='auc'):
    dtrain = lgb.Dataset(X, label=y)
    def lightgbm_cv_for_bo(learning_rate, max_depth, num_leaves, min_child_weight,
                           subsample, colsample_bytree,bagging_freq):
        paramt = {
                  'boosting_type': 'gbdt',
                  'objective': objective,
                  'nthread' : 4,
                  'metric': metric,
                  'learning_rate' : learning_rate,
                  'max_depth' : int(max_depth),
                  'num_leaves': min(2**int(max_depth), int(num_leaves)),
                  'min_child_weight' : min_child_weight,
                  'subsample' : max(min(subsample, 1), 0),
                  'colsample_bytree' : max(min(colsample_bytree, 1), 0),
                  'bagging_freq': int(bagging_freq)
                  }

        lgbc = lgb.cv(paramt,
                    dtrain,
                    num_boost_round = 100,
                    stratified = True,
                    nfold = nfold,
                    early_stopping_rounds = 30,
                    metrics = metric,
                    show_stdv = True,
                    verbose_eval=10,
                    seed=88)
        if metric == 'auc':
            return lgbc['%s-mean' % metric][-1]
        if metric == 'rmse':
            return -lgbc['%s-mean' % metric][-1]

    lgb_bo_obj = bo.BayesianOptimization(lightgbm_cv_for_bo, params_space, random_state=88)
    lgb_bo_obj.maximize(init_points=init_points, n_iter=n_iter)

    optimal_params = {
          'boosting_type': 'gbdt',
          'objective': objective,
          'nthread' : 4,
          'metric': metric,
          'seed' : 88
          }

    other_params = ['learning_rate', 'max_depth', 'num_leaves', 'min_child_weight',
                    'subsample', 'colsample_bytree', 'bagging_freq']
    for a_param in other_params:
        if a_param in lgb_bo_obj.max['params']:
            optimal_params[a_param] = lgb_bo_obj.max['params'][a_param]
            if a_param in ['max_depth', 'bagging_freq']:
                optimal_params[a_param] = int(optimal_params[a_param])
            if a_param == 'num_leaves':
                optimal_params[a_param] = min(2**int(optimal_params['max_depth']), int(optimal_params['num_leaves']))
            if a_param in ['subsample', 'colsample_bytree']:
                optimal_params[a_param] = max(min(optimal_params[a_param], 1), 0)

    return optimal_params

def lightgbm_best_tree(X, y, param=None, nfold=5, metric='auc'):
    dtrain = lgb.Dataset(X, label=y)
    cv_res = lgb.cv(param,
                    dtrain,
                    num_boost_round = 10000,
                    stratified = True,
                    nfold = nfold,
                    early_stopping_rounds = 30,
                    metrics = metric,
                    show_stdv = True,
                    verbose_eval=10,
                    seed=88)
    param['num_boost_round'] = len(cv_res['%s-mean' % metric])
    return param

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

    def lgbm_feature_select(self):
        tmp_param = {'boosting_type': 'gbdt',
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

        final_param = {
                'num_leaves': 3,
                'colsample_bytree': 1,
                'metric': 'auc',
                'min_child_weight': 2.0,
                'max_depth': 2,
                'subsample': 0.7,
                'seed': 88,
                'nthread' : 4,
                'objective': 'binary',
                'num_boost_round': 100,
                'learning_rate': 0.182191218, #差小数点后几位就差0.005个点
                'boosting_type': 'gbdt',
                'bagging_freq': 2
                }

        self.lgbm_train(final_param)

        split_importance = self.gbm_model.feature_importance(importance_type='split')
        gain_importance = self.gbm_model.feature_importance(importance_type='gain')
        feature_name = self.gbm_model.feature_name()

        imp_df = pd.DataFrame({'feature':feature_name,
                      'lightGBM_split':split_importance,
                      'lightGBM_gain': gain_importance})

        split_feature= list(imp_df[imp_df['lightGBM_split']>0]['feature'].values)
        gain_feature = list(imp_df[imp_df['lightGBM_gain']>0]['feature'].values)

        self.lgb_fe_importance = list(set(split_feature+gain_feature))

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

        self.lgbm_feature_select()  #特征选择的时候就训练了一次模型，但特征不是最终的，调参后训练的采用的最终的,但
        self.feature_list = list(set(list(self.feature_iv['feature']) + list(self.feature_ks['feature']) + self.lgb_fe_importance))  #合并iv ks 训练模型的特征
        print "**************筛选后特征*************"
        print len(self.feature_list)
        self.train_test = self.train_test[self.feature_list]

    def lgbm_grid_search(self):
        params = {
              'boosting_type': 'gbdt',
              'max_depth': 2,
              'objective': 'binary',
              'num_leaves': 3,
              'learning_rate': 0.182191218,
              'subsample': 1,
              'num_boost_round':100,
              'min_child_weight': 2.0,
              'colsample_bytree': 1,
              'seed':88,
              'bagging_freq' : 2,
              'metric': 'auc'}

        gridParams = {
        'max_depth': np.arange(8,14,2)
        }


        mdl = lgb.LGBMClassifier(boosting_type='gbdt',
                     objective='binary',
                     nthread=5,
                     silent=True,
                     num_leaves = params['num_leaves'],
                     learning_rate= params['learning_rate'],
                     colsample_bytree = params['colsample_bytree'],
                     seed = 88,
                     bagging_freq = 2,
                     metric = params['metric'],
                     max_depth=params['max_depth'],
                     subsample=params['subsample'],
                     min_child_weight=params['min_child_weight'])

        grid = GridSearchCV(mdl, gridParams, verbose=1, cv=5, n_jobs=-1,scoring='accuracy')
        grid.fit(self.train_test.values,self.train_test_y)
        print(grid.best_params_)
        print(grid.best_score_)

        params['max_depth'] = grid.best_params_['max_depth']
        lgbm.lgbm_train(params,params['num_boost_round'] )
        train_acc_gd = lgbm.lgbm_pred(lgbm.train,lgbm.train_y)
        test_acc_gd  = lgbm.lgbm_pred(lgbm.test,lgbm.test_y)

        print train_acc_gd
        print test_acc_gd

    def lgbm_train(self,params,num_boost_round=100):
        self.train[:] = self.train[:].astype(float)
        train_lgb_dateset = self.data_set(self.train,self.train_y)
        self.gbm_model = lgb.train(params,train_lgb_dateset,num_boost_round=num_boost_round)

    def lgbm_pred(self,data,data_ytrue):
        data = data[:].astype(float)
        data_pred=self.gbm_model.predict(data)
        data_pred = [ 1 if i> 0.62 else 0 for i in data_pred]
        return accuracy_score(data_ytrue,data_pred)

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

if __name__ == '__main__':
    ty = 'bayes&train&test'

    lgbm = LgbmModel()

    if ty == 'grid':
        lgbm.lgbm_grid_search()

    if ty == 'bayes&train&test':
        train_acc_list = []

        params_space = {'learning_rate': (0.18, 0.18),
                        'max_depth': (2, 10),
                        'num_leaves': (3,256), #上面3个提升准确率
                        'min_child_weight': (2, 2), #叶子上数据的最小数量
                        'subsample': (1.0, 1.0),  #调最上面3个先把这个设成1
                        'colsample_bytree' :(1.0, 1.0),
                        'bagging_freq' : (2.0,2.0)
                      }
        #基本确定：colsample 1 learning_rate :0.1832  min_child_weight:1 max_depth：2 subsample 0.8 (就想看看maxx_depth：3)
        for i in range(1):
            if i:
                left = params_space['min_child_weight'][1]
                right = params_space['min_child_weight'][1] + 5
                params_space['min_child_weight'] = (left,right)

            optimal_params = lightgbm_bayesian_optimization(lgbm.train,lgbm.train_y,params_space=params_space)
            optimal_params = lightgbm_best_tree(lgbm.train, lgbm.train_y, optimal_params)  

            params_result = pd.DataFrame(optimal_params,index=[0])
            lgbm.lgbm_train(optimal_params,optimal_params['num_boost_round']) #先用默认的,optimal_params['num_boost_round']
            params_result['train_acc'] = lgbm.lgbm_pred(lgbm.train,lgbm.train_y)
            params_result['test_acc']  = lgbm.lgbm_pred(lgbm.test,lgbm.test_y)
            if i :
                params_result.to_csv("./newfe_lgbm_params_result.csv",mode="ab+",index=None,header=None)
            else:
                params_result.to_csv("./newfe_lgbm_params_result.csv",mode="ab+",index=None)
            train_acc_list.append(params_result['train_acc'].values[0])
            if len(train_acc_list)>3 and len(set(train_acc_list[-3:])) == 1:
                break   #如果训练acc 连续3个循环都没有提升跳出循环

    if ty == 'kaggle': 
        params={
                'num_leaves': 3,
                'colsample_bytree': 1,
                'metric': 'auc',
                'min_child_weight': 2.0,
                'max_depth': 2,
                'subsample': 0.7,
                'seed': 88,
                'nthread' : 4,
                'objective': 'binary',
                'num_boost_round': 100,
                'learning_rate': 0.182191218, #差小数点后几位就差0.005个点
                'boosting_type': 'gbdt',
                'bagging_freq': 2
                }

        #pd.DataFrame(lgbm.train.columns.values).to_csv("./in_model_fea_new.csv",index=None)
        lgbm.lgbm_train(params,params['num_boost_round'])
        train_acc =lgbm.lgbm_pred(lgbm.train,lgbm.train_y)
        test_acc = lgbm.lgbm_pred(lgbm.test,lgbm.test_y)
        print {u"train-acc:" : train_acc,u"test-acc:":test_acc}

        params_result = pd.DataFrame(params,index=[0])
        params_result['train_acc'] = train_acc
        params_result['test_acc']  = test_acc
        params_result.to_csv("./train_params.csv",mode="ab+",index=None)

        lgbm.trans_feature_more()
        lgbm.kaggle_result()
