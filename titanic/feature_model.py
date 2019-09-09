# -*- coding: utf-8 -*-
__author__ = 'lishanshan'
import pandas as pd
from data_analyze import get_title
from sklearn.linear_model import LinearRegression
from some_model import *
from sklearn import model_selection
from sklearn.ensemble import  RandomForestClassifier

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
        if len(set(df[col]))<=8:
            l_dummy.append(col)
    df = pd.get_dummies(df,prefix=l_dummy,columns=l_dummy)
    return df

def SibSp_class(num):
    if num == 0:
        return 0
    elif num <3:
        return 1
    elif num < 5:
        return 2
    else:
        return 3

def Parch_class(num):
    if num == 0:
        return 0
    elif num <4:
        return 1
    else:
        return 2

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

def fea_weight(train_x,train_y):
    params={
        'booster':'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth':5,
        'subsample':0.8,
        'colsample_bytree':0.9,
        'eta': 0.15,
        'seed':2019,
        'nthread':32,
        'silent':1,
        }
    dtrain=xgb.DMatrix(train_x,label=train_y)
    bst=xgb.train(params,dtrain,num_boost_round=100,verbose_eval=False)
    fweight = bst.get_score(importance_type='weight')
    feature_weight = pd.DataFrame(fweight.items(),columns=['feature','weight']).sort_values(by='weight',ascending=False)
    feature_weight.to_csv("./feature_weight.csv",index=None)
    return feature_weight

def scaler(df_tr,df_te):
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(df_tr)
    tr = scaler.transform(df_tr)
    te = scaler.transform(df_te)
    return tr, te, scaler

def xgb_tr(train_x,train_y):
    params = {'objective':'binary:logistic','max_depth':3,'n_estimators':50,'learning_rate':0.1,'subsample':0.8,'colsample_bytree':0.8,'seed':2019,'nthread':32}
    bst = xgb.XGBClassifier(**params)
    bst.fit(np.array(train_x),train_y,eval_metric='auc')
    return bst

def feature_more(train_x,clf):
    from sklearn.preprocessing import OneHotEncoder
    train_clf = pd.DataFrame(clf.apply(train_x))
    xgb_enc=OneHotEncoder()
    new_data = xgb_enc.fit_transform(train_clf)
    new_data = new_data.toarray()
    n_col = new_data.shape[1]
    new_data = np.concatenate((train_x,new_data),axis=1)
    return new_data,xgb_enc,n_col

def test_feature_more_trans(df_test,clf,enc):
    test_clf=pd.DataFrame(clf.apply(df_test))
    new_test = enc.transform(test_clf)
    new_test = new_test.toarray()
    new_test = np.concatenate((df_test,new_test),axis=1)
    return new_test

def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
    # 随机森林
    rf_est = RandomForestClassifier(random_state=42)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X,titanic_train_data_Y)
    #将feature按Importance排序
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X), 'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']


    # AdaBoost
    ada_est = ensemble.AdaBoostClassifier(random_state=42)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.5, 0.6]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    #排序
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),'importance': ada_grid.best_estimator_.feature_importances_}).sort_values( 'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']

    # ExtraTree
    et_est = ensemble.ExtraTreesClassifier(random_state=42)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [15]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    #排序
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X), 'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 25 Features from ET Classifier:')
    print(str(features_top_n_et[:25]))

    # 将三个模型挑选出来的前features_top_n_et合并
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et], ignore_index=True).drop_duplicates()

    return features_top_n

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

    #TODO: 这里加个交叉特征 sex拼接pclass 然后再one-hot done
    train_test['sexpclass']=train_test['Sex'].str.cat(train_test['Pclass'].astype(str)) #加这维特征确实有提升。

    #合并parch and sibsp
    train_test['family'] = train_test['SibSp']+train_test['Parch']

    train_test['parchclass'] = train_test['Parch'].apply(lambda x:Parch_class(x)) #并不觉得效果会提升，本来parch一个就6种，不用分级应该也可以的。
    train_test['SibSp_class'] = train_test['SibSp'].apply(lambda x:SibSp_class(x))

    train_test = get_dummy(train_test)

    #通过预测得到的Age      分数是：0.77511 先不用预测age了 就用title对应的均值。
    # missing_age_train=train_test[train_test['Age'].notnull()]
    # missing_age_test =train_test[train_test['Age'].isnull()]
    # train_test.loc[train_test['Age'].isnull(),'Age'] = fill_missing_age(missing_age_train,missing_age_test)

    train_model = train_test[train_test['Survived'].notnull()]
    train_model.drop(['PassengerId'],axis=1,inplace=True)
    train_model_Y = train_model.pop('Survived')

    #测试集
    x=train_test[train_test['Survived'].isnull()]
    x.pop('Survived')
    test_PassengerId = x.pop('PassengerId')

    base_col = train_model.columns
    #train_np,test_np,scala=scaler(train_model,x)  #不加标准化 acc=0.89  加标准化 0.81 不加了
    bst = xgb_tr(train_model.values,train_model_Y)
    new_train,xgb_enc,n_col=feature_more(train_model.values,bst)
    test_x = test_feature_more_trans(x.values,bst,xgb_enc)

    col = [ 'f'+str(i) for i in range(n_col)]
    more_col = list(base_col)+col
    train_model = pd.DataFrame(new_train,columns=more_col)
    test_x = pd.DataFrame(test_x,columns=more_col)

    #corrdf=train_model.corr()  #暂时不用相关度看 特征了。
    #feature_name=list(corrdf['Survived'].abs().sort_values(ascending=False)[:25].index)
    #train_model=train_model[feature_name]


    #fea_weight(train_model,train_model_Y) #看下特征重要程度2 done

    features_top_n = get_top_n_features(train_model, train_model_Y, 220)
    train_model = train_model[features_top_n]
    test_x = test_x[features_top_n]

    #lr_model(train_model,train_model_Y,x,test_PassengerId)
    #run_xgb_model(train_model,train_model_Y,x,test_PassengerId)
    #run_decision_tree(train_model,train_model_Y)
    #融合模型
    voting_model_cv(train_model,train_model_Y,test_x,test_PassengerId)

if __name__ == '__main__':
    main()
