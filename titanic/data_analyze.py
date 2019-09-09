# -*- coding: utf-8 -*-
__author__ = 'lishanshan'
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
fig = plt.figure("data anaylize")
fig_c=0
#数据类型， 分类：Pclass  Embarked  数值：Age SibSp  Parch  Fare 字符：Name  Sex  Ticket  Cabin

#样本分析：
def sample_anaylize():
    plt.figure("sample anaylize")
    train['Survived'].value_counts().plot.pie(autopct='%1.1f%%')

#各数据缺失值分析:缺失量，比例
def missing_analyze(data):
    miss_count = {}
    for col in data.columns:
        mis_num = data[col].isnull().sum()
        if mis_num:
            miss_count[col]=(float(mis_num)/data.shape[0])
    return miss_count

def col_label_rel(col):
    col_label_list = col+ ['Survived']
    return train[col_label_list].groupby(col).mean()

def col_survived_count(col):
    col_survived_list = col + ['Survived']
    return train.groupby(col_survived_list)['Survived'].count()

def bar_fig(data,name):
    ax=get_subplot()
    plt.title(name,fontsize=6)
    ax.bar(data['Survived'].index,data['Survived'].values,width=0.3,alpha=0.5)

    for x,y in zip(data['Survived'].index,data['Survived'].values):
        plt.text(x,y,'%.2f'%y,ha='center',va='bottom')

def miss_value_fig():
    train_missing_anaylze = missing_analyze(train)
    test_missing_anaylze =  missing_analyze(test)

    #Embarked 可以用众数填充；Cabin缺失值77%+ 无法填充 （可看下缺失与否与label关系）；Age 尝试按称呼填充
    ax=get_subplot()
    plt.title("Train missing value Ratio",fontsize=6)
    ax.bar(train_missing_anaylze.keys(),train_missing_anaylze.values(),width=0.3,alpha=0.5)

    for x,y in zip(train_missing_anaylze.keys(),train_missing_anaylze.values()):
        plt.text(x,y,'%.4f'%y,ha='center',va='bottom')

    ax = get_subplot()
    plt.title("Test missing value Ratio",fontsize=6)
    ax.bar(test_missing_anaylze.keys(),test_missing_anaylze.values(),width=0.3,alpha=0.5)
    for x,y in zip(test_missing_anaylze.keys(),test_missing_anaylze.values()):
        plt.text(x,y,'%.4f'%y,ha='center',va='bottom')

def get_subplot():
    global fig_c
    fig_c = fig_c+1
    return fig.add_subplot(4,2,fig_c)

def sex_pclass_label_fig(sex_pclass_label_rel):
    ax=get_subplot()
    plt.title("Sex Pclass Survived Ratio",fontsize=6)
    x_index = [];y_value=[]

    for k,v in sex_pclass_label_rel.index.values:
        x_index.append(str(k)+str(v))
    for v in sex_pclass_label_rel.values:
        y_value.append(v[0])

    ax.bar(np.array(x_index),np.array(y_value),width=0.3,alpha=0.5)
    for x,y in zip(np.array(x_index),np.array(y_value)):
        plt.text(x,y,'%.2f'%y,ha='center',va='bottom')

def age_survived():
    plt.figure("Age Survived Anaylze")
    ax=plt.subplot(121)
    sns.violinplot("Pclass","Age",hue="Survived",data=train,split=True,ax=ax)
    ax.set_title("Pclass and Age vs Survived")
    ax.set_yticks(range(0,110,10))

    axx=plt.subplot(122)
    sns.violinplot("Sex","Age",hue="Survived",data=train,split=True,ax=axx)
    axx.set_title("Sex and Age vs Survived")
    axx.set_yticks(range(0,110,10))

def pclass_fare_survived():
    plt.figure("pclass and fare analyze")
    ax=plt.subplot(111)
    sns.violinplot("Pclass","Fare",hue="Survived",data=train,split=True,ax=ax)
    ax.set_title("pclass and fare vs Survived")
    ax.set_xticks(range(0,3,1))

def get_title(name):
    import re
    title_search = re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)

def tickets_analyze():
    #print train['Ticket'].groupby(train['Ticket']).count().shape
    #训练集整体891，Ticket分组以后681 ?? 为什么是票信息有210个是重复的？ 那些重复的票信息有什么关系
    #分析重复ticket名字与fare关系
    trick_group = train['Ticket'].groupby(train['Ticket']).count()
    inde = list(trick_group[trick_group.values>1].index)
    print train[train['Ticket'].isin(inde)].loc[:,['Ticket','Fare']].sort_values(by="Ticket")

def main():
    miss_value_fig() #训练集、测试集 缺失值比例

    #下面是类别型数据与survived 关系
    sex_label_rel = col_label_rel(['Sex']) #性别与是否获救关系
    bar_fig(sex_label_rel,"Sex Survived Ratio")

    pclass_label_rel = col_label_rel(['Pclass']) #仓级与是否获救关系
    #print col_survived_count(['Pclass'])
    bar_fig(pclass_label_rel,"Pclass Survived Ratio")

    sex_pclass_label_rel = col_label_rel(['Sex','Pclass']) #性别 仓级 组合与是否获救关系
    sex_pclass_label_fig(sex_pclass_label_rel)

    embarked_survived_rel = col_label_rel(['Embarked']) #C =瑟堡，Q =皇后镇，S =南安普敦
    #print col_survived_count(['Embarked'])
    bar_fig(embarked_survived_rel,"Embarked and Survived Ratio")

    parch_survived_rel = col_label_rel(['Parch'])
    bar_fig(parch_survived_rel,"Parch and Survived Ratio")

    sibsp_survived_rel = col_label_rel(['SibSp'])
    bar_fig(sibsp_survived_rel,"sibsp and Survived Ratio")


    age_survived()  #数值型数据与survivied 关系分析   年龄、 亲友人数

    #分析fare前 先将团体票情况处理了  得到每个人的fare

    train['Fare'] = train['Fare']/(train['Fare'].groupby(by=train['Ticket']).transform('count'))
    pclass_fare_survived()

    #各类title数据量大小
    name_distict = train['Name'].apply(lambda  x: get_title(x))
    print name_distict.groupby(name_distict).count()


    tickets_analyze()  #ticket重复的是团体票
    plt.show()

'''
结论：
Parch ： 0  1-3  >3 分级
sibsp: 0  1-2  3-4  >4
做组合特征 female+Sex
由核密度图fare分级: <=10  <10-60  60-100 >=100
'''

if __name__ == '__main__':
    main()
