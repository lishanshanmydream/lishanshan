# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import collections
import numpy as np
import datetime
import math
import gc
PART = 10

def _get_cut_pos(cut_num, vec, head_pos, tail_pos):
    mid_pos = (head_pos + tail_pos) / 2
    if vec[mid_pos] == cut_num and (mid_pos == tail_pos or vec[mid_pos + 1] > cut_num):
        return mid_pos
    elif vec[mid_pos] <= cut_num:
        return _get_cut_pos(cut_num, vec, mid_pos+1, tail_pos)
    else:
        return _get_cut_pos(cut_num, vec, head_pos, mid_pos-1)

def psi(score_1, score_2, part_num=PART):
    null_1 = pd.isnull(score_1)
    score_1 = score_1[null_1 == False]
    score_1 = np.array(score_1.T)
    score_1.sort()
    null_2 = pd.isnull(score_2)
    score_2 = score_2[null_2 == False]
    score_2 = np.array(score_2.T)

    # score_1
    length = len(score_1)
    cut_list = [min(score_1)]
    order_num = []
    cut_pos_last = -1
    for i in np.arange(part_num):
        if i == part_num-1 or score_1[length*(i+1)/part_num-1] != score_1[length*(i+2)/part_num-1]:
            cut_list.append(score_1[length*(i+1)/part_num-1])
            if i != part_num-1:
                cut_pos = _get_cut_pos(score_1[length*(i+1)/part_num-1], score_1, length*(i+1)/part_num-1, length*(i+2)/part_num-2)    # find the position of the rightest cut
            else:
                cut_pos = length-1
            order_num.append(cut_pos - cut_pos_last)
            cut_pos_last = cut_pos
    order_num = np.array(order_num)
    order_ratio_1 = order_num / float(length)
    # print 'cut_list', cut_list
    # print 'order_ratio_1', order_ratio_1, sum(order_ratio_1)

    # score_2
    length = len(score_2)
    order_num = []
    for i in range(len(cut_list)):
        if i == 0:
            continue
        elif i == 1:
            order_num.append(len(score_2[(score_2 <= cut_list[i])]))
        elif i == len(cut_list)-1:
            order_num.append(len(score_2[(score_2 > cut_list[i-1])]))
        else:
            order_num.append(len(score_2[(score_2 > cut_list[i-1]) & (score_2 <= cut_list[i])]))
    order_num = np.array(order_num)
    order_ratio_2 = order_num / float(length)
    # print 'order_ratio_2', order_ratio_2, sum(order_ratio_2)

    # psi
    psi = sum([(order_ratio_1[i] - order_ratio_2[i]) * math.log((order_ratio_1[i] / order_ratio_2[i]), math.e) for i in range(len(order_ratio_1))])

    return psi

def nvl(grouped,key):
    if key in grouped.keys():
        return float(grouped[key])
    return float(0)

def caculateIv(gi,bi,g,b):
    gi, bi, g, b = float(gi),float(bi),float(g),float(b)
    if gi==0 or bi==0:
        return 0
    return (gi/g-bi/b) * math.log((gi/g)/(bi/b))

def getKsIv(y_true,y_pred,Kpart=10,dataType='continues',pos_label=1):
    #print 'lsllslslsls',len(y_pred[y_pred.isnull()])
    y_true, y_pred = np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1)

    y_true =[1 if i==pos_label else 0 for i in y_true]

    df_original = pd.DataFrame({
            'y_pred':y_pred,
            'y_true':y_true,
    })
    df_original['y_pred'].fillna(-999,inplace =True)
    df = df_original[df_original['y_pred'] != -999].copy()
    # print 'df shape:',df.shape
    df_nan = df_original[df_original['y_pred'] == -999].copy()
    # print 'df_nan shape',df_nan.shape
    df = df.sort_values(by='y_pred')
    df=df.reset_index(drop=True)
    dataunique =  df.y_pred.unique()
    kpart = min(len(dataunique),Kpart)
    section,positive,negative=[],[],[]  # 区间，该区间正例数量，负例数量
    if kpart<Kpart or dataType !='continues': # 如果离散性变量或连续性变量少于设定的数量（可以按照离散性变量处理）则直接按照（预测值，真实值)分组统计出各个属性值对应的正负样本数
        grouped = df.groupby(by=['y_pred','y_true'])['y_true'].count()
        for i in dataunique:
            section.append(i)
            positive.append(nvl(grouped,(i,1)))
            negative.append(nvl(grouped,(i,0)))
    else:
        section_length = math.ceil(df.shape[0]*1.0/kpart) # 计算出区间长度

        section_start_pre,section_end_pre  = -9999,-9999
        for i in range(kpart):
            if i*section_length > df.shape[0]-1:
                break
            section_start,section_end = df.y_pred[i*section_length],df.y_pred[min(df.shape[0]-1,(i+1)*section_length)]

           # print section_start_pre, section_start, section_end_pre, section_end
            if i!=0 and abs(section_start_pre - section_start)<1e-5 and abs(section_end_pre - section_end) < 1e-5 or abs(section_start - section_end) < 1e-5 and abs(section_start - section_end_pre) < 1e-5:  # 该区间已添加过，或者区间长度为0,则跳过
                continue
            section_start_pre,section_end_pre  = section_start,section_end
            # 计算出区间起始值
            if i==0:  # 如果是第一个区间，则是左右闭合区间，取出该区间的数据，统计出该区间的正负样本数
                 grouped = df[(df.y_pred>=section_start) & (df.y_pred<=section_end)].groupby(by='y_true')['y_true'].count()
                 section.append('[%.2f,%.2f]' %(section_start,section_end))
            else: # 其他情况，则是左开右闭区间，取出该区间的数据，统计出该区间的正负样本数
                grouped = df[(df.y_pred>section_start) & (df.y_pred<=section_end)].groupby(by='y_true')['y_true'].count()
                section.append('(%.2f,%.2f]' %(section_start,section_end))
            positive.append(nvl(grouped,1))
            negative.append(nvl(grouped,0))
    positive_sum =[sum(positive[:i]) for i in range(1,len(positive)+1)]  # 累计正样本数
    negative_sum =[sum(negative[:i]) for i in range(1,len(negative)+1)]  # 累计负样本数
    positive_nan = float( sum(df_nan['y_true']))
    positive_nan_1 = float(sum(df_nan['y_true']==1))
    negative_nan = float(df_nan.shape[0]-positive_nan)
    num_nan = float(df_nan.shape[0])
    #print positive_nan,positive_nan_1,negative_nan,num_nan
    positive_negative_list = list(np.array(positive)+np.array(negative))
    positive_ratio_list = list(np.array(positive)/(np.array(positive)+np.array(negative)))
    fpr_list = list(np.array(positive_sum)/sum(positive))
    tpr_list = list(np.array(negative_sum)/sum(negative))

    if num_nan != 0:
        section.append('nan section')
        positive.append(positive_nan)
        negative.append(negative_nan)
        positive_sum.append(positive_nan)
        negative_sum.append(negative_nan)
        positive_negative_list.append(num_nan)
        positive_ratio_list.append(positive_nan*1.0/num_nan)
        fpr_list.append(1)
        tpr_list.append(1)

    else:
        section.append('nan section')
        positive.append(positive_nan)
        negative.append(negative_nan)
        positive_sum.append(positive_nan)
        negative_sum.append(negative_nan)
        positive_negative_list.append(num_nan)
        positive_ratio_list.append(0)
        fpr_list.append(0)
        tpr_list.append(0)

    result = pd.DataFrame(
            {
                "section":section,  # 区间
                "positive_negative":positive_negative_list, # 该区间样本数量
                "positive":positive, # 该区间正样本数量
                "negative":negative, # 该区间负样本数量
                "positive_ratio":positive_ratio_list, # 该区间负样本比率(逾期率)
                "positive_sum":positive_sum,  # 累计正样本数量
                "negative_sum":negative_sum,  # 累计负样本数量
                "p_pall_ratio":fpr_list, # 累计正样本数量占所有正样本的比率
                "n_nall_ratio":tpr_list, # 累计负样本数量占所有负样本的比率
            })


    #print sum(result["positive_negative"])

    result['positive_negative_ratio'] = [i*1.0 /df_original.shape[0] for i in  result['positive_negative'] ]
    result['ks'] = [abs(result['p_pall_ratio'][i]-result['n_nall_ratio'][i]) for i in range(len(result['p_pall_ratio']))]
    g,b = sum(negative),sum(positive)
    result['iv'] = [caculateIv(result['negative'][i],result['positive'][i],g,b) for i in range(len(result['positive']))]
    result = result[['section','positive_negative','positive','negative','positive_negative_ratio','positive_ratio','p_pall_ratio','n_nall_ratio','ks','iv']]
    # pd.DataFrame(columns=[name],index=None).to_csv("./feature_detail_analyze.csv",mode="a",index=None)
    # result.to_csv("./feature_detail_analyze.csv",mode="a",index=None)

    dic = {
            'ks':result['ks'].max(),
            'iv':result['iv'].sum(),
            'detail':result
            }

    #return dic
    return dic