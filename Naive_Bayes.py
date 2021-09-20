import datetime
start = datetime.datetime.now()

import math
import pandas as pd 
import numpy as np 
import nltk

#前半部分为处理数据及保存，后半部分朴素贝叶斯实现

# 数据清洗 数据读取并处理,data为重复值、缺失值处理后的文件
# file=pd.read_csv('F:\\Reviews.csv')#读取数据
def clean(fileaddress):
    file=pd.read_csv(fileaddress)#读取数据
    data=file.drop_duplicates(subset=['ProductId','UserId','Text'],keep='first',inplace=False)#去除同一用户对同一商品的相同评价
    data = data.replace(' ', np.NaN)#转换空值形式
    Summary_null = data[pd.isnull(data['Summary'])]#Summary列为空值的数据
    data=data.dropna(axis=0,subset=['Summary'])#去除Summary列为空值的行
    return data
# data=clean('F:\\Reviews.csv')

# # import nltk
# # nltk.download()
# # nltk.download("punkt")
# # nltk.download('averaged_perceptron_tagger')
# # nltk.download('stopwords')

# 文本处理
def word_split(data):
    import re
    from nltk.corpus import stopwords#调停用词
    stoplist=set(stopwords.words('english'))#停用词列表
    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize
    ps = PorterStemmer()
    lines=[]
    labels=[]
    for i in range(len(data)):
        # row=data.iloc[i,9]#读取Text列每行数据
        row=data['Summary'].iloc[i]#读取summary列第i行数据
        row=row.lower()#转换为小写字母
        row = re.sub("[^a-zA-Z]", " ", row)#正则化去特殊字符
        row=' '.join(row.split())#去正则后多出的空格
        row=[i for i in row.split() if i not in stoplist]#删除停用词并分词；
        ro=[]
        for k in row:
            ro.append(ps.stem(k))#每行取词干
        lines.append(ro)
    for i in range(len(data)):
	    score=data.iloc[i,6]#读取Score列每行数据
	    labels.append(score)
    for i in range(len(labels)):#正面评价label为0，其余为1
	    if labels[i]>=4:labels[i]=0
	    else:labels[i]=1
    return [lines,labels]
# a=word_split(data)
# lines=a[0];labels=a[1]

# # 保存数据
# import pymysql
# from sqlalchemy import create_engine
# import openpyxl
# c={'lines':lines,
#    'labels':labels}
# data=pd.DataFrame(c)
# wwords=data.to_csv("F:\\words.csv")
words=pd.read_csv('F:\\words.csv')#读取出dataframe格式数据
allset = np.array(words[['lines','labels']])#转换成array格式

#随机生成测试集、训练集
def tt_set(allset):
    import random
    np.random.seed(1000)
    np.random.shuffle(allset)
    # 随机生成测试集、训练集
    # allset=allset.tolist()
    testset=allset[0:int(len(allset)*0.3)]#测试集
    trainset=allset[int(len(allset)*0.3):]#训练集
    return[trainset,testset]
# ttset=tt_set(allset)
# trainset=ttset[0];testset=ttset[1]

#保存训练集
# dataa=pd.DataFrame(trainset);dataa.columns=['lines_train','labels_train']
# trainsetcsv=dataa.to_csv("F:\\trainset.csv")
trainsetpd=pd.read_csv("F:\\trainset.csv",index_col=0)
#lines列表
lines_train_li=trainsetpd['lines_train'].tolist()#提取lines为列表，形式"['good', 'much']"
lines_train=[eval(i) for i in lines_train_li]#去引号
#labels列表
labels_train=trainsetpd['labels_train'].tolist()#提取labels为列表

#保存测试集
# datad=pd.DataFrame(testset);datad.columns=['lines_test','labels_test']
# testsetcsv=datad.to_csv("F:\\testset.csv")
testsetpd=pd.read_csv("F:\\testset.csv",index_col=0)#读取
#lines列表
lines_test_li=testsetpd['lines_test'].tolist()#提取lines为列表，形式"['good', 'much']"
lines_test=[eval(i) for i in lines_test_li]#去引号
#labels列表
labels_test=testsetpd['labels_test'].tolist()#提取labels为列表

# 计算每一个词出现频数，按字典格式输出，取频数前300个单词为特征集
def vocab_300(trainlines):
    allwords=[]#合并所有单词进一个列表
    vocablist=[]
    for i in range(len(trainlines)):
        allwords.extend(trainlines[i])
    from collections import Counter
    wordscount=Counter(allwords)#按频数生成字典
    vocabdict=wordscount.most_common(300)#以频数最高300个单词为特征集，形式[('good', 5), ('quality', 3), ('dog', 1)]
    for k in vocabdict:#提取出单词
        vocablist.append(k[0])
    return vocablist
vocablist=vocab_300(lines_train)
# vocablist = nltk.pos_tag(vocablist)#标注词性

# 将输入文本列表转换成向量形式
def sowvec(vocablist,lines):
    linesvec_list=[]
    for line in lines:
        linevec=[0]*len(vocablist)
        for k in line:
            if k in vocablist:
                linevec[vocablist.index(k)]=line.count(k)#单词表中单词在该句子中出现次数
        linesvec_list.append(linevec)
    return np.array(linesvec_list)#格式array
linesvec_train=pd.DataFrame(sowvec(vocablist,lines_train))#转df格式
linesvec_test=pd.DataFrame(sowvec(vocablist,lines_test))

# 训练
def trainNB(trainsetpd,linesvec_train):#训练集和特征向量 df格式
    # 计算p(x|yi)条件概率
    #df格式 添加特征向量
    trainsetpd_u=trainsetpd.join(linesvec_train,how='left')#训练集右侧添加特征向量，添加多列，每列为数值
    # 取正负集
    trainset0=trainsetpd_u.query('labels_train==0')#取标签为0的行 df格式
    trainset1=trainsetpd_u.query('labels_train==1')#取标签为1的行 df格式
    # 正面评价
    # 计算单词总数
    tlines0=trainset0['lines_train']#提取df的lines_train列
    tlines0np=np.array(tlines0)#转np格式
    tlines0li=[eval(i) for i in tlines0np]#转列表格式去引号
    tlines0pd=pd.DataFrame(tlines0li)#转pd格式
    mxn0=tlines0pd.shape
    num_voc_p0=mxn0[0]*mxn0[1]-tlines0pd.isnull().sum().sum()#计算单词数
    #向量和
    tlinesvec0=trainset0.iloc[:,2:]#提取向量列
    tlinesvec0_sum=tlinesvec0.apply(lambda x: x.sum(), axis=0)#纵向相加
    tlinesvec0_sum=np.array(tlinesvec0_sum)
    # 计算p(x|yi)条件概率
    theta0=(tlinesvec0_sum+1)/(num_voc_p0+len(vocablist))#每个单词表单词标记为正面评价的概率,已平滑
    logtheta0=np.log(theta0)#取对数 array格式

    # 负面评价 
    # 计算单词总数
    tlines1=trainset1['lines_train'];tlines1np=np.array(tlines1)#转np格式
    tlines1li=[eval(i) for i in tlines1np];tlines1pd=pd.DataFrame(tlines1li)#转pd格式
    mxn1=tlines1pd.shape;num_voc_p1=mxn1[0]*mxn1[1]-tlines1pd.isnull().sum().sum()#计算单词数
    #向量和
    tlinesvec1=trainset1.iloc[:,2:]#提取向量列
    tlinesvec1_sum=np.array(tlinesvec1.apply(lambda x: x.sum(), axis=0))#纵向相加取array格式
    # 计算p(x|yi)条件概率
    theta1=(tlinesvec1_sum+1)/(num_voc_p1+len(vocablist))#每个单词表单词标记为正面评价的概率,已平滑
    logtheta1=np.log(theta1)#取对数 array格式

 # 计算p(yi) 类先验概率
    py0=(trainset0.shape[0]+1)/(trainsetpd.shape[0]+2)#修正前py0=counter/len(labels)
    py1=1-py0
    return [py0,py1,logtheta0,logtheta1]

trainvalue=trainNB(trainsetpd,linesvec_train)
py0=trainvalue[0];py1=trainvalue[1]
logtheta0=trainvalue[2];logtheta1=trainvalue[3]
print('py0=',py0,'py1=',py1)

# 分类器
def classifyNB(testsetpd,linesvec_test):
    label_classify=[];counter=0
    # 计算p并比较
    for k in np.array(linesvec_test):
        p0=np.log(py0)+np.sum(logtheta0*k)
        p1=np.log(py1)+np.sum(logtheta1*k)
    # 比较
        if p0>=p1:
            label_classify.append(0)
        else:label_classify.append(1)
    # 正确率
    for i in range(len(label_classify)):
        if label_classify[i]==labels_test[i]:counter=counter+1
    truerate=counter/len(label_classify)
    return [truerate,label_classify]
result=classifyNB(testsetpd,linesvec_test)
print('正确率=',result[0])

end = datetime.datetime.now()
print('totally time is',end-start)


# import matplotlib.pyplot as plt
# x=[300,600,900,1500,3000]
# y1=[0.819413995,0.833297088,0.838651659,0.846968584,0.853051988]
# y2=[0.819078967,0.832497722,0.840315044,0.846986217,0.852440709]
# plot1 = plt.plot(x, y1, 's',label='seed1000')
# plot2 = plt.plot(x, y2, 'o',label='seed999')
# plt.xlabel('length of vocabulary')
# plt.ylabel('accuracy')
# plt.legend()
# plt.title('result')
# plt.savefig('result.png')
# plt.show()