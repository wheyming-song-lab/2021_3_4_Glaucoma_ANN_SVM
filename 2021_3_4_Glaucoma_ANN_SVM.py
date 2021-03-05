# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:00:18 2021

@author: what1
"""

#要統一資料，同一份資料(視網膜以及身經厚度)要來自同一個病人，才能比較，所以統一使用data_both

import pandas as pd
from sklearn.neural_network import MLPClassifier
#import math
from sklearn.svm import SVC
import random
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#from tensorflow.keras import optimizers

dataG = pd.read_excel(open('numbername.xlsx', 'rb'),
              sheet_name='G')  
dataH = pd.read_excel(open('numbername.xlsx', 'rb'),
              sheet_name='H-1')

#dataG_index = dataG.axes
#dataH_index = dataH.axes

dataG_Retina = dataG.copy()
dataH_Retina = dataH.copy()

dataG_Retina = dataG_Retina.drop(['代號', '姓名' , 'T', 'TS', 'NS', 'N', 'NI', 'TI', 'G', 'TYPE.1'], axis=1)
dataH_Retina = dataH_Retina.drop(['代號', '姓名' , 'T', 'TS', 'NS', 'N', 'NI', 'TI', 'G', 'Unnamed: 15'], axis=1)

dataG_RNFL = dataG.copy()
dataH_RNFL = dataH.copy()

dataG_RNFL = dataG_RNFL.drop(['代號', '姓名' , 'UO', 'UI', 'LI', 'LO', 'TYPE'], axis=1)
dataH_RNFL = dataH_RNFL.drop(['代號', '姓名' , 'UO', 'UI', 'LI', 'LO', 'Unnamed: 15'], axis=1)
#修改列名
dataG_RNFL.rename(columns={'TYPE.1':'TYPE'}, inplace = True)

# dataG_Retina_index = dataG_Retina.axes
# dataG_RNFL_index = dataG_RNFL.axes

#dataH_both = dataH.drop(['代號', '姓名','Unnamed: 15'], axis=1)
#dataG_both = dataG.drop(['代號', '姓名','TYPE.1'], axis=1)
dataH_both = dataH.drop(['代號', 'Unnamed: 15'], axis=1)
dataG_both = dataG.drop(['代號', 'TYPE.1'], axis=1)

dataG_both = dataG_both.dropna()
dataH_both = dataH_both.dropna()

data_both = pd.concat([dataG_both , dataH_both], ignore_index=True, sort=True)

# dataG_Retina_predict = dataG_Retina.copy()
dataG_Retina = dataG_Retina.dropna() 
dataH_Retina = dataH_Retina.dropna() 

# dataG_RNFL_predict = dataG_RNFL.copy()
dataG_RNFL = dataG_RNFL.dropna()
dataH_RNFL = dataH_RNFL.dropna()

#合併G 和 H-1的 Retina 和 RNFL
data_Retina  = pd.concat([dataH_Retina , dataG_Retina])
data_RNFL = pd.concat([dataH_RNFL , dataG_RNFL])

#刪除TYPE行中含有0的列並且重新排列index
data_RNFL = data_RNFL.drop(data_RNFL[data_RNFL['TYPE'] == 0].index) 
data_Retina = data_Retina.drop(data_Retina[data_Retina['TYPE'] == 0].index) 
data_Retina = data_Retina.reset_index()
data_RNFL = data_RNFL.reset_index() 

#要統一資料，同一份資料(視網膜以及身經厚度)要來自同一個病人，才能比較，所以統一使用data_both
data_both = pd.concat([data_both, pd.DataFrame(columns=['UM'])], sort=False)
data_both['UM'] = data_both.apply(lambda x: (2*x['UI'] + 6.75 * x['UO'])/(2 + 6.75), axis=1)
data_both = pd.concat([data_both, pd.DataFrame(columns=['LM'])], sort=False)
data_both['LM'] = data_both.apply(lambda x: (2*x['LI'] + 6.75  * x['LO'])/(2 + 6.75), axis=1)  

data_both = pd.concat([data_both, pd.DataFrame(columns=['WN_RNFL'])], sort=False)
data_both['WN_RNFL'] = data_both.apply(lambda x: 15*(2*x['N'] + x['NS'] + x['NI'])/8, axis=1)
data_both = pd.concat([data_both, pd.DataFrame(columns=['WT_RNFL'])], sort=False)
data_both['WT_RNFL'] = data_both.apply(lambda x: 15*(2*x['T'] + x['TS'] + x['TI'])/8, axis=1)

data_both = pd.concat([data_both, pd.DataFrame(columns=['WA_RNFL'])], sort=False)
data_both['WA_RNFL'] = data_both.apply(lambda x: (4*x['WN_RNFL'] + x['G'] + 4*x['WT_RNFL'])/9, axis=1)

data_both = pd.concat([data_both, pd.DataFrame(columns=['WA_Retina'])], sort=False)
data_both['WA_Retina'] = data_both.apply(lambda x: (x['UM'] + x['LM'])/2, axis=1)

data_both = data_both.drop(data_both[data_both['TYPE'] == 0].index) 
data_both = data_both.reset_index()

#按病人分類
check_number_data = True
name_list = []
data_train = pd.DataFrame()

while(check_number_data):
    check_name = True
    rd = random.randint(0,len(data_both)-1)
    tmp = data_both.loc[[rd]]
    tmp_name = tmp["姓名"].tolist()[0]
    
    for i in name_list:
        if(i == tmp_name):
            check_name = False
            break
        
    if(check_name):    
        name_list.append(tmp_name)
        filter_data = (data_both["姓名"] == tmp_name)
        tmp_data = data_both[filter_data]
        data_train = pd.concat([data_train , tmp_data])
    if(len(data_train) >= int(len(data_both)*0.8)):
        check_number_data = False
        
data_test = data_both[~data_both.index.isin(data_train.index)]

#按張數分類
data_train = data_both.sample(frac=0.8)
data_test = data_both[~data_both.index.isin(data_train.index)]


# TYPE的分類，-1 :有病，1：沒病
# type_dataG_Retina = dataG_Retina.groupby("TYPE")
#type_data_both = data_both.groupby("姓名")
#type_data_Retina = data_Retina.groupby("TYPE")
#type_data_RNFL = data_RNFL.groupby("TYPE")

# print(type_dataG_Retina.size())
#print(type_data_both.size())
#print(type_data_Retina.size())
#print(type_data_RNFL.size())

#############################################################################################

#data_G_mean = data_both['G'].mean()
#data_G_std = data_both['G'].std()
#print(data_G_mean)
#print(data_G_std)

#############################################################################################

#pars = [16,      #Number of Neural 
#        4,      #Number of hidden layers
#        16,      #Number of Kernel
#        ]      
#print("輸入尺寸:({0[0]},{0[1]}), \n色彩頻道數:{0[2]}, \n卷積層數:{0[3]}, \n每層池化次數:{0[4]}, \n"
#      "卷積核尺寸d(d*d):{0[5]}*{0[5]},\n池化尺寸p(p*p):{0[6]}*{0[6]}, \n"
#      "卷積核數:{0[7]},\n隱藏層數:{0[8]}, \n神經元數:{0[9]},\n"
#      "BatchSize:{0[10]}, \nNoBatch:{0[11]}, \nepoch:{0[12]}, \n"
#      "策略:{0[13]}, \n區隔數:{0[14]}, \n重複試驗次數:{0[15]}".format(pars))


############################################################################################

#import cv2
#import os 
#from os import listdir
#from os.path import join
#
#cd_G = 'G-retinex//'
#cd_H = 'Health-retinex//'
#pictures_G = listdir(cd_G)
#pictures_H = listdir(cd_H)
#
#Images =[]
#for i in range(len(listdir(cd_G))):
#    Image = cv2.imread(join(cd_G,pictures_G[i]))
#    Images.append(Image)
#    
#for i in range(len(listdir(cd_H))):
#    Image = cv2.imread(join(cd_H,pictures_H[i]))
#    Images.append(Image)
#    

############################################################################################

def InputDataType1():
    dataX_train = data_train[["UO", "UI", "LI", "LO"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["UO", "UI", "LI", "LO"]]
    dataY_test = data_test["TYPE"]
    IDT = ['Retina("UO", "UI", "LI", "LO")']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

def InputDataType2():
    dataX_train = data_train[["UO", "UI", "LI", "LO", "UM", "LM"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["UO", "UI", "LI", "LO", "UM", "LM"]]
    dataY_test = data_test["TYPE"]
    IDT = ['Retina + UM + LM ']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

def InputDataType3():
    dataX_train = data_train[["UM", "LM"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["UM", "LM"]]
    dataY_test = data_test["TYPE"]
    IDT = ['UM + LM']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

def InputDataType4():
    dataX_train = data_train[["UO", "UI", "LI", "LO", "T", "TS", "NS", "N", "NI", "TI", "G"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["UO", "UI", "LI", "LO", "T", "TS", "NS", "N", "NI", "TI", "G"]]
    dataY_test = data_test["TYPE"]
    IDT = ['Retina + RNFL']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

def InputDataType5():
    dataX_train = data_train[["G"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["G"]]
    dataY_test = data_test["TYPE"]
    IDT = ['G']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

def InputDataType6():
    dataX_train = data_train[["WN_RNFL", "WT_RNFL"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["WN_RNFL", "WT_RNFL"]]
    dataY_test = data_test["TYPE"]
    IDT = ['WN_RNFL + WT_RNFL']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

def InputDataType7():
    dataX_train = data_train[["T", "TS", "NS", "N", "NI", "TI", "G"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["T", "TS", "NS", "N", "NI", "TI", "G"]]
    dataY_test = data_test["TYPE"]
    IDT = ['RNFL("T", "TS", "NS", "N", "NI", "TI", "G")']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

def InputDataType8():
    dataX_train = data_train[["T", "TS", "NS", "N", "NI", "TI", "G", "UM", "LM"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["T", "TS", "NS", "N", "NI", "TI", "G", "UM", "LM"]]
    dataY_test = data_test["TYPE"]
    IDT = ['RNFL + UM + LM']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

def InputDataType9():
    dataX_train = data_train[["WN_RNFL", "WT_RNFL", "G"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["WN_RNFL", "WT_RNFL", "G"]]
    dataY_test = data_test["TYPE"]
    IDT = ['WN_RNFL + WT_RNFL + G ']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

def InputDataType10():
    dataX_train = data_train[["WN_RNFL", "WT_RNFL", "G", "UM", "LM"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["WN_RNFL", "WT_RNFL", "G", "UM", "LM"]]
    dataY_test = data_test["TYPE"]
    IDT = ['WN_RNFL + WT_RNFL + G + UM + LM']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

def InputDataType11():
    dataX_train = data_train[["WA_Retina"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["WA_Retina"]]
    dataY_test = data_test["TYPE"]
    IDT = ['WA_Retina']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

def InputDataType12():
    dataX_train = data_train[["WA_RNFL", "UM", "LM"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["WA_RNFL", "UM", "LM"]]
    dataY_test = data_test["TYPE"]
    IDT = ['WA_RNFL + UM + LM']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

def InputDataType13():
    dataX_train = data_train[["WA_RNFL"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["WA_RNFL"]]
    dataY_test = data_test["TYPE"]
    IDT = ['WA_RNFL']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

def InputDataType14():
    dataX_train = data_train[["WA_RNFL", "WA_Retina"]]
    dataY_train = data_train["TYPE"]
    dataX_test = data_test[["WA_RNFL", "WA_Retina"]]
    dataY_test = data_test["TYPE"]
    IDT = ['WA_RNFL + WA_Retina']
    return (dataX_train, dataX_test, dataY_train, dataY_test, IDT)

#
def Default():
    return "Invalid Strategy"

InputDataTypedict = {
    1: InputDataType1,
    2: InputDataType2,
    3: InputDataType3,
    4: InputDataType4,
    5: InputDataType5,
    6: InputDataType6,
    7: InputDataType7,
    8: InputDataType8,
    9: InputDataType9,
    10: InputDataType10,
    11: InputDataType11,
    12: InputDataType12,
    13: InputDataType13,
    14: InputDataType14
}
#
#
def getStrategy(Strategy):
    fun = InputDataTypedict.get(Strategy, Default)
    return fun()
#
#
#
##############################################################################################
#def plot_svc_decision_function(model, ax=None, plot_support=True):
#    """Plot the decision function for a 2D SVC"""
#    if ax is None:
#        ax = plt.gca()
#    xlim = ax.get_xlim()
#    ylim = ax.get_ylim()
#    
#    # create grid to evaluate model
#    x = np.linspace(xlim[0], xlim[1], 30)
#    y = np.linspace(ylim[0], ylim[1], 30)
#    Y, X = np.meshgrid(y, x)  
#    xy = np.vstack([X.ravel(), Y.ravel()]).T
#    P = svm.decision_function(xy).reshape(X.shape)
#    
#    # plot decision boundary and margins
#    ax.contour(X, Y, P, colors='k',
#               levels=[-1, 0, 1], alpha=0.5,
#               linestyles=['--', '-', '--'])
#    
#    # plot support vectors
#    if plot_support:
#        ax.scatter(svm.support_vectors_[:, 0],
#                   svm.support_vectors_[:, 1],
#                   s=300, linewidth=1, facecolors='black');
#    ax.set_xlim(xlim)
#    ax.set_ylim(ylim)
    
##strategy = input('請輸入你的策略[數字(1-6)]:')
##(dataX_train, dataX_test, dataY_train, dataY_test) = getStrategy(int(strategy))
##
##dataY_test = dataY_test.astype(int)
##dataY_train = dataY_train.astype(int)
#
for i in range(len(InputDataTypedict)):
    IDT = []
    (dataX_train, dataX_test, dataY_train, dataY_test, IDT) = getStrategy(i+1)
    
    dataY_test = dataY_test.astype(int)
    dataY_train = dataY_train.astype(int)
    
    svm = SVC(kernel =  'linear', probability = True)
    svm.fit(dataX_train, dataY_train)
    Y_SVM_predict = svm.predict(dataX_test)
    
    
#    plt.figure(figsize = (10,8))
#    plt.scatter(np.linspace(1, len(dataX_train), len(dataX_train)), dataX_train, c= dataY_train, cmap='autumn');  
#    plot_svc_decision_function(svm)
                
    TN = 0
    TP = 0
    FP_TN = 0
    TP_FN = 0
    j = 0
    for i in dataY_test:
        #如果真實是沒病也預測沒病,則H+1
        if(i == -1):
            FP_TN += 1
            if(Y_SVM_predict[j] == -1):
                TN += 1
    #    如果真實是有病也預測有病,則G+1
        if(i == 1):
            TP_FN += 1
            if(i == Y_SVM_predict[j]):
                TP += 1 
        j+=1
    
    FP = FP_TN - TN
    FN = TP_FN - TP
    
    Specificity = TN/FP_TN
    Sensitivity = TP/TP_FN
    Accuracy = (TN + TP)/(FP_TN + TP_FN)
    Precision = TP / (TP+FP)
    print('----------------------------------------------')
    print('InputDataType:{}'.format(IDT))
    print('Classifier :  SVM(Python)')
#    print('Number of Layers:{}'.format(mlp.n_layers_))
#    print('Number of Nodes:{}'.format(Nodes))
    print('Number of Glaucoma:{}'.format(TP_FN))
    print('Number of Normal:{}'.format(FP_TN))
    print('Specificity: {:.2f}'.format(Specificity))
    print('Sensitivity: {:.2f}'.format(Sensitivity))
    print('Accuracy: {:.2f}'.format(Accuracy))
    print('Precision:{:.2f}'.format(Precision))
    



    Nodes = [9,9]
    mlp = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=Nodes, random_state=1, max_iter = 500, learning_rate = 'adaptive')
    mlp.fit(dataX_train, dataY_train)     
    
    Y_ANN_predict = mlp.predict(dataX_test)
    #print(mlp.score(dataX_test, dataY_test))                  
    TN = 0
    TP = 0
    FP_TN = 0
    TP_FN = 0
    j = 0
    for i in dataY_test:
        #如果真實是沒病也預測沒病,則H+1
        if(i == -1):
            FP_TN += 1
            if(Y_ANN_predict[j] == -1):
                TN += 1
    #    如果真實是有病也預測有病,則G+1
        if(i == 1):
            TP_FN += 1
            if(i == Y_ANN_predict[j]):
                TP += 1 
        j+=1
    
    FP = FP_TN - TN
    FN = TP_FN - TP
    
    Specificity = TN/FP_TN
    Sensitivity = TP/TP_FN
    Accuracy = (TN + TP)/(FP_TN + TP_FN)
    Precision = TP / (TP+FP)
    print('----------------------------------------------')
    print('InputDataType:{}'.format(IDT))
    print('Classifier :  ANN(Python)')
    print('Number of Layers:{}'.format(mlp.n_layers_))
    print('Number of Glaucoma:{}'.format(TP_FN))
    print('Number of Normal:{}'.format(FP_TN))
    print('Specificity: {:.2f}'.format(Specificity))
    print('Sensitivity: {:.2f}'.format(Sensitivity))
    print('Accuracy: {:.2f}'.format(Accuracy))
    print('Precision:{:.2f}'.format(Precision))
#    
#    print('Number of Nodes:{}'.format(Nodes))
#    print('Number of Glaucoma:{}'.format(TP_FN))
#    print('Number of Normal:{}'.format(FP_TN))
#    print('Specificity: {:.2f}'.format(Specificity))
#    print('Sensitivity: {:.2f}'.format(Sensitivity))
#    print('Accuracy: {:.2f}'.format(Accuracy))
#    print('Precision:{:.2f}'.format(Precision))
    
#subrouting from internet(check correct)
#    print('Precision: {:.2f}'.format(precision_score(dataY_test, Y_predict)))
#    print('Recall: {:.2f}'.format(recall_score(dataY_test, Y_predict)))
#    print('Accuracy: {:.2f}'.format(mlp.score(dataX_test, dataY_test)))
#model = Sequential()
#model = Sequential()
#model.add(Dense(4, input_dim=4, activation='sigmoid'))
#model.add(Dense(8))
#model.add(Dense(4))
#model.add(Dense(2))
#model.add(Dense(1, activation='sigmoid'))
##model.add(Dense(1, activation='sigmoid'))
#
#sgd = optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
#model.compile(loss='mean_squared_error', metrics=['accuracy'])
#model.fit(dataX_train, dataY_train, epochs=10000, batch_size=20)
#
#_, accuracy = model.evaluate(dataX_test, dataY_test)
#print('Accuracy: %.2f' % (accuracy*100))
    
#mlp = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(15, 15), random_state=1)
#mlp.fit(dataX_train, dataY_train)     
#
#Y_predict = mlp.predict(dataX_test)
##print(mlp.score(dataX_test, dataY_test))                  
#print(mlp.n_layers_)
#print(mlp.n_iter_)
#print(mlp.loss_)
#print(mlp.out_activation_)

# from sklearn.model_selection import GridSearchCV
# param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
# grid = GridSearchCV(SVC(),param_grid,verbose=3)
# grid.fit(dataX_train, dataY_train)
# grid_predictions = grid.predict(dataX_test)

# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(dataY_test,predictions))
# print('\n')
# print(classification_report(dataY_test,predictions))
#
# print(confusion_matrix(dataY_test,grid_predictions))
# print('\n')
# print(classification_report(dataY_test,grid_predictions))


