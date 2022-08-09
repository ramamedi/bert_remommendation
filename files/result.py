import collections
from distutils.command.build import build
from gc import collect
from typing import Type
import pymongo 
from pymongo import MongoClient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



client = pymongo.MongoClient("mongodb+srv://ramam:ramnight@cluster0.wy9d4.mongodb.net/?retryWrites=true&w=majority")
db = client.netflix
collection = db.movies



def clean_category_data(txt):
    array = txt.replace(",","").split()
    if 'Movies' in array:
        array.remove('Movies')
    if '&' in array:
        array.remove('&')
    return array

def check_intersection(categories1 , categories2):
    flag = False
    for cat in categories1:
        if cat in categories2:
            flag = True
    return flag


def data_distribution_pie():
    positive_data = collection.count_documents({'netflix':1})
    negative_data =  collection.count_documents({'netflix':0}) 

    numberOfRecords = positive_data + negative_data
    labels = "Similar", "Not similar"
    sizes = [positive_data, negative_data]
    explode = (0.1, 0)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True,colors=['#E50914', 'gray'] , startangle=90)
    ax1.axis('equal')  
    plt.title('Data distribution')
    plt.xlabel(str(numberOfRecords) + ' Movies')
    plt.savefig('pie.png')
    plt.clf()

#build confusion matrix
def build_confusion_mat_for_same_genre(decisionBoundary, typeOfScore = 'pearson'):   
    data = collection.find({})
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0
    numberOfChecked = 0
    print(typeOfScore)
    for i, row in enumerate(data):
        if check_intersection(clean_category_data(row['second_movie_genre']), clean_category_data(row['first_movie_genre'])):
            numberOfChecked += 1
            if row[typeOfScore] >= decisionBoundary and row['netflix'] == 1:
                truePositive += 1
            if row[typeOfScore] >= decisionBoundary and row['netflix'] == 0:
                falsePositive += 1
            if row[typeOfScore] <= decisionBoundary and row['netflix'] == 1:
                falseNegative += 1
            if row[typeOfScore] <= decisionBoundary and row['netflix'] == 0:
                trueNegative += 1
    
    cf_matrix = [[truePositive, falseNegative], 
    [falsePositive, trueNegative]]
    print(cf_matrix)
    return {'TP': truePositive , 'FP': falsePositive , 'TN':trueNegative, 'FN': falseNegative, 'cf_matrix': cf_matrix}

def build_confusion_mat(decisionBoundary , typeOfScore = 'pearson'):   
    data = collection.find({})
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0
    for i, row in enumerate(data):
        if row[typeOfScore] >= decisionBoundary and row['netflix'] == 1:
            truePositive += 1
        if row[typeOfScore] >= decisionBoundary and row['netflix'] == 0:
            falsePositive += 1
        if row[typeOfScore] <= decisionBoundary and row['netflix'] == 1:
            falseNegative += 1
        if row[typeOfScore] <= decisionBoundary and row['netflix'] == 0:
            trueNegative += 1

    cf_matrix = [[truePositive, falseNegative], 
                 [falsePositive, trueNegative]]
                 
    return {'TP': truePositive , 'FP': falsePositive , 'TN':trueNegative, 'FN': falseNegative, 'cf_matrix': cf_matrix}

def print_cf_matrix(decisionBoundary, same_genre = False):
    #same genre build confusion matrix only if two movies have same genre
    if same_genre:
        cf_matrix = build_confusion_mat_for_same_genre(decisionBoundary)['cf_matrix']
    else:
        cf_matrix = build_confusion_mat(decisionBoundary)['cf_matrix']
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',  fmt='g')
    ax.set_title('Confusion Matrix for desicion boundery:'+str(decisionBoundary)+'\n\n');
    ax.xaxis.tick_top()
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticklabels(['True','False'])
    ax.yaxis.set_ticklabels(['True','False'])
    if same_genre:
        plt.savefig('cf_sameGenre'+str(decisionBoundary)+'.png')
    else: 
        plt.savefig('cf_'+str(decisionBoundary)+'.png') 
    ax.clear()
 
def run_over_desicion_boundry():
    for decisionBoundary in np.linspace(0,1,11):
        print_cf_matrix(decisionBoundary)
       
def get_precision_recall(decisionBoundary, same_genre = False, typeOfScore = 'pearson'):
    if same_genre:
        conf_mat =  build_confusion_mat_for_same_genre(decisionBoundary, typeOfScore)
    else: 
        conf_mat =  build_confusion_mat(decisionBoundary, typeOfScore)
    precision = conf_mat['TP']/(conf_mat['TP'] + conf_mat['FP'])
    recall = conf_mat['TP']/(conf_mat['TP'] + conf_mat['FN'])
    accurecy = (conf_mat['TP'] + conf_mat['TN'])/(conf_mat['TP'] + conf_mat['FN'] + conf_mat['TN'] + conf_mat['FP'])
    return {'precision': "{:.2f}".format(precision) , 'recall':"{:.2f}".format(recall), 'accurecy':"{:.2f}".format(accurecy)}

def line_precision_recall(same_genre):
    values = np.array([ 0.4 , 0.5 , 0.6, 0.7, 0.8 ])
    precision_list_cosine = []
    recall_list_cosine = []
    precision_list_pearson = []
    recall_list_pearson = []
    for val in values: 
        precision_list_cosine.append(float(get_precision_recall(val, same_genre, 'cosine')['precision']))
        recall_list_cosine.append(float(get_precision_recall(val, same_genre, 'cosine')['recall']))
        precision_list_pearson.append(float(get_precision_recall(val, same_genre, 'pearson')['precision']))
        recall_list_pearson.append(float(get_precision_recall(val, same_genre, 'pearson')['recall']))

    fig, ax = plt.subplots()
    ax.plot(values, precision_list_cosine, linestyle= 'dashed', color= '#E50914',label = "Cosine Precision")
    ax.plot(values, recall_list_cosine, linestyle= 'dashed', color = 'gray', label = "Cosine Recall")
    ax.plot(values, precision_list_pearson, color= '#E50914',label = "Pearson Precision")
    ax.plot(values, recall_list_pearson, color = 'gray', label = "Pearson Recall")
    ax.spines.left.set_bounds((0, 1))
    ax.spines.bottom.set_bounds((0.4, 0.8))
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xlabel('Desicion boundary')
    ax.set_ylabel('Score')
    plt.legend()
    plt.show()
    # plt.savefig('recall_pressicion_line_'+str(same_genre)+'.png') 

def box_plot_for_similarty():
    pearson = []
    cosine = []

    items = collection.find({})
    for item in items:
        pearson.append(item['pearson'])
        cosine.append(item['cosine'])
        # euclidian_dis.append(item['euclidean'])

    data = [pearson, cosine]
    fig7, ax7 = plt.subplots()
    ax7.set_title('Similarity scores distribution')
    ax7.boxplot(data)
    ax7.spines.left.set_bounds((0.2, 0.9))
    ax7.spines.right.set_visible(False)
    ax7.spines.top.set_visible(False)
    ax7.set_ylabel('Score')
    plt.xticks([1, 2], ['pearson', 'cosine'])
    plt.savefig('similarity_box_plot.png')


def box_plot_for_euc_distance():
    euclidian_dis = []

    items = collection.find({})
    for item in items:
        euclidian_dis.append(item['euclidean'])

    data = [euclidian_dis]
    fig7, ax7 = plt.subplots()
    ax7.set_title('Euclidean distance distribution')
    ax7.boxplot(data)
    ax7.spines.right.set_visible(False)
    ax7.spines.top.set_visible(False)
    ax7.set_ylabel('Distance')
    plt.xticks([1], ['Euclidean distance'])
    plt.savefig('euc_distance_box_plot.png')

def remove_duplicate():
    items = collection.find({'euclidean':0})
    for item in items:
        print(item)
    collection.delete_one({'euclidean':0})
    
def line_same_vs_not():
    values = np.array([ 0.4 , 0.5 , 0.6, 0.7, 0.8 ])
    precision_list_true = []
    recall_list_true = []
    precision_list_false = []
    recall_list_false = []
    for val in values: 
        precision_list_true.append(float(get_precision_recall(val, True, 'cosine')['precision']))
        recall_list_true.append(float(get_precision_recall(val, True, 'cosine')['recall']))
        precision_list_false.append(float(get_precision_recall(val, False, 'cosine')['precision']))
        recall_list_false.append(float(get_precision_recall(val, False, 'cosine')['recall']))

    fig, ax = plt.subplots()
    ax.plot(values, precision_list_true, linestyle= 'dashed', color= '#1AC9E6',label = "Same genre only Precision")
    ax.plot(values, recall_list_true, linestyle= 'dashed', color = '#19AADE', label = "Same genre only Recall")
    ax.plot(values, precision_list_false, color= '#AF4BCE',label = "All data Precision")
    ax.plot(values, recall_list_false, color = '#DB4CB2', label = "All data Recall")
    ax.spines.left.set_bounds((0, 1))
    ax.spines.bottom.set_bounds((0.4, 0.8))
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xlabel('Desicion boundary')
    ax.set_ylabel('Score')
    plt.legend()
    plt.savefig('Same_genre_help'+'.png') 




def build_confusion_for_euc_dis(distance):   
    data = collection.find({})
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0
    numberOfChecked = 0
    for i, row in enumerate(data):
        if check_intersection(clean_category_data(row['second_movie_genre']), clean_category_data(row['first_movie_genre'])):
            numberOfChecked += 1
            if row['euclidean'] <= distance and row['netflix'] == 1:
                truePositive += 1
            if row['euclidean'] <= distance and row['netflix'] == 0:
                falsePositive += 1
            if row['euclidean'] >= distance and row['netflix'] == 1:
                falseNegative += 1
            if row['euclidean'] >= distance and row['netflix'] == 0:
                trueNegative += 1
                
    cf_matrix = [[truePositive, falseNegative], 
                 [falsePositive, trueNegative]]
                 
    return {'TP': truePositive , 'FP': falsePositive , 'TN':trueNegative, 'FN': falseNegative, 'cf_matrix': cf_matrix}

def get_precision_recall_euc_distance(distance):
    conf_mat =  build_confusion_for_euc_dis(distance)
    precision = conf_mat['TP']/(conf_mat['TP'] + conf_mat['FP'])
    recall = conf_mat['TP']/(conf_mat['TP'] + conf_mat['FN'])
    accurecy = (conf_mat['TP'] + conf_mat['TN'])/(conf_mat['TP'] + conf_mat['FN'] + conf_mat['TN'] + conf_mat['FP'])
    return {'precision': "{:.2f}".format(precision) , 'recall':"{:.2f}".format(recall), 'accurecy':"{:.2f}".format(accurecy)}


def line_precision_recall_euc():
    values = range(10,22)
    precision_list = []
    recall_list = []
    for val in values: 
        precision_list.append(float(get_precision_recall_euc_distance(val)['precision']))
        recall_list.append(float(get_precision_recall_euc_distance(val)['recall']))


    fig, ax = plt.subplots()
    ax.plot(values, precision_list, linestyle= 'dashed', color= '#19AADE',label = "Euclidean distance Precision")
    ax.plot(values, recall_list, color = '#DB4CB2', label = "Euclidean distance Recall")
    ax.spines.left.set_bounds((0, 1))
    ax.spines.bottom.set_bounds((8, 22))
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xlabel('Desicion boundary')
    ax.set_ylabel('Score')
    plt.legend()
    # plt.show()
    plt.savefig('eucDistancePrecision'+'.png') 



def print_euc_confusion_matrix(distance):
    cf_matrix =  build_confusion_for_euc_dis(distance)['cf_matrix']
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',  fmt='g')
    ax.set_title('Confusion Matrix for Euclidean distance:\n\n');
    ax.xaxis.tick_top()
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticklabels(['True','False'])
    ax.yaxis.set_ticklabels(['True','False'])
    plt.savefig('cf_eucdis'+'.png') 
    ax.clear()
 

def average(data):
    return sum(data)/len(data)

def dict_of_distribution(data):
    return {'IQR': np.percentile(data,75) - np.percentile(data, 25) , 'q1':np.percentile(data, 25), 'median': np.percentile(data, 50), 'q3':np.percentile(data, 75), 'avg':sum(data)/len(data) }

