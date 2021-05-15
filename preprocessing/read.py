# Sofie Sunde - Spring 2021
# Read from dataset, save and load dataframe
# Code structure inspired by Karoline Bonnerud - https://github.com/karolbon/bot-or-human

import json
import numpy as np
import pandas as pd
import pickle

#def readTrainingSet():
 #   with open("/Users/sofiesunde/GitHub/FinNum/datasets/FinNum_training_rebuilded.json", "r", encoding="latin-1") as file:
  #      document = json.load(file)
   # trainingSet = []
    #for tweet in document:
     #   trainingSet.append(tweet)
    #return trainingSet

def readDocument(filepath):
    with open(filepath, 'r', encoding='latin-1') as file:
        document = json.load(file)
    dataframe = pd.DataFrame(document)
    dataframe.columns = ['idx', 'id', 'target_num', 'category', 'subcategory', 'tweet']
    index = []
    for i in range(len(dataframe)):
        index.append(i+1)
    dataframe.index = [index]
    # Remove idx, id and subcategory from dataframe, inspiration from https://www.educative.io/edpresso/how-to-delete-a-column-in-pandas
    dataframe.drop(['idx', 'id', 'subcategory'], inplace=True, axis=1)

    #dataframe.loc[len(dataframe['target_num']) != 1]

    return dataframe

# spørs hvilket test set du bruker, test set fra 2020 ligger i datasets mappen
def readTestSet(filepath):
    with open(filepath, 'r', encoding='latin-1') as file:
        document = json.load(file)
    dataframe = pd.DataFrame(document)
    dataframe.columns = ['idx', 'id', 'target_num', 'tweet']
    return dataframe

def saveDataframe(dataframe, filename):
    dataframe.to_pickle('datasets/' + filename + '.txt')

def loadDataframe(filename):
    dataframe = pd.read_pickle('datasets/' + filename + '.txt')
    return dataframe

