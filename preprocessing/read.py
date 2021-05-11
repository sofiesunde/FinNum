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
    # Remove subcategory from dataframe, inspiration from https://www.educative.io/edpresso/how-to-delete-a-column-in-pandas
    dataframe.drop('subcategory', inplace=True, axis=1)
    return dataframe

# spørs hvilket test set du bruker, test set fra 2020 ligger i datasets mappen
def readTestSet(filepath):
    with open(filepath, 'r', encoding='latin-1') as file:
        document = json.load(file)
    dataframe = pd.DataFrame(document)
    dataframe.columns = ['idx', 'id', 'target_num', 'tweet']
    return dataframe

# Inspirasjon fra lab 5 løsningsforslag
# denne brukes ikke akk nå
def loadTweets(filepath):
    #Load + preprocessing
    #For your own sake if you're gonna run this thing, please add nrows=10000 to the read_csv function 🤦‍♂️
    document = pd.read_json(filepath)#, encoding='latin-1')
    return document
    #document['text'] = document['text'].apply(lambda x: x.lower())
    ##data['target'] = np.array(data['text'].apply(sentimentTweets))

# inspirasjon fra kode på git
def saveDataframe(dataframe, filename):
    dataframe.to_pickle('datasets/' + filename + '.txt')
    #dataframe.to_json('datasets/' + filename + '.json')

def loadDataframe(filename):
    dataframe = pd.read_pickle('datasets/' + filename + '.txt')
    #dataframe = pd.read_json('datasets/' + filename + '.json')
    return dataframe

#print(readTrainingSet()[150])

#print(readDocument("/Users/sofiesunde/GitHub/FinNum/datasets/FinNum_training_rebuilded.json")[150])