# Sofie Sunde - Spring 2021
# Read from dataset, save and load dataframe
# Code structure inspired by Karoline Bonnerud - https://github.com/karolbon/bot-or-human

import json
import pandas as pd

def readDocument(filepath):
    with open(filepath, 'r', encoding='latin-1') as file:
        document = json.load(file)
    dataframe = pd.DataFrame(document)
    dataframe.columns = ['idx', 'id', 'target_num', 'category', 'subcategory', 'tweet']
    #index = []
    #for i in range(len(dataframe)):
    #    index.append(i+1)
    #dataframe.index = [index]
    # Remove idx, id and subcategory from dataframe, inspiration from https://www.educative.io/edpresso/how-to-delete-a-column-in-pandas
    dataframe.drop(['idx', 'id', 'subcategory'], inplace=True, axis=1)

    #dataframe.loc[len(dataframe['target_num']) != 1]

    return dataframe

# Test set from 2020 used, where the categories are included, so this method is not used, as of now
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
