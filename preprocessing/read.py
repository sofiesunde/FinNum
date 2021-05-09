# code to read from dataset
import json
import numpy as np
import pandas as pd

#def readTrainingSet():
 #   with open("/Users/sofiesunde/GitHub/FinNum/datasets/FinNum_training_rebuilded.json", "r", encoding="latin-1") as file:
  #      document = json.load(file)
   # trainingSet = []
    #for tweet in document:
     #   trainingSet.append(tweet)
    #return trainingSet

def readDocument(filepath):
    with open(filepath, "r", encoding="latin-1") as file:
        document = json.load(file)
    dataframe = pd.DataFrame(document)
    print(dataframe.head())
    return dataframe

# Inspirasjon fra lab 5 løsningsforslag
def loadTweets(filepath):
    #Load + preprocessing
    #For your own sake if you're gonna run this thing, please add nrows=10000 to the read_csv function 🤦‍♂️
    document = pd.read_json(filepath, encoding="latin-1")
    return document
    #document['text'] = document['text'].apply(lambda x: x.lower())
    ##data['target'] = np.array(data['text'].apply(sentimentTweets))

# inspirasjon fra kode på git
def saveDataframe(dataframe, filename):
    dataframe.to_pickle("datasets/" + filename + ".pkl")

def loadDataframe(filename):
    dataframe = pd.read_pickle("datasets/" + filename + ".pkl")
    return dataframe


#print(readTrainingSet()[150])

#print(readDocument("/Users/sofiesunde/GitHub/FinNum/datasets/FinNum_training_rebuilded.json")[150])