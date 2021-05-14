# Sofie Sunde - Spring 2021
# Preprocess data, process features
# Code structure inspired by Karoline Bonnerud - https://github.com/karolbon/bot-or-human

from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from preprocessing.read import saveDataframe, loadDataframe
import numpy as np
import pandas as pd
import pickle

# Remove stopwords (inspired by NLTK, Chapter 4.1 - Wordlist Corpora)
def removeStopwords(document):
    englishStopwords = stopwords.words('english')
    documentList = document.split(' ')
    wordDocument = [word for word in documentList if word.lower() not in englishStopwords]
    wordDocument = ' '.join(wordDocument)
    return wordDocument

# Normalization
# vurdere å fjerne komma osv? kan bruke tweet processor istedenfor evt. må bare passe på $
# må kanskje tokenization??
def preProcess(document):
    preProcessed = document.lower()
    preProcessed = preProcessed.replace('\n', ' ')
    preProcessed = removeStopwords(preProcessed)
    # substitute https into URL, @ into UserReply, and removes them
    #preProcessed = re.sub('https([^\s]+)', '<URL>', preProcessed)
    #preProcessed = re.sub('<URL>', '', preProcessed)
    preProcessed = re.sub('https([^\s]+)', '', preProcessed)
    #preProcessed = re.sub('@([^\s]+)', '<UserReply>', preProcessed)
    #preProcessed = re.sub('<UserReply>', '', preProcessed)
    preProcessed = re.sub('@([^\s]+)', '', preProcessed)
    # substitute $cashtag into CashTag, and removes it
    #preProcessed = re.sub('\$[A-Za-z]([^\s]+)', '<CashTag>', preProcessed)
    #preProcessed = re.sub('<CashTag>', '', preProcessed)
    preProcessed = re.sub('\$[A-Za-z]([^\s]+)', '', preProcessed)
    return preProcessed

document = "@alexandra Hi my name is Sofie SUnde and I believe this link should be removed, https://www.google.com/?client=safari, although this $ADSD is not the same as this $123"


#print(preProcess(document))

# Evaluate features

#def multipleTargetNums(dataframe):
 #   if len(target_num) > 1:
  #      for label in target_num:
   #     dataframe[['targetnum1', 'targetnum2']] = dataframe['target_num'](dataframe.teams.tolist(), index=dataframe.index)

# ide : lage alt som liste med 7 felter der det er 0 eller 1 på de kategoriene,
# tar ikke høyde for at tweeten kan ha flere tall slik at det skal være flere kategorier....

# lister inni liste, klarer den ikke uansett om det er liste med tall eller binær eller liste inni liste, rart, fordi numToCategory er jo samme som target num?
def categoryToNum(category):
    # One hot encode
    n = len(category)
    categories = np.zeros((n, 7), dtype=int).tolist()
    for i, label in enumerate(category):
        if label == 'Monetary':
            categories[i][0] = 1

        elif label == 'Percentage':
            categories[i][1] = 1

        elif label == 'Option':
            categories[i][2] = 1

        elif label == 'Indicator':
            categories[i][3] = 1

        elif label == 'Temporal':
            categories[i][4] = 1

        elif label == 'Quantity':
            categories[i][5] = 1

        elif label == 'Product Number':
            categories[i][6] = 1

        #categories = map(float, categories[i])
    print('Categories list:')
    print(categories)
    return categories

def numToCategory(category):
    categories = []
    #map(float, categories[i].split(","))
    for label in category:
        if label == 'Monetary':
            categories.append(1)
            #return 1
        elif label == 'Percentage':
            categories.append(2)
            #return 2
        elif label == 'Option':
            categories.append(3)
            #return 3
        elif label == 'Indicator':
            categories.append(4)
            #return 4
        elif label == 'Temporal':
            categories.append(5)
            #return 5
        elif label == 'Quantity':
            categories.append(6)
            #return 6
        elif label == 'Product Number':
            categories.append(7)
            #return 7
    print(categories)
    return categories

# her får man ikke skilt på hvilket target num som er hvilken kategori:OO
def categorize(category):
    categories = [0, 0, 0, 0, 0, 0, 0]
    #map(float, categories[i].split(","))
    for label in category:
        if label == 'Monetary':
            categories[0] = 1
            #return 1
        elif label == 'Percentage':
            categories[1] = 1
            #return 2
        elif label == 'Option':
            categories[2] = 1
            #return 3
        elif label == 'Indicator':
            categories[3] = 1
            #return 4
        elif label == 'Temporal':
            categories[4] = 1
            #return 5
        elif label == 'Quantity':
            categories[5] = 1
            #return 6
        elif label == 'Product Number':
            categories[6] = 1
            #return 7
    print(categories)
    return categories

def categoryToBinary(category):
    categories = []
    for label in category:
        if label == 'Monetary':
            categories.append('{:04b}'.format(1))
            #return 1
        elif label == 'Percentage':
            categories.append('{:04b}'.format(2))
            #return 2
        elif label == 'Option':
            categories.append('{:04b}'.format(3))
            #return 3
        elif label == 'Indicator':
            categories.append('{:04b}'.format(4))
            #return 4
        elif label == 'Temporal':
            categories.append('{:04b}'.format(5))
            #return 5
        elif label == 'Quantity':
            categories.append('{:04b}'.format(6))
            #return 6
        elif label == 'Product Number':
            categories.append('{:04b}'.format(7))
            #return 7
    print(categories)
    return categories


#def categorizationFeatures(dataframe):
# Find all digits in tweet -> må ta riktig target num til riktig tekstforklaring på en eller annen måte :OOO
#y = [X: $1.19] [Y: $5.29] [Z 999/1000]
#x = re.findall(r"\$[^ ]+", y)
#ha med klassifiseringsregler her???? % osv?


# TF-IDF
# min_df can be adjusted, 1 = standard
# stop_words kan fjernes
# ngrams kan gjøres om til kun bare unigrams, som er default

def tfidf(dataframe, training, validation):
    if training:
        print('tfidf started')
        print(dataframe['tweet'])
        tfidf = TfidfVectorizer(stop_words='english', min_df=0.01, max_df=0.9,  ngram_range=(1, 3))
        p = tfidf.fit_transform(dataframe['tweet'])
        print(p)
        #dataframe._tfidf = tfidf
        pickle.dump(tfidf, open('datasets/tfidf.txt', 'wb'))
        #pickle.dump(tfidf, open('datasets/tfidf.json', 'wb'))
        # tfidfvectorizer object unable to serialize, json file unattainable
        #tfidfJson = json.dumps(tfidf)
        #json.dump(tfidf, open('.//attributes.json', 'w'))
        #saveDataframe(tfidfJson, 'tfidf')
    else:
        print('loading tfidf started')
        #tfidf = pickle.load(open('datasets/tfidf.txt', 'wb'))
        #tfidf = json.load(open('datasets/tfidf.json', 'wb'))
        #tfidf = loadDataframe('tfidf')
        #dataframe['tweet'] = tfidf.transform(dataframe['tweet'])
    return dataframe

# dersom du vil jobbe med kun tweets med 1 tall
def multipleProcessing(dataframe, multiple):
    n = len(multiple)
    print(dataframe['index'])
    for label in enumerate(multiple):
        if n > 1:
            dataframe.drop(dataframe['index'])

# Get names of indexes for which column Age has value 30
#indexNames = dfObj[ dfObj['Age'] == 30 ].index
# Delete these row indexes from dataFrame
#dfObj.drop(indexNames , inplace=True)


def multipleProcessing1(dataframe):
    print(dataframe)
    nums = dataframe['target_num']
    counter = 0
    for i, num in nums:
        counter += 1
        if len(num) > 1:
            #multiple = dataframe[num].index
            dataframe.drop(dataframe['index'] == i)


        #multiple = dataframe[len(num) > 1]
        #dataframe.drop(multiple, inplace=True)
    #for label in dataframe['target_num']:
     #   if len(label) > 1:
      #      dataframe.drop(dataframe['index'])
    return dataframe

# Feature Engineering
def featureEngineering(dataframe, training, validation):
    #dataframe = multipleProcessing(dataframe)
    # Preprocess tweet
    dataframe['tweet'] = dataframe['tweet'].apply(lambda tweet: preProcess(tweet))
    #dataframe['target_num'] = dataframe['tweet'].apply(lambda num: multipleTargetNums(num))
    #dataframe = dataframe.drop(dataframe[len(dataframe['target_num'] > 1)].index, inplace=True)
    #dataframe = multipleProcessing1(dataframe)
    #dataframe['target_num'] = dataframe['target_num'].apply(lambda num: multipleProcessing(dataframe, num))
    #dataframe['target_num'] = dataframe['target_num'].apply(lambda dataframe: multipleProcessing(dataframe, num))
    #dataframe[['target_num', 'category']] = dataframe[['target_num', 'category']].apply(lambda num: len(num.split(',')) < 2)
    print('dataframe with only one num in tweet')
    print(dataframe)
    #print(df[df['alfa'].apply(lambda x: len(x.split(',')) < 3)])
    # Categorization on dataframe
    dataframe['category'] = dataframe['category'].apply(lambda category: numToCategory(category))
    # TFIDF on dataframe
    featureDataframe = tfidf(dataframe, training, validation)
    return featureDataframe

#dataframe = loadDataframe('package.json')
#featureEngineering(dataframe, training=True)