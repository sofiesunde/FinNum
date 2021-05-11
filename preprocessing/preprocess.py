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
    #preProcessed = removeStopwords(preProcessed)
    # substitute https into URL, @ into UserReply, and removes them
    preProcessed = re.sub('https([^\s]+)', '<URL>', preProcessed)
    preProcessed = re.sub('<URL>', '', preProcessed)
    preProcessed = re.sub('@([^\s]+)', '<UserReply>', preProcessed)
    preProcessed = re.sub('<UserReply>', '', preProcessed)
    # substitute $cashtag into CashTag, and removes it
    preProcessed = re.sub('\$[A-Za-z]([^\s]+)', '<CashTag>', preProcessed)
    preProcessed = re.sub('<CashTag>', '', preProcessed)

    return preProcessed

document = "@alexandra Hi my name is Sofie SUnde and I believe this link should be removed, https://www.google.com/?client=safari, although this $ADSD is not the same as this $123"

#print(preProcess(document))

# Evaluate features
# tar ikke høyde for at tweeten kan ha flere tall slik at det skal være flere kategorier
def categoryToNum(category):
    print(category)
    categories = []
    for label in category:
        #print(label)
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
        else:
            categories.append(0)
            #return 0
    return categories

#CountVectorizer???????????

#Inspirasjon fra løsningsforslag lab 5
#def categorizeTweets(dataframe):
    #dataframe['category'] = np.array(dataframe['tweet'].apply(HVA SKAL VÆRE HER?!?!?!?!???!?!?!))

#def categorizationFeatures(dataframe):
# Find all digits in tweet -> må ta riktig target num til riktig tekstforklaring på en eller annen måte :OOO
#y = [X: $1.19] [Y: $5.29] [Z 999/1000]
#x = re.findall(r"\$[^ ]+", y)
#ha med klassifiseringsregler her???? % osv?

# må man her på en eller annen måte også si at det er 7 kategorier?
# en tfidf for hver kategori?????? også når man kjører den så putter man inn kategory???
# hvor skal target_num inn?????

# TF-IDF
# min_df can be adjusted, 1 = standard
# stop_words kan fjernes
# ngrams kan gjøres om til kun bare unigrams, som er default
def tfidf(dataframe, training):
    if training:
        print('tfidf started')
        tfidf = TfidfVectorizer(stop_words='english', min_df=0.1, ngram_range=(1,3))
        X = tfidf.fit_transform(dataframe['tweet'])
        print(tfidf.get_feature_names())
        pickle.dump(tfidf, open('datasets/tfidf.txt', 'wb'))
        # tfidfvectorizer object unable to serialize, json file unattainable
        #tfidfJson = json.dumps(tfidf)
        #json.dump(tfidf, open('.//attributes.json', 'w'))
        #saveDataframe(tfidfJson, 'tfidf')
    else:
        tfidf= pickle.load(open('datasets/tfidf.txt', 'wb'))
        #tfidf = loadDataframe('tfidf')
        X = tfidf.transform(dataframe['tweet'])
    return dataframe

# Feature Engineering
def featureEngineering(dataframe, training):
    # Preprocess tweet
    dataframe['tweet'] = dataframe['tweet'].apply(lambda tweet: preProcess(tweet))
    # TFIDF on dataframe
    featureDataframe = tfidf(dataframe, training)
    # Textual on dataframe
    #featureDataframe = categorizationFeatures(featureDataframe)
    return featureDataframe