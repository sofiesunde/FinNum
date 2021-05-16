# Sofie Sunde - Spring 2021
# Preprocess data, process features
# Code structure inspired by Karoline Bonnerud - https://github.com/karolbon/bot-or-human

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from preprocessing.read import saveDataframe, loadDataframe
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag, word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer


# Remove stopwords (inspired by NLTK, Chapter 4.1 - Wordlist Corpora)
def removeStopwords(document):
    englishStopwords = stopwords.words('english')
    documentList = document.split(' ')
    wordDocument = [word for word in documentList if word.lower() not in englishStopwords]
    wordDocument = ' '.join(wordDocument)
    return wordDocument


# Normalization
def preProcess(document):
    preProcessed = document.lower()
    preProcessed = preProcessed.replace('\n', ' ')
    preProcessed = removeStopwords(preProcessed)
    # substitute https into URL, @ into UserReply, and removes them
    # preProcessed = re.sub('https([^\s]+)', '<URL>', preProcessed)
    # preProcessed = re.sub('<URL>', '', preProcessed)
    preProcessed = re.sub('https([^\s]+)', '', preProcessed)
    # preProcessed = re.sub('@([^\s]+)', '<UserReply>', preProcessed)
    # preProcessed = re.sub('<UserReply>', '', preProcessed)
    preProcessed = re.sub('@([^\s]+)', '', preProcessed)
    # substitute $cashtag into CashTag, and removes it
    # preProcessed = re.sub('\$[A-Za-z]([^\s]+)', '<CashTag>', preProcessed)
    # preProcessed = re.sub('<CashTag>', '', preProcessed)
    preProcessed = re.sub('\$[A-Za-z]([^\s]+)', '', preProcessed)
    return preProcessed

# Evaluate features
def categoryCount(category):
    monetary = 0
    percentage = 0
    option = 0
    indicator = 0
    temporal = 0
    quantity = 0
    productNum = 0
    for i, label in enumerate(category):
        if 'Monetary' in label:
            monetary += 1

        elif 'Percentage' in label:
            percentage += 1

        elif 'Option' in label:
            option += 1

        elif 'Indicator' in label:
            indicator += 1

        elif 'Temporal' in label:
            temporal += 1

        elif 'Quantity' in label:
            quantity += 1

        elif 'Product Number' in label:
            productNum += 1

    categories = [monetary, percentage, option, indicator, temporal, quantity, productNum]
    return categories

# TF-IDF
def tfidf(dataframe, training):

    if training:
        print(dataframe)
        print('tfidf started')
        tfidfVector = TfidfVectorizer(stop_words='english', min_df=0.0129, max_df=0.75, ngram_range=(1, 3))
        #dataframe['tweet'] = tfidfVector.fit_transform(dataframe['tweet']).toarray()
        features = tfidfVector.fit_transform(dataframe['tweet']).toarray()
        print(features)
        print(features.shape)
        pickle.dump(tfidfVector, open('tfidf.pkl', 'wb'))
    else:
        print('loading tfidf started')
        tfidfVector = pickle.load(open('tfidf.pkl', 'rb'))
        print('corpus: ', tfidfVector, type(tfidfVector))

        features = tfidfVector.transform(dataframe['tweet']).toarray()

    for i, col in enumerate(tfidfVector.get_feature_names()):
        dataframe[col] = pd.Series(features[:, i])

    dataframe = dataframe.fillna(0)
    return dataframe


# Feature Engineering
def featureEngineering(dataframe, training):
    # Split list of target nums into several columns
    #dataframe = multiProcessing(dataframe)
    print(dataframe)
    # Preprocess tweet
    dataframe['tweet'] = dataframe['tweet'].apply(lambda tweet: preProcess(tweet))

    # Methods tested, not used
    #dataframe['target_num'] = dataframe['target:num'].apply()
    # dataframe['target_num'] = dataframe['tweet'].apply(lambda num: multipleTargetNums(num))
    # dataframe = dataframe.drop(dataframe[len(dataframe['target_num'] > 1)].index, inplace=True)
    # dataframe = multipleProcessing1(dataframe)
    # dataframe['target_num'] = dataframe['target_num'].apply(lambda num: multipleProcessing(dataframe, num))
    # dataframe['target_num'] = dataframe['target_num'].apply(lambda dataframe: multipleProcessing(dataframe, num))
    # dataframe[['target_num', 'category']] = dataframe[['target_num', 'category']].apply(lambda num: len(num.split(',')) < 2)
    # print('dataframe with only one num in tweet')
    # print(dataframe)
    # print(df[df['alfa'].apply(lambda x: len(x.split(',')) < 3)])
    # categoryDataframe = dataframe['category'].apply(lambda category: oneHotEncoder(category))
    # print(categoryDataframe)
    # dataframe['category'] = dataframe['category'].apply(lambda category: categoryToNum(category))

    # Categorization on dataframe
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(dataframe.head(5))
    # One Hot Encoding of categories
    one_hot = pd.get_dummies(dataframe['category'].apply(pd.Series).stack()).sum(level=0)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(one_hot.head(5))
    #one_hot = pd.get_dummies(dataframe['category'])

    # Drop category column as it is now encoded
    dataframe = dataframe.drop('category', axis=1)

    # Join the encoded dataframe
    dataframe = dataframe.join(one_hot, how='outer')

    # POS-Tagging, 'ruined' the formatting of the dataframe
    #dataframe['tweet_pos'] = dataframe.apply(lambda x: pos_tags(x['tweet']), axis=1)

    # TFIDF on dataframe
    featureDataframe = tfidf(dataframe, training)

    return featureDataframe

########################################################################################################################

# Methods tested but not used:

# POS tag 2-3 chars abbrivation mapping to 1 char abbrevations
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
#pos_code_map = {'CC': 'A', 'CD': 'B', 'DT': 'C', 'EX': 'D', 'FW': 'E', 'IN': 'F', 'JJ': 'G', 'JJR': 'H', 'JJS': 'I',
#                'LS': 'J', 'MD': 'K', 'NN': 'L', 'NNS': 'M',
#                'NNP': 'N', 'NNPS': 'O', 'PDT': 'P', 'POS': 'Q', 'PRP': 'R', 'PRP$': 'S', 'RB': 'T', 'RBR': 'U',
#                'RBS': 'V', 'RP': 'W', 'SYM': 'X', 'TO': 'Y', 'UH': 'Z',
#                'VB': '1', 'VBD': '2', 'VBG': '3', 'VBN': '4', 'VBP': '5', 'VBZ': '6', 'WDT': '7', 'WP': '8',
#                'WP$': '9', 'WRB': '@'}
# Python 2 code_pos_map={v: k for k, v in pos_code_map.iteritems()}
#code_pos_map = {v: k for k, v in pos_code_map.items()}

# abbrivation converters
#def convert(tag):
#    try:
#        code = pos_code_map[tag]
#    except:
#        code = '?'
#    return code

#def inv_convert(code):
#    try:
#        tag = code_pos_map[code]
#    except:
#        tag = '?'
#    return tag

# POS tag converting

#def pos_tags(text):
#    tokenizer = RegexpTokenizer(r'\w+')
#    text_processed = tokenizer.tokenize(text)
#    return "".join(convert(tag) for (word, tag) in nltk.pos_tag(text_processed))

#def text_pos_inv_convert(text):
#    return "-".join(inv_convert(c.upper()) for c in text)

#def stem(document):
#    stemmer = SnowballStemmer('english')
#    document['tweets'] = document['tweets'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split('')]))
#    return document

#def multipleTargetNums(dataframe):
#   if len(target_num) > 1:
#      for label in target_num:
#     dataframe[['targetnum1', 'targetnum2']] = dataframe['target_num'](dataframe.teams.tolist(), index=dataframe.index)


#def categoryToNum(category):
    # One hot encode
#    n = len(category)
#    categories = np.zeros((n, 7), dtype=int)
#    for i, label in enumerate(category):
#        if label == 'Monetary':
#            categories[i, 0] = 1

#        elif label == 'Percentage':
#            categories[i, 1] = 1

#        elif label == 'Option':
#            categories[i, 2] = 1

#        elif label == 'Indicator':
#            categories[i, 3] = 1

#        elif label == 'Temporal':
#            categories[i, 4] = 1

#        elif label == 'Quantity':
#            categories[i, 5] = 1

#        elif label == 'Product Number':
#            categories[i, 6] = 1

#    return categories

#def monetary(category):
    # One hot encode
#    for label in category:
#        if label == 'Monetary':
#            categories = 1
#        else:
#            categories = 0

#    return categories

#def oneHotEncoder(category):
    # One hot encode
#    categoriesDataframe = pd.DataFrame(
#        columns=['Monetary', 'Percentage', 'Option', 'Indicator', 'Temporal', 'Quantity', 'Product Number'])

#    n = len(category)
#    categories = np.zeros((n, 7), dtype=int)

#    for label in category:
#        print(label)
        # categoriesDataframe.loc[i] = [1 if label == 'Monetary' else 0, 1 if label == 'Percentage' else 0, 1 if label == 'Option' else 0, 1 if label == 'Indicator' else 0, 1 if label == 'Temporal' else 0, 1 if label == 'Quantity' else 0, 1 if label == 'Product Number' else 0]

#        if not label == 'Monetary':
#            categoriesDataframe['Monetary'].append(0)
#        else:
#            categoriesDataframe['Monetary'].append(1)

#        if not label == 'Percentage':
#            categoriesDataframe['Percentage'].append(0)
#        else:
#            categoriesDataframe['Percentage'].append(1)

#        if not label == 'Option':
#            categoriesDataframe['Option'].append(0)
#        else:
#            categoriesDataframe['Option'].append(1)

#        if not label == 'Indicator':
#            categoriesDataframe['Indicator'].append(0)
#        else:
#            categoriesDataframe['Indicator'].append(1)

#        if not label == 'Temporal':
#            categoriesDataframe['Temporal'].append(0)
#        else:
#            categoriesDataframe['Temporal'].append(1)

#        if not label == 'Quantity':
#            categoriesDataframe['Quantity'].append(0)
#        else:
#            categoriesDataframe['Quantity'].append(1)

#        if not label == 'Product Number':
#            categoriesDataframe['Product Number'].append(0)
#        else:
#            categoriesDataframe['Product Number'].append(1)

        # categoriesDataframe['Monetary'].append(1 if label == 'Monetary' else 0)
        # categoriesDataframe['Percentage'].append(1 if label == 'Percentage' else 0)
        # categoriesDataframe['Option'].append(1 if label == 'Option' else 0)
        # categoriesDataframe['Indicator'].append(1 if label == 'Indicator' else 0)
        # categoriesDataframe['Temporal'].append(1 if label == 'Temporal' else 0)
        # categoriesDataframe['Quantity'].append(1 if label == 'Quantity' else 0)
        # categoriesDataframe['Product Number'].append(1 if label == 'Product Number' else 0)

#    print(categoriesDataframe)
#    return categoriesDataframe

#def numToCategory(category):
#    categories = []
    # map(float, categories[i].split(","))
#    for label in category:
#        if label == 'Monetary':
#            categories.append(1)

#        elif label == 'Percentage':
#            categories.append(2)

#        elif label == 'Option':
#            categories.append(3)

#        elif label == 'Indicator':
#            categories.append(4)

#        elif label == 'Temporal':
#            categories.append(5)

#        elif label == 'Quantity':
#            categories.append(6)

#        elif label == 'Product Number':
#            categories.append(7)

#    print(categories)
#    return categories

#def findTarget(dataframe):
#    counter = 0
#    for num in dataframe['target_num']:
#        counter += 1
#        dataframe.append(['target num' + str(counter)])


#def multipleProcessing(dataframe, multiple):
#    n = len(multiple)
#    print(dataframe['index'])
#    for label in enumerate(multiple):
#        if n > 1:
#            dataframe.drop(dataframe['index'])


# finner på index :
# data[data['year']==2016]
# Get names of indexes for which column Age has value 30
# indexNames = dfObj[ dfObj['Age'] == 30 ].index
# Delete these row indexes from dataFrame
# dfObj.drop(indexNames , inplace=True)

#def multicategories(category):
#    print(category)

#    categoryDataframe = pd.DataFrame(columns=['category'])
#    multiLabelBinarizer = MultiLabelBinarizer(classes=['Monetary'])
#    print(multiLabelBinarizer)
#    transform = [lambda label: multiLabelBinarizer.fit_transform(label)]
#    category = [transform for label in category]


#    for label in category:

#        multiLabelBinarizer.fit_transform(label)
#        print(label)

#    categoryDataframe['category'].append(category)
#    print(category)
#    return category

#def multiProcessing(dataframe):

#    nums = dataframe['target_num']
#    for num in nums:
#        while len(num) < 7:
#            num.append(0)

#    dataframe[['target_num1', 'target_num2', 'target_num3', 'target_num4', 'target_num5', 'target_num6', 'target_num7']] = pd.DataFrame(dataframe.target_num.tolist(), index=dataframe.index)
#    dataframe = dataframe.fillna(0)
#    return dataframe

#def multipleProcessing1(dataframe):
#    print(dataframe)
#    nums = dataframe['target_num']
#    counter = 0
#    for i, num in nums:
#        counter += 1
#        if len(num) > 1:
#            multiple = dataframe[num].index
#            dataframe.drop(dataframe[['index']] == i)

#        multiple = dataframe[len(num) > 1]
#        dataframe.drop(multiple, inplace=True)
#    for label in dataframe['target_num']:
#       if len(label) > 1:
#          dataframe.drop(dataframe['index'])
#    return dataframe