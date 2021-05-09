# code to process data

from nltk.corpus import stopwords
import re

# Remove stopwords (inspired by NLTK, Chapter 4.1 - Wordlist Corpora)
def removeStopwords(document):
    englishStopwords = stopwords.words('english')
    documentList = document.split(' ')
    wordDocument = [word for word in documentList if word.lower() not in englishStopwords]
    wordDocument = ' '.join(wordDocument)
    return wordDocument

def preProcess(document):
    preProcessed = document.lower()
    preProcessed = preProcessed.replace('\n', ' ')
    #preProcessed = removeStopwords(preProcessed)
    # substitute https into URL, @ into UserReply
    preProcessed = re.sub('https([^\s]+)', "<URL>", preProcessed)
    preProcessed = re.sub('@([^\s]+)', "<UserReply>", preProcessed)
    # substitute $cashtag into CashTag
    preProcessed = re.sub('\$[A-Za-z]([^\s]+)', "<CashTag>", preProcessed)
    return preProcessed

document = "@alexandra Hi my name is Sofie SUnde and I believe this link should be removed, https://www.google.com/?client=safari, although this $ADSD is not the same as this $123"

#print(preProcess(document))

# Evaluate features

# Find all digits in tweet
#y = [X: $1.19] [Y: $5.29] [Z 999/1000]
#x = re.findall(r"\$[^ ]+", y)

# TF-IDF
#def tfIdf(dataframe):


# Feature Engineering
# df = dataframe
def featureEngineering(dataframe):
    # preprocess tweet
    dataframe['tweet'] = dataframe['tweet'].apply(lambda tweet: preProcess(tweet))
    return dataframe
    #df_tfidf = tfidf_features(df, is_training)
    #df_features = textual_features(df_tfidf)
    #return df_features