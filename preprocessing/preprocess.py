# Sofie Sunde - Spring 2021
# Preprocess data, process features

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

#def textual(dataframe):
# Find all digits in tweet -> må ta riktig target num til riktig tekstforklaring på en eller annen måte :OOO
#y = [X: $1.19] [Y: $5.29] [Z 999/1000]
#x = re.findall(r"\$[^ ]+", y)

# TF-IDF
#def tfidf(dataframe, training):


# Feature Engineering
def featureEngineering(dataframe, training):
    # Preprocess tweet
    dataframe['tweet'] = dataframe['tweet'].apply(lambda tweet: preProcess(tweet))
    # TFIDF on dataframe
    #tfidfDataframe = tfidif(dataframe, training)
    # Textual on dataframe
    #featuresDataframe = textual(tfidfDataframe)
    #return featuresDataframe
    return dataframe