# Sofie Sunde - Spring 2021
# Main to read data, preprocess data and run models
# Code structure inspired by Karoline Bonnerud - https://github.com/karolbon/bot-or-human

from configuration import configuration as cfg
from preprocessing.read import readDocument, saveDataframe, loadDataframe, readTestSet
from preprocessing.preprocess import featureEngineering, categoryToNum
from models import SupportVectorMachineClassifier, RandomForestClassifier, EnsembleClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import time

def main():
    # Read and save or load dataframe
    if cfg['readDocumentSaveDataframe']:
        dataframe = readDocument(cfg['filepathTrainingSet'])
        saveDataframe(dataframe, 'DataframeTrainingSet')
        print('dataframe saved')
    else:
        dataframe = loadDataframe(cfg['filename'])
        print("dataframe loaded")

    # Load or process dataframe
    if cfg['loadPreprocessedDataframe']:
        dataframe = loadDataframe(cfg['processedFilename'])
        print('processed dataframe loaded')
    else:
        print('featureEnigineering started')
        dataframe = featureEngineering(dataframe, training=True)
        saveDataframe(dataframe, cfg['processedFilename'])
        print('processed dataframe saved')

    print(dataframe)
    print(dataframe.head(10))
    print(dataframe.columns)
    Y = dataframe['category'] = dataframe['category'].apply(categoryToNum)
    print(Y)
    print(dataframe.head(10))
    # X er både target num og tweet !!!!!
    X = dataframe['target_num'] in dataframe['tweet']
    print(X)
    XTrain, XTest, YTrain, YTest = train_test_split(
        X, Y, test_size=cfg['testSize'], random_state=42)

    # Training
    print('start training SVM: ' + str(time.ctime()))
    svmClassifier = SupportVectorMachineClassifier()
    svmClassifier.svmClassifier.fit(XTrain, YTrain)

    print('start training RF: ' + str(time.ctime()))
    rfClassifier = RandomForestClassifier()
    rfClassifier.rfClassifier.fit(XTrain, YTrain)

    print('done training')

    # Classification and accuracy
    svmClassified = svmClassifier.svmClassifier.predict(XTest)
    svmAccuracy = accuracy_score(YTest, svmClassified)
    print('accuracy SVM: ' + svmAccuracy)

    rfClassified = rfClassifier.rfClassifier.predict(XTest)
    rfAccuracy = accuracy_score(YTest, rfClassified)
    print('accuracy RF: ' + rfAccuracy)

    # du bør ha med LOSS også i tillegg til accuracy
    # developmentSet is for validation(?) er ikke validation samme som test:O
    developmentSet = readDocument(cfg['filepathDevelopmentSet'])
    # testSet is for testing
    testSet = readTestSet(cfg['filepathTestSet'])
    return dataframe

print(main())

