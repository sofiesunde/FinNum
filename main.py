# Sofie Sunde - Spring 2021
# Main to read data, preprocess data and run models
# Code structure inspired by Karoline Bonnerud - https://github.com/karolbon/bot-or-human

from configuration import configuration as cfg
from preprocessing.read import readDocument, saveDataframe, loadDataframe, readTestSet
from preprocessing.preprocess import featureEngineering, categoryToNum
from models import SupportVectorMachineClassifier, RandomForestClassifier, EnsembleClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
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
        print('featureEngineering started')
        dataframe = featureEngineering(dataframe, training=True)
        saveDataframe(dataframe, cfg['processedFilename'])
        print('processed dataframe saved')

    # må på en eller annen måte presisere target num i tweeten som input tror jeg
    X = dataframe['target_num'] #= dataframe.apply(multiple)
    Y = dataframe['category'] = dataframe['category'].apply(categoryToNum)

    # X er både target num og tweet !!!!!
    # aner ikke om dette blir riktig, skal category være med?
    # skal man si noe om at første element i listen target_num = første element i category???
    dataframe.drop(columns=['idx', 'id', 'category'])


    print(X)
    #X = dataframe['target_num'] in dataframe['tweet']
    XTrain, XValidate, YTrain, YValidate = train_test_split(
        X, Y, test_size=cfg['testSize'], random_state=42)

    # Training
    print('start training SVM: ' + str(time.ctime()))
    svmClassifier = SupportVectorMachineClassifier()
    svmClassifier.svmClassifier.fit(list(XTrain), list(YTrain))

    #enumerate as boolean??????

    #print('start training RF: ' + str(time.ctime()))
    #rfClassifier = RandomForestClassifier()
    #rfClassifier.rfClassifier.fit(list(XTrain), YTrain)

    print('done training')

    # HER SKAL DEV SETTET BRUKES


    # Classification and accuracy with validation
    print('start predicting SVM: ' + str(time.ctime()))
    svmClassified = svmClassifier.svmClassifier.predict(XValidate)
    svmAccuracy = accuracy_score(YValidate, svmClassified)
    svmLoss = log_loss(YValidate)
    print('accuracy SVM: ' + svmAccuracy)
    print('loss SVM: ' + svmLoss)

    #print('start predicting RF: ' + str(time.ctime()))
    #rfClassified = rfClassifier.rfClassifier.predict(XValidate)
    #rfAccuracy = accuracy_score(YValidate, rfClassified)
    #rfLoss = log_loss(YValidate)
    #print('accuracy RF: ' + rfAccuracy)
    #print('loss RF: ' + rfLoss)

    # du bør ha med LOSS også i tillegg til accuracy, er dette riktig loss????
    # developmentSet is for validation
    developmentSet = readDocument(cfg['filepathDevelopmentSet'])
    # testSet is for testing
    testSet = readTestSet(cfg['filepathTestSet'])
    return dataframe

print(main())

