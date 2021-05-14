# Sofie Sunde - Spring 2021
# Main to read data, preprocess data and run models
# Code structure inspired by Karoline Bonnerud - https://github.com/karolbon/bot-or-human

from configuration import configuration as cfg
from preprocessing.read import readDocument, saveDataframe, loadDataframe, readTestSet
from preprocessing.preprocess import featureEngineering, categoryToNum
from models import SupportVectorMachineClassifier, RandomForestClassifier, EnsembleClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, plot_confusion_matrix, precision_recall_fscore_support
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plot
import time



def main():
    # Read and save or load dataframe
    #if cfg['readDocumentSaveDataframe']:
     #   dataframe = readDocument(cfg['filepathTrainingSet'])
      #  saveDataframe(dataframe, 'DataframeTrainingSet')
       # print('dataframe saved')
    #else:
     #   dataframe = loadDataframe(cfg['filename'])
      #  print("dataframe loaded")

    # Load or process dataframe
    #if cfg['loadPreprocessedDataframe']:
     #   dataframe = loadDataframe(cfg['processedFilename'])
      #  print('processed dataframe loaded')
    #else:
     #   print('featureEngineering started')
      #  dataframe = featureEngineering(dataframe, training=True, validation=False)
        #saveDataframe(dataframe, cfg['processedFilename'])
       # print('processed dataframe saved')

    # må på en eller annen måte presisere target num i tweeten som input tror jeg
    #print('her kommer dataframe')
    #print(dataframe)
    # skal man si noe om at første element i listen target_num = første element i category???
    # Her begynner koden din nå
    ###################################################################################################
    rfClassifier = RandomForestClassifier()
    multiRFClassifier = MultiOutputClassifier(rfClassifier, n_jobs=7)
    if (cfg['training']):
        # Training
        # Read training data from training set
        trainingDataframe = readDocument(cfg['filepathTrainingSet'])
        # Feature Engineering on dataframe
        trainingDataframe = featureEngineering(trainingDataframe, training=True, validation=False)
        # Feature Vectors
        XTrain = trainingDataframe[['target_num', 'tweet']]
        # Category
        YTrain = trainingDataframe['category']

        # Training model
        print('start training RF: ' + str(time.ctime()))
        multiRFClassifier.rfClassifier.fit(XTrain, YTrain)
        print('done training RF: ' + str(time.ctime()))

        # Validation
        # Read validation data from development test set
        validationDataframe = readDocument(cfg['filepathDevelopmentSet'])
        # Feature Engineering on dataframe
        validationDataframe = featureEngineering(validationDataframe, training=True, validation=True)
        # Feature Vectors
        XValidate = validationDataframe[['target_num', 'tweet']]
        # Category
        YValidate = validationDataframe['category']

        # Validation of model
        print('start predicting RF: ' + str(time.ctime()))
        rfValidated = multiRFClassifier.rfClassifier.predict(XValidate)
        print('done predicting RF: ' + str(time.ctime()))
        # Finding accuracy and loss for current iteration
        rfAccuracy = accuracy_score(YValidate, rfValidated)
        rfLoss = log_loss(YValidate)
        print('accuracy RF: ' + rfAccuracy)
        print('loss RF: ' + rfLoss)

    else:
        # Testing
        # Read test data from test set
        testDataframe = readDocument(cfg['filepathTestSet'])
        # Feature Engineering on dataframe
        testDataframe = featureEngineering(testDataframe, training=False, validation=False)
        # Feature Vectors
        XTest = testDataframe[['target_num', 'tweet']]
        # Category
        YTest = testDataframe['category']

        # Testing performance of model
        print('start testing RF: ' + str(time.ctime()))
        RFClassified = multiRFClassifier.rfClassifier.predict(XTest)
        print('done testing RF: ' + str(time.ctime()))
        # Finding accuracy and loss for current iteration
        RFAccuracy = accuracy_score(YTest, RFClassified)
        RFLoss = log_loss(YTest)
        RFMetrics = precision_recall_fscore_support(
            YTest, RFClassified, average='binary')
        print('accuracy RF: ' + RFAccuracy)
        print('loss RF: ' + RFLoss)
        print('metrics RF: ' + RFMetrics)
        # Må du her gjøre om tilbake tall til kategori????
        class_names = ['Monetary', 'Percentage', 'Option', 'Indicator', 'Temporal', 'Quantity', 'Product Number']
        plotRF = plot_confusion_matrix(rfClassifier.classifier,
                                        XTest, YTest, display_labels=class_names,
                                        cmap=plot.cm.Blues, normalize='true')
        plotRF.ax_.set_title("Random Forest")
        plot.savefig('results/RF.png')



    #XTrain, XValidate, YTrain, YValidate = train_test_split(
    #    X, Y, test_size=cfg['testSize'], random_state=42)

    # Training
    #print('start training SVM: ' + str(time.ctime()))
    #svmClassifier = SupportVectorMachineClassifier()
    #svmClassifier.svmClassifier.fit(XTrain, YTrain)

    #enumerate as boolean??????

    #print('start training RF: ' + str(time.ctime()))
    #rfClassifier = RandomForestClassifier()
    #rfClassifier.rfClassifier.fit(XTrain, YTrain)

    #print('done training')

    # HER SKAL DEV SETTET BRUKES


    # Classification and accuracy with validation
    #print('start predicting SVM: ' + str(time.ctime()))
    #svmClassified = svmClassifier.svmClassifier.predict(XValidate)
    #svmAccuracy = accuracy_score(YValidate, svmClassified)
    #svmLoss = log_loss(YValidate)
    #print('accuracy SVM: ' + svmAccuracy)
    #print('loss SVM: ' + svmLoss)

    #print('start predicting RF: ' + str(time.ctime()))
    #rfClassified = rfClassifier.rfClassifier.predict(XValidate)
    #rfAccuracy = accuracy_score(YValidate, rfClassified)
    #rfLoss = log_loss(YValidate)
    #print('accuracy RF: ' + rfAccuracy)
    #print('loss RF: ' + rfLoss)

    # du bør ha med LOSS også i tillegg til accuracy, er dette riktig loss????
    # developmentSet is for validation
    #developmentSet = readDocument(cfg['filepathDevelopmentSet'])
    # testSet is for testing
    #testSet = readTestSet(cfg['filepathTestSet'])


if __name__ == '__main__':
    main()
