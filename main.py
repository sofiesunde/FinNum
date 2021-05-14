# Sofie Sunde - Spring 2021
# Main to read data, preprocess data and run models
# Code structure inspired by Karoline Bonnerud - https://github.com/karolbon/bot-or-human

from configuration import configuration as cfg
from preprocessing.read import readDocument, saveDataframe, loadDataframe, readTestSet
from preprocessing.preprocess import featureEngineering, categoryToNum
from models import SupportVectorMachineClassifier, RandomForestClassifier, EnsembleClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import shuffle
import numpy as np
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
        rfClassifier = RandomForestClassifier()
        rfClassifier.rfClassifier.fit(XTrain, YTrain)
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
        rfClassified = rfClassifier.rfClassifier.predict(XValidate)
        print('done predicting RF: ' + str(time.ctime()))
        # Finding accuracy and loss for current iteration
        rfAccuracy = accuracy_score(YValidate, rfClassified)
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
        testRFClassifier = 
        rfClassified = rfClassifier.rfClassifier.predict(XTest)
        print('done testing RF: ' + str(time.ctime()))
        # Finding accuracy and loss for current iteration
        rfAccuracy = accuracy_score(YTest, rfClassified)
        rfLoss = log_loss(YTest)
        print('accuracy RF: ' + rfAccuracy)
        print('loss RF: ' + rfLoss)

        print("Doing final predictions with Random Forest.", str(time.ctime()))
        final_rf_predictions = final_rf_classifier.classifier.predict(test_X)
        final_rf_accuracy = accuracy_score(test_y, final_rf_predictions)
        final_rf_metrics = precision_recall_fscore_support(
            test_y, final_rf_predictions, average='binary')

        plot_rf = plot_confusion_matrix(final_rf_classifier.classifier,
                                        test_X, test_y, display_labels=class_names,
                                        cmap=plt.cm.Blues, normalize='true')
        plot_rf.ax_.set_title("Random Forest")
        plt.savefig('results/rf.png')

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


