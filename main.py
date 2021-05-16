# Sofie Sunde - Spring 2021
# Main to read data, preprocess data and run models
# Code structure inspired by Karoline Bonnerud - https://github.com/karolbon/bot-or-human

from configuration import configuration as cfg
from preprocessing.read import readDocument, saveDataframe, loadDataframe, readTestSet
from preprocessing.preprocess import featureEngineering, categoryToNum, categoryCount
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import *
#from models import SupportVectorMachineClassifier, RandomForestClassifier, EnsembleClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, plot_confusion_matrix, precision_recall_fscore_support
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd



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
    #if (cfg['training']):
        # Training
        # Read training data from training set
     #   trainingDataframe = readDocument(cfg['filepathTrainingSet'])
      #  print(trainingDataframe.info())
       # print(trainingDataframe.category.value_counts())
        #for label in trainingDataframe[['category']:
         #   if label == 'Monetary':
          #      trainingDataframe.loc[['category'] == 'Monetary', 'category'] = 'Monetary'




    # forest = RandomForestClassifier(n_estimators=n_estimator, random_state=1)
    # sampler = SMOTE(random_state=2)
    # pipeline = make_pipeline(sampler, forest)
    # multi_target_forest = MultiOutputClassifier(pipeline, n_jobs=-1)
    # model = multi_target_forest.fit(vecs, vec_labels)

    # Code found at https://github.com/scikit-learn-contrib/imbalanced-learn/issues/337
    rfClassifier = RandomForestClassifier()
    svmClassifier = SVC()
    # burde nok ha med class_label: weight se fremgangsmåte på https://towardsdatascience.com/why-weight-the-importance-of-training-on-balanced-datasets-f1e54688e7df
    # her kan du evt. prøve SMOTE istedenfor standardscaler fordi det er et ujevnt dataset, men lurer på om smote gjør det samme som reduserte accuracyen i eksempelrapport
    rfPipeline = make_pipeline(StandardScaler(), rfClassifier)
    multiRFClassifier = MultiOutputClassifier(rfPipeline, n_jobs=-1)
    svmPipeline = make_pipeline(StandardScaler(), svmClassifier)
    multiSVMClassifier = MultiOutputClassifier(svmPipeline, n_jobs=-1)

    if (cfg['training']):
        # Training
        # Read training data from training set
        trainingDataframe = readDocument(cfg['filepathTrainingSet'])
        print(trainingDataframe)
        # Exploration of data
        print('Exploration of data: ')
        print(['Monetary', 'Percentage', 'Option', 'Indicator', 'Temporal', 'Quantity', 'Product Number'])
        print(categoryCount(trainingDataframe['category']))

        # Feature Engineering on dataframe
        trainingDataframe = featureEngineering(trainingDataframe, training=True)



        # Feature Vectors
        #Xtrain må legge til features som behandler tall og "området rundt tall"
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(trainingDataframe.tail(5))
        print('Trainingdataframe')
        print(trainingDataframe)
        features = trainingDataframe.drop(columns=['target_num', 'tweet', 'Indicator', 'Monetary', 'Option', 'Percentage', 'Product Number', 'Quantity', 'Temporal'])

        XTrain = features
        print('XTrain: ')
        print(XTrain)

        # Category
        print('YTrain: ')

        YTrain = trainingDataframe[['Indicator', 'Monetary', 'Option', 'Percentage', 'Product Number', 'Quantity', 'Temporal']]
        print(YTrain)

        print('Shape, XTrain, Ytrain: ')
        print(XTrain.shape, YTrain.shape)

        # Training model
        print('start training RF: ' + str(time.ctime()))
        multiRFClassifier.fit(XTrain, YTrain)
        print('done training RF: ' + str(time.ctime()))

        print('start training SVM: ' + str(time.ctime()))
        multiSVMClassifier.fit(XTrain, YTrain)
        print('done training SVM: ' + str(time.ctime()))

        # Validation
        # Read validation data from development test set
        validationDataframe = readDocument(cfg['filepathDevelopmentSet'])
        # Feature Engineering on dataframe
        validationDataframe = featureEngineering(validationDataframe, training=False)
        # Feature Vectors
        validationFeatures = validationDataframe.drop(columns=['target_num', 'tweet', 'Indicator', 'Monetary', 'Option', 'Percentage', 'Product Number', 'Quantity', 'Temporal'])
        XValidate = validationFeatures
        # Category
        YValidate = validationDataframe[
            ['Indicator', 'Monetary', 'Option', 'Percentage', 'Product Number', 'Quantity', 'Temporal']]

        # Validation of model
        print('start predicting RF: ' + str(time.ctime()))
        rfValidated = multiRFClassifier.predict(XValidate)
        print('done predicting RF: ' + str(time.ctime()))

        # Finding accuracy and loss for current iteration, not supported by SKLearn, ValueError: multiclass - multioutput is not supported
        #rfAccuracy = accuracy_score(YValidate, rfValidated)
        #print('accuracy RF: ' + str(rfAccuracy))
        #rfLoss = log_loss(YValidate, rfValidated)
        #print('loss RF: ' + str(rfLoss))

        # Estimate of accuracy and standard deviation instead
        scores = cross_val_score(multiRFClassifier, XValidate, YValidate, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


        print('start predicting SVM: ' + str(time.ctime()))
        svmValidated = multiSVMClassifier.predict(XValidate)
        print('done predicting SVM: ' + str(time.ctime()))

        # Finding accuracy and loss for current iteration, not supported by SKLearn, ValueError: multiclass - multioutput is not supported
        #svmAccuracy = accuracy_score(YValidate, svmValidated)
        #print('accuracy SVM: ' + str(svmAccuracy))
        #svmLoss = log_loss(YValidate, svmValidated)
        #print('loss SVM: ' + str(svmLoss))

        # Estimate of accuracy and standard deviation instead
        scores = cross_val_score(multiRFClassifier, XValidate, YValidate, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


    else:
        # Testing
        # Read test data from test set
        testDataframe = readDocument(cfg['filepathTestSet'])
        # Feature Engineering on dataframe
        testDataframe = featureEngineering(testDataframe, training=False)
        # Feature Vectors
        XTest = testDataframe[['target_num', 'tweet']]
        # Category
        YTest = testDataframe['category']

        # Testing performance of model
        print('start testing RF: ' + str(time.ctime()))
        RFClassified = multiRFClassifier.predict(XTest)
        print('done testing RF: ' + str(time.ctime()))
        # Finding accuracy and loss for current iteration
        rfAccuracy = accuracy_score(YTest, RFClassified)
        RFLoss = log_loss(YTest)
        RFMetrics = precision_recall_fscore_support(
            YTest, RFClassified, average='binary')
        print('accuracy RF: ' + rfAccuracy)
        print('loss RF: ' + RFLoss)
        print('metrics RF: ' + RFMetrics)

        print('start testing SVM: ' + str(time.ctime()))
        svmClassified = multiSVMClassifier.predict(XTest)
        print('done testing SVM: ' + str(time.ctime()))
        # Finding accuracy and loss for current iteration
        svmAccuracy = accuracy_score(YTest, svmClassified)
        svmLoss = log_loss(YTest)
        svmMetrics = precision_recall_fscore_support(
            YTest, svmClassified, average='binary')
        print('accuracy SVM: ' + svmAccuracy)
        print('loss SVM: ' + svmLoss)
        print('metrics SVM: ' + svmMetrics)

        # Må du her gjøre om tilbake tall til kategori????
        class_names = ['Monetary', 'Percentage', 'Option', 'Indicator', 'Temporal', 'Quantity', 'Product Number']
        plotRF = plot_confusion_matrix(rfClassifier.classifier, XTest, YTest, display_labels=class_names, cmap=plt.cm.Blues, normalize='true')
        plotRF.ax_.set_title("Random Forest")
        plt.savefig('results/RF.png')

    ###################################################################################################

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
