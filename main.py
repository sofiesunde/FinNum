# Sofie Sunde - Spring 2021
# Main to read data, preprocess data and run models
# Code structure inspired by Karoline Bonnerud - https://github.com/karolbon/bot-or-human

from configuration import configuration as cfg
from preprocessing.read import readDocument, saveDataframe, loadDataframe, readTestSet
from preprocessing.preprocess import featureEngineering, categoryCount
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, plot_confusion_matrix, precision_recall_fscore_support
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle
import numpy as np
#from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.utils import class_weight, compute_sample_weight

def main():
    # Code found at https://github.com/scikit-learn-contrib/imbalanced-learn/issues/337
    rfClassifier = RandomForestClassifier(n_estimators=1000)
    svmClassifier = SVC(decision_function_shape='ovr')

    # Classifiers
    #sampler = SMOTE(random_state=2)
    rfPipeline = make_pipeline(StandardScaler(with_std=False, with_mean=False), rfClassifier)
    multiRFClassifier = MultiOutputClassifier(rfPipeline, n_jobs=1)
    svmPipeline = make_pipeline(StandardScaler(with_std=False, with_mean=False), svmClassifier)
    multiSVMClassifier = MultiOutputClassifier(svmPipeline, n_jobs=1)
    # Ensamble not possible with multioutput yet
    #votingClassifier = MultiOutputClassifier(VotingClassifier(
     #   estimators=[('Support Vector Machine', multiSVMClassifier), ('Random Forest', multiRFClassifier)], voting='hard', n_jobs=1))

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

    # Class weights
    class_weight = {'Monetary': 42.19,
                    'Percentage': 13.75,
                    'Option': 1.61,
                    'Indicator': 2.60,
                    'Temporal': 28.96,
                    'Quantity': 9.47,
                    'Product Number': 1.42}

    # Tuning with sample weights
    #sample_weights = compute_sample_weight(class_weight=class_weight, y=YTrain)

    # Training model
    print('start training RF: ' + str(time.ctime()))
    multiRFClassifier.fit(XTrain, YTrain)
    #multiRFClassifier.fit(XTrain, YTrain, **{'RF__sample_weight': sample_weights})
    print('done training RF: ' + str(time.ctime()))

    print('start training SVM: ' + str(time.ctime()))
    multiSVMClassifier.fit(XTrain, YTrain)
    #multiSVMClassifier.fit(XTrain, YTrain, **{'RF__sample_weight': sample_weights})
    print('done training SVM: ' + str(time.ctime()))

    #print('start training ensamble: ' + str(time.ctime()))
    #votingClassifier.fit(XTrain, YTrain)
    #print('done training ensamble: ' + str(time.ctime()))

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
    #print('start predicting RF: ')
    rfValidated = multiRFClassifier.predict(XValidate)
    print('Performance of RF: ')

    # Finding accuracy and loss for current iteration, not supported by SKLearn, ValueError: multiclass - multioutput is not supported
    #rfAccuracy = accuracy_score(YValidate, rfValidated)
    #print('accuracy RF: ' + str(rfAccuracy))
    #rfLoss = log_loss(YValidate, rfValidated)
    #print('loss RF: ' + str(rfLoss))

    # Estimate of accuracy and standard deviation instead
    scores = cross_val_score(multiRFClassifier, XValidate, YValidate, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    #print('start predicting SVM: ')
    svmValidated = multiSVMClassifier.predict(XValidate)
    print('Performance of SVM: ')

    # Finding accuracy and loss for current iteration, not supported by SKLearn, ValueError: multiclass - multioutput is not supported
    #svmAccuracy = accuracy_score(YValidate, svmValidated)
    #print('accuracy SVM: ' + str(svmAccuracy))
    #svmLoss = log_loss(YValidate, svmValidated)
    #print('loss SVM: ' + str(svmLoss))

    # Estimate of accuracy and standard deviation instead
    scores = cross_val_score(multiSVMClassifier, XValidate, YValidate, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    #print('start predicting ensamble: ' + str(time.ctime()))
    #votingValidated = votingClassifier.predict(XValidate)
    #print('done predicting ensamble: ' + str(time.ctime()))

    # Estimate of accuracy and standard deviation instead
    #scores = cross_val_score(votingClassifier, XValidate, YValidate, cv=5)
    #print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    if (cfg['testing']):

        # Testing
        # Read test data from test set
        testDataframe = readDocument(cfg['filepathTestSet'])
        # Feature Engineering on dataframe
        testDataframe = featureEngineering(testDataframe, training=False)
        # Feature Vectors
        testFeatures = testDataframe.drop(columns=['target_num', 'tweet', 'Indicator', 'Monetary', 'Option', 'Percentage', 'Product Number','Quantity', 'Temporal'])
        XTest = testFeatures

        # Category
        YTest = testDataframe[['Indicator', 'Monetary', 'Option', 'Percentage', 'Product Number', 'Quantity', 'Temporal']]

        # Testing performance of model
        finalRFClassified = multiRFClassifier.predict(XTest)
        print('Final performance of RF: ')
        scores = cross_val_score(multiRFClassifier, XTest, YTest, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

        finalRFClassified = multiSVMClassifier.predict(XTest)
        print('Final performance of SVM: ')
        scores = cross_val_score(multiSVMClassifier, XTest, YTest, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

        # sklearn.metrics not compatible with multiclass-multioutput
        #class_names = ['Monetary', 'Percentage', 'Option', 'Indicator', 'Temporal', 'Quantity', 'Product Number']
        #plotRF = plot_confusion_matrix(rfClassifier.classifier, XTest, YTest, display_labels=class_names, cmap=plt.cm.Blues, normalize='true')
        #plotRF.ax_.set_title("Random Forest")
        #plt.savefig('results/RF.png')


if __name__ == '__main__':
    main()
