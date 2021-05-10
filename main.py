# Sofie Sunde - Spring 2021
# Main to read data, preprocess data and run models

from configuration import configuration as cfg
from preprocessing.read import readDocument, saveDataframe, loadDataframe, readTestSet
from preprocessing.preprocess import featureEngineering
from models import SupportVectorMachineClassifier, RandomForestClassifier, EnsembleClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

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
        dataframe = featureEngineering(dataframe, training=True)
        saveDataframe(dataframe, cfg['processedFilename'])
        print('processed dataframe saved')

    print(dataframe)
    # du burde fjerne subcategories fra datasettet
    # Defining categories: DETTE FUNGERER IKKE SOM DET SKAL:)
    # må på en eller annen måte også si at dersom det er to target values vil de ha en tilhørende kategori

    #positives = data['label'][data.label == 0]
    #negatives = data['label'][data.label == 1]
    #monetary = dataframe['category'][dataframe.items() == 'Monetary']
    #monetary = dataframe.items('category', 'Monetary')
    #print(dataframe.loc())

    #print(len(monetary))
    #percentage =

    for category in dataframe.columns:
        if category == 'Monetary':
            Y = 1
        elif category == 'Percentage':
            Y = 2
        elif category == 'Option':
            Y = 3
        elif category == 'Indicator':
            Y = 4
        elif category == 'Temporal':
            Y = 5
        elif category == 'Quantity':
            Y = 6
        elif category == 'Product Number':
            Y = 7
        else:
            Y = 0
            print('no matching category')
    print(Y)
    #Y = dataframe.where()
    dataframe.drop(columns=['subcategory'])
    X = dataframe.columns.where(category='tweet')

    XTrain, XTest, YTrain, YTest = train_test_split(
        X, Y, test_size=cfg['testSize'], random_state=42)

    # Training
    print('start training SVM: ' + str(datetime.now()))
    svmClassifier = SupportVectorMachineClassifier()
    svmClassifier.svmClassifier.fit(XTrain, YTrain)

    print('start training RF: ' + str(datetime.now()))
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