# Main to read data, preprocess data and run models
# Sofie Sunde - Spring 2021

from preprocessing.read import readDocument, saveDataframe, loadDataframe
from preprocessing.preprocess import featureEngineering
from configuration import configuration as cfg

def main():
    # Read and save or load dataframe
    if cfg['readDocumentSaveDataframe']:
        dataframe = readDocument(cfg['filepathTrainingSet'])
        saveDataframe(dataframe, 'DataframeTrainingSet')
        print("dataframe saved")
    else:
        dataframe = loadDataframe(cfg['filename'])
        print("dataframe loaded")

    # Load or process dataframe
    if cfg['loadPreprocessedDataframe']:
        dataframe = loadDataframe(cfg['processedFilename'])
        print("processed dataframe loaded")
    else:
        dataframe = featureEngineering(dataframe, training=True)
        saveDataframe(dataframe, cfg['processedFilename'])
        print("processed dataframe saved")


    # developmentSet is for validation
    developmentSet = readDocument(cfg['filepathDevelopmentSet'])
    # testSet is for testing
    testSet = readDocument(cfg['filepathTestSet'])
    return dataframe

print(main())
