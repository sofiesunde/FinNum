from preprocessing.read import readDocument, loadTweets
from preprocessing.preprocess import removeStopwords

#  main to run read, preprocess and run models

configuration = {
    'filepathTraining': 'datasets/FinNum_training_rebuilded.json',
    'filepathDevelopment': 'datasets/FinNum_dev_rebuilded.json',
    'filepathTest': 'datasets/FinNum_dev_rebuilded.json'
}

def main():
    trainingSet = readDocument(configuration['filepathTraining'])
    developmentSet = readDocument(configuration['filepathDevelopment'])
    testSet = readDocument(configuration['filepathTest'])
    return developmentSet

print(main())