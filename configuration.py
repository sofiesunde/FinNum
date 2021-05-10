# Sofie Sunde - Spring 2021
# Configuration to simplify change
# Inspiration from Martin Thoma (https://martin-thoma.com/configuration-files-in-python/)

configuration = {
    'filepathTrainingSet': 'datasets/FinNum_training_rebuilded.json',
    'filepathDevelopmentSet': 'datasets/FinNum_dev_rebuilded.json',
    'filepathTestSet': 'datasets/FinNum_dev_rebuilded.json',
    'readDocumentSaveDataframe': True,
    'filename': 'DataframeTrainingSet',
    'loadPreprocessedDataframe': False,
    'processedFilename': 'DataframePreprocessedSet',
    'testSize': 0.33
}