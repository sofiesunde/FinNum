# Sofie Sunde - Spring 2021
# Configuration to simplify change
# Inspiration from Martin Thoma (https://martin-thoma.com/configuration-files-in-python/)

configuration = { # filepath til test sett er foreløpig feil
    'filepathTrainingSet': 'datasets/FinNum_training_rebuilded.json',
    'filepathDevelopmentSet': 'datasets/FinNum_dev_rebuilded.json',
    'filepathTestSet': 'datasets/FinNum_dev_rebuilded.json',
    'readDocumentSaveDataframe': True,
    'filename': 'DataframeTrainingSet',
    'loadPreprocessedDataframe': False,
    'processedFilename': 'DataframePreprocessedSet',
    'testSize': 1,
    'training': True #før du endrer denne til false må du sjekke at du ikke skal fjerne kategori fra datasettet
}
# 'filepathTrainingSet': 'datasets/package.json',
