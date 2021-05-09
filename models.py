# Classifier models
# Sofie Sunde - Spring 2021

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

#se om denne skal fjernes herfra kanskje
from sklearn.model_selection import train_test_split


# Support Vector Machine Classifier
class SupportVectorMachineClassifier():
    def __init__(self):
        self.svmClassifier = make_pipeline(StandardScaler(), SVC())

# Random Forest Classifier
class RandomForestClassifier():
    def __init__(self):
        self.rfClassifier = RandomForestClassifier()

# Ensamble Classifier of SVM and Random Forest Classifiers
class EnsembleClassifier():
    def __init__(self):
        svm = make_pipeline(StandardScaler(), SVC())
        rf = RandomForestClassifier()

        self.votingClassifier = VotingClassifier(
            classifiers = [('Support Vector Machine', svm), ('Random Forest', rf)], voting='hard')