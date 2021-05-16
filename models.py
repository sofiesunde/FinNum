# Sofie Sunde - Spring 2021
# Classifier models - this is not used per now, classifiers are built in main
# Code structure inspired by Karoline Bonnerud - https://github.com/karolbon/bot-or-human

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

# Support Vector Machine Classifier
class SupportVectorMachineClassifier():
    def __init__(self):
        self.svmClassifier = make_pipeline(StandardScaler(), OneVsRestClassifier(SVC(), n_jobs=1))

# Random Forest Classifier
class RandomForestClassifier():
    def __init__(self):
        self.rfClassifier = RandomForestClassifier()

# Ensamble Classifier of SVM and Random Forest Classifiers
class EnsembleClassifier():
    def __init__(self):
        svm = make_pipeline(StandardScaler(), OneVsRestClassifier(SVC(), n_jobs=1))
        rf = RandomForestClassifier()

        self.votingClassifier = VotingClassifier(
            classifiers=[('Support Vector Machine', svm), ('Random Forest', rf)], voting='hard', n_jobs=1)

        # Code found at https://github.com/scikit-learn-contrib/imbalanced-learn/issues/337
        #forest = RandomForestClassifier(n_estimators=n_estimator, random_state=1)
        #sampler = SMOTE(random_state=2)
        #pipeline = make_pipeline(sampler, forest)
        #multi_target_forest = MultiOutputClassifier(pipeline, n_jobs=-1)
        #model = multi_target_forest.fit(vecs, vec_labels)