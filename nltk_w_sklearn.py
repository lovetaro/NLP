
import nltk
import random
from nltk.corpus import movie_reviews
import pickle

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:3000]    #get 3000 words form all_words

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features: 
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

from nltk.classify.scikitlearn import SklearnClassifier
#wrapper to include skleanr algorithms within the NLTK classifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
nltk.classify.accuracy(MNB_classifier, testing_set)*100 #convert it to %

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
nltk.classify.accuracy(BNB_classifier, testing_set)

GN_classifier = SklearnClassifier(BernoulliNB())
GN_classifier.train(training_set)
nltk.classify.accuracy(GN_classifier, testing_set)

LogisticReg = SklearnClassifier(LogisticRegression())
LogisticReg.train(training_set)
nltk.classify.accuracy(LogisticReg, testing_set)

#Too arbitrary of accuracy -> in order to improve accuracy and get rid of the noise, => ** Voting System (algo of algos)

from nltk.classify import ClassifierI   #This is a classifier class!
from statistics import mode

class vote_classifier(ClassifierI):   #inherit the nltk.classifierI
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
        
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes)) #how many occurences in each vote
        conf = choice_votes / len(votes)
        return conf

voted_classifier = vote_classifier(classifier, MNB_classifier, LogisticReg, BNB_classifier, GN_classifier)
print(voted_classifier.classify(testing_set[0][0]), voted_classifier.confidence(testing_set[0][0]))
nltk.classify.accuracy(voted_classifier, testing_set)

#Ex- how to save classifier into pickle
NB_classifier = nltk.NaiveBayesClassifier.train(training_set)

"""
print("nltkNBclassifier accuracy score", nltk.classify.accuracy(classifier, testing_set))
classifier.show_most_informative_features(15)
"""

save_classifier = open("naivebayes.pickle","wb")
pickle.dump(NB_classifier, save_classifier)
save_classifier.close()
