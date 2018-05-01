
#1. Tagging

import nltk
from nltk import word_tokenize
text = word_tokenize("And now something completely different")
nltk.pos_tag(text)

text = nltk.Text(word.lower() for word in nltk.corpus.brown.words())
print(text.similar('woman'), text.similar('bought'))

#2. Tagged Corpora
tagged_token = nltk.tag.str2tuple('fly/NN')

sent = '''The/AT grand/JJ jury/NN completed/VBD on/IN a/AT number/NN
of/IN other/AP topic/NNS ./. among/IN them/PPO the/AT Atlanta/NP 
and/CC'''
w_list = [nltk.tag.str2tuple(w) for w in sent.split()]


wsj = nltk.corpus.treebank.tagged_words(tagset='universal')
word_tag_fd = nltk.FreqDist(wsj)
word_tag_fd.most_common(10)

word_list = [wt[0] for (wt,_) in word_tag_fd.most_common() if wt[1]=='VERB']

cfd1 = nltk.ConditionalFreqDist(wsj)
cfd1['yield'].most_common()
cfd1['cut'].most_common()

cfd2 = nltk.ConditionalFreqDist((tag, word) for (word, tag) in wsj)
verb_list = list(cfd2['VBN'])


#3. N-Gram Taggers : Train and Test - simplest model 

from nltk.corpus import brown
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

uni_tagger = nltk.UnigramTagger(brown_tagged_sents)
uni_tagger.tag(brown_sents[2007]) 
uni_tagger.evaluate(brown_tagged_sents)
# Above is essentially training and testing on the SAME DATASET ==> easier! not facing any overfitting problem


size = int(len(brown_tagged_sents)*0.8)
print(size)

train_sents = brown_tagged_sents[:size]  #80%는 training set
test_sents = brown_tagged_sents[size:]   #나머지 20%는 testing set

uni_tagger = nltk.UnigramTagger(train_sents)
uni_tagger.evaluate(test_sents)

#3-2. Bigram Tagger
bi_tagger = nltk.BigramTagger(train_sents)
bi_tagger.tag(brown_sents[2008])


#4. Whether male or female
from nltk.corpus import names
import random

def gender_feature(name):
    return {'last_letter':name[-1]}

names.fileids()
labeled_names = ([(name, 'male') for name in names.words('male.txt')] + 
                  [(name,'female') for name in names.words('female.txt')])   #These TWO LISTS are combined into one
                  
random.shuffle(labeled_names)
featuresets = [ (gender_feature(n), gender) for (n,gender) in labeled_names]
train_set, test_set = featuresets[500:], featuresets[:500]

classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
classifier.show_most_informative_features()
