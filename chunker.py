import nltk
from nltk.corpus import conll2000

cp = nltk.RegexpParser("")
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
print(cp.evaluate(test_sents))


class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                     for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)
    
    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos,chunktag) in tagged_pos_tags]
        conlltags = [(word,pos,chunktag) for ((word,pos),chunktag)
                    in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])

uni = UnigramChunker(train_sents)
print(uni.evaluate(test_sents))


class NChunkTagger(nltk.TaggerI):
    
    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append( (featurset, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(train_set,
                                                     algorithm='megam', trace=0)    
    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence,i,history)
            tag=self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

