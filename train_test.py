from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
import nltk
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB

from nltk.classify import ClassifierI
from statistics import mode
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
#documents=  [list(movie_reviews.words(fileid)),category) 
#                for category in movie_reviews.categories() 
#                for fileid in movie_reviews.fileids(category) 
#            ]

documents=[]

pos=open("C:/Users/gopiprasanthp/Desktop/ML/NLP/pos.txt","r").read()
neg=open("C:/Users/gopiprasanthp/Desktop/ML/NLP/neg.txt","r").read()

pos_words=word_tokenize(pos)
neg_words=word_tokenize(neg)

for s in pos.split('.\n'):
    documents.append((s,'pos'))

for s in neg.split('.\n'):
    documents.append((s,'neg'))

random.shuffle(documents)

all_words=[]
for w in pos_words:
    all_words.append(w.lower())
for w in neg_words:
    all_words.append(w.lower())


print(documents[1][0])


all_words=[];

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words=nltk.FreqDist(all_words)
all_words.most_common(15)

all_words['stupid']

word_features=list(all_words.keys())[:3000]

#words to features

def get_Features(document):
    words=set(document)
    features={}
    for w in words:
        features[w]=(w in word_features)
    return features;

featuresets=[(get_Features(rev),category) for (rev,category) in documents ]

#text classification with naive bayes classifier

train_set=featuresets[:1900]
test_set=featuresets[1900:]


class VotinClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers=classifiers
        
    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            votes.append(c.classify(features))
            
        return mode(votes)
    def confidence(self,features):
       votes=[]
       for c in self._classifiers:
          votes.append(c.classify(features))
        
       most_count=votes.count(mode(votes))
       return most_count/len(votes)
        



naive_classifier=nltk.NaiveBayesClassifier.train(train_set)
print("Accuracy of NaiveBayesClassifier  Model : " , nltk.classify.accuracy(naive_classifier,test_set)*100)

MultiNaiveBayes=SklearnClassifier(MultinomialNB())
MultiNaiveBayes.train(train_set)
print("Accuracy of MultinomialNB Model : " , nltk.classify.accuracy(MultiNaiveBayes,test_set)*100)        

#GaussianNaiveBayes=SklearnClassifier(GaussianNB())
#GaussianNaiveBayes.train(train_set)
#print("Accuracy of Model : " , nltk.classify.accuracy(GaussianNaiveBayes,test_set)*100)        
BernoulliNaiveBayes=SklearnClassifier(BernoulliNB())
BernoulliNaiveBayes.train(train_set)
print("Accuracy of BernoulliNB Model : " , nltk.classify.accuracy(BernoulliNaiveBayes,test_set)*100)    

Linearclassifier=SklearnClassifier(LogisticRegression())
Linearclassifier.train(train_set)
print("Accuracy of LogisticRegression Model : " , nltk.classify.accuracy(Linearclassifier,test_set)*100)    

sgdclassifier=SklearnClassifier(SGDClassifier())
sgdclassifier.train(train_set)
print("Accuracy of SGDClassifier Model : " , nltk.classify.accuracy(sgdclassifier,test_set)*100)    

svcclassifier=SklearnClassifier(SVC())
svcclassifier.train(train_set)
print("Accuracy of SVC Model : " , nltk.classify.accuracy(svcclassifier,test_set)*100)    

linearsvc=SklearnClassifier(LinearSVC())
linearsvc.train(train_set)
print("Accuracy of LinearSVC Model : " , nltk.classify.accuracy(linearsvc,test_set)*100)    

Nusvcclassifier=SklearnClassifier(NuSVC(nu=0.5))
Nusvcclassifier.train(train_set)
print("Accuracy of Nu Svc Model : " , nltk.classify.accuracy(Nusvcclassifier,test_set)*100)    

votingclassifier=VotinClassifier(MultiNaiveBayes,
                                                   naive_classifier,
                                                   BernoulliNaiveBayes,
                                                   Linearclassifier,
                                                   svcclassifier,
                                                   linearsvc,
                                                   Nusvcclassifier)

print("Accuracy of Voting Classifier Model : " , nltk.classify.accuracy(votingclassifier,test_set)*100)    

#print(naive_classifier.show_most_informative_features(15))
#
##saving with pickle

save_classifier=open("naivebayes.pickle","wb")
pickle.dump(naive_classifier,save_classifier)
save_classifier.close()
#loading pickle
load_classifier=open("naivebayes.pickle","rb")
naive_classifier=pickle.load(load_classifier)
load_classifier.close()
