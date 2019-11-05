from nltk.corpus import movie_reviews
import nltk
import random

#documents=  [list(movie_reviews.words(fileid)),category) 
#                for category in movie_reviews.categories() 
#                for fileid in movie_reviews.fileids(category) 
#            ]

documents=[]
for category in movie_reviews.categories():
    
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)),category))
        
random.shuffle(documents)

print(documents[1])


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

naive_classifier=nltk.NaiveBayesClassifier.train(train_set)

print("Accuracy of Model : " , nltk.classify.accuracy(naive_classifier,test_set)*100)        

print(naive_classifier.show_most_informative_features(15))
