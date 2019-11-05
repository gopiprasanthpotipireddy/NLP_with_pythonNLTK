from nltk.corpus import gutenberg,indian
from nltk.tokenize import sent_tokenize
bible=gutenberg.raw("bible-kjv.txt")
telugu=indian.raw("telugu.pos")

#Corpora : Differemnt Text Datasets
#import nltk 
#print(nltk.__file__)

sent=sent_tokenize(bible)
print(sent[0:10])
