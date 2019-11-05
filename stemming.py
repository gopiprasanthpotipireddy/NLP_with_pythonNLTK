from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

#stemming : stemming is finding the root of the word that has same or similar meaning 
ps=PorterStemmer()

words=['rider','ride','riding']

for w in words:
    print(ps.stem(w))