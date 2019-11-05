from nltk.corpus import wordnet

#wordnet : Lexicon , Lookup for words ,synonyms,antonyms,and different context examples of words

#synonyms
syns=wordnet.synsets("climb")

print(syns[0].name())

#definition
print(syns[0].definition())

print(syns[0].lemmas()[0].name())

#examples
print(syns[0].examples())
synonyms=[]
antonyms=[]

for word in syns:
    for l in word.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(synonyms)
print(antonyms)


#semantic similarity between words

w1=wordnet.synset("goat.n.01")
w2=wordnet.synset("sheep.n.01")

w1=wordnet.synset("goat.n.01")
w2=wordnet.synset("cow.n.01")

w1=wordnet.synset("gate.n.01")
w2=wordnet.synset("door.n.01")

w1=wordnet.synset("list.v.01")
w2=wordnet.synset("set.v.01")

print(w1.wup_similarity(w2))