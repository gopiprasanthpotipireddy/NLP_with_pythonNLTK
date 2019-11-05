import nltk
from nltk.corpus import state_union
from nltk import PunktSentenceTokenizer

#pos tag: tag each word in a sentence with the pos 

train_text=state_union.raw("2005-GWBush.txt")
sample_text=state_union.raw("2006-GWBush.txt")
custom_tokenizer=PunktSentenceTokenizer(train_text)
sent=custom_tokenizer.tokenize(sample_text)

def pos_tag():
    for i in sent:
        try:
            
            words=nltk.word_tokenize(i)
            pos_words=nltk.pos_tag(words)
            print(pos_words)
            
        except Exception as e:
            print(str(e))
    
    
pos_tag()
