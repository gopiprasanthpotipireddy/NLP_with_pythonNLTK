import nltk
from nltk.corpus import state_union
from nltk import PunktSentenceTokenizer

#chinking : remove from pos from chunked words

train_text=state_union.raw("2005-GWBush.txt")
sample_text=state_union.raw("2006-GWBush.txt")
custom_tokenizer=PunktSentenceTokenizer(train_text)
sent=custom_tokenizer.tokenize(sample_text)

def process_content():
    for i in sent:
        try:
            
            words=nltk.word_tokenize(i)
            pos_words=nltk.pos_tag(words)
            chunkGram=r"""Chunk: {<.*>+}
                                   }<VB.?|IN|DT>{ """
            chunkParser=nltk.RegexpParser(chunkGram)
            chunked= chunkParser.parse(pos_words)
            chunked.draw()
        except Exception as e:
            print(str(e))

process_content()
