#import nltk
#nltk.download()

from nltk.tokenize import sent_tokenize,word_tokenize
example_text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce suscipit neque sit amet egestas maximus. Nam vitae ligula euismod, interdum erat ac, dictum metus. Pellentesque hendrerit consectetur ante, sed lacinia velit ullamcorper ut. Praesent gravida quam sit amet ligula auctor, id vulputate tellus ultrices. Ut a justo ac ipsum posuere varius sed sit amet mauris. Etiam eget erat neque. Pellentesque vitae mollis mi. Praesent scelerisque ipsum mi, vel lobortis lorem ultricies sit amet. Suspendisse a sapien ac lorem venenatis malesuada. Pellentesque a viverra purus. Morbi aliquet tortor semper gravida fermentum."

##tokenizing - splitting by words or sentences
##lexicons and corpora
#corpora - body of text ex: political journals,entertainment news , educational speeches etc
#lexicon : words and their meanings
# a word can have different meaning in different text
#In a geographical news context 'hill' means top base 
#In a educational or motivational speech 'hill' means top

print(sent_tokenize(example_text))
print(word_tokenize(example_text))

for i in word_tokenize(example_text):
    print(i)

 