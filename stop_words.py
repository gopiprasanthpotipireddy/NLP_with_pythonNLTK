from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#stopword : the word that doesnt have any meaning to it and not useful in the analysis

example_text="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce suscipit neque sit amet egestas maximus. Nam vitae ligula euismod, interdum erat ac, dictum metus. Pellentesque hendrerit consectetur ante, sed lacinia velit ullamcorper ut. Praesent gravida quam sit amet ligula auctor, id vulputate tellus ultrices. Ut a justo ac ipsum posuere varius sed sit amet mauris. Etiam eget erat neque. Pellentesque vitae mollis mi. Praesent scelerisque ipsum mi, vel lobortis lorem ultricies sit amet. Suspendisse a sapien ac lorem venenatis malesuada. Pellentesque a viverra purus. Morbi aliquet tortor semper gravida fermentum."

stop_words=set(stopwords.words("english"))

words=word_tokenize(example_text)

filtered=[w for w in words if not w in stop_words]

print(filtered)
