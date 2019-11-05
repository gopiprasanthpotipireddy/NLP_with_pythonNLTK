from  nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

#Lemmatizing : Identical Stemming instead it replace the word with meaningful Dictionary Word 
print(lemmatizer.lemmatize("running",pos="v")) #run
print(lemmatizer.lemmatize("better",pos="a")) #good
print(lemmatizer.lemmatize("better")) #better
