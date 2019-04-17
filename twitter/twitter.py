from nltk.corpus import wordnet
def get_simple_pos(tag):

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
from nltk.corpus import stopwords
import string
stops = set(stopwords.words('english'))
punctuations = list(string.punctuation)
stops.update(punctuations)
from nltk import pos_tag
def clean_review(words):
    output_words = []
    for w in words:
        if w.lower() not in stops:
            pos = pos_tag([w])
            clean_word = lemmatizer.lemmatize(w, pos = get_simple_pos(pos[0][1]))
            output_words.append(clean_word.lower())
    return output_words
    import numpy as np
import pandas as pd
from nltk import word_tokenize
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
print(type(train))
X_test=test.text
X_train=train.text
Y_train=train.airline_sentiment
print(type(X_train))
print(type(Y_train))
print(len(X_train))
print(len(X_test))
document=[]
for text in X_train:
    document.append(word_tokenize(text))
for text in X_test:
    document.append(word_tokenize(text))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
clean_document = [clean_review(doc) for doc in document]
clean_text_documents = [" ".join(doc) for doc in clean_document]
clean_text_documents_train=clean_text_documents[:10980]
clean_text_documents_test=clean_text_documents[10980:]
from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer(stop_words='english',ngram_range=(1, 2),max_df=0.5)# instantiate the vectorizer
vect.fit(clean_text_documents_train)
X_train_dtm=vect.transform(clean_text_documents_train)
X_test_dtm=vect.transform(clean_text_documents_test)
# from sklearn.naive_bayes import MultinomialNB
# nb=MultinomialNB()
# nb.fit(X_train_dtm,Y_train)# train the model using X_train_dtm
# pred=nb.predict(X_test_dtm)#make class predictions for X_test_dtm
# np.savetxt("senti2.csv",pred,fmt="%s")
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train_dtm,Y_train)# train the model using X_train_dtm
pred=log.predict(X_test_dtm)#make class predictions for X_test_dtm
np.savetxt("senti10.csv",pred,fmt="%s")
# from sklearn.ensemble import RandomForestClassifier
# rnd=RandomForestClassifier()
# rnd.fit(X_train_dtm,Y_train)# train the model using X_train_dtm
# pred=rnd.predict(X_test_dtm)#make class predictions for X_test_dtm
# np.savetxt("senti6.csv",pred,fmt="%s")
# from sklearn import svm
# svc=svm.SVC()
# svc.fit(X_train_dtm,Y_train)# train the model using X_train_dtm
# pred=svc.predict(X_test_dtm)#make class predictions for X_test_dtm
# np.savetxt("senti7.csv",pred,fmt="%s")        
