import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re
import warnings
# set plot style
sns.set()

import string
# Model evaluation
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords, wordnet  
from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')
df = df_train.copy()
test_data = df_test.copy()

df_all = df

def remove_stopword(name):
    new_name = []
    for i in name.split():
        if i not in stopwords.words('english'):
            new_name.append(i)
        else:
            pass
    name = (' '.join(new_name))
    return name
    
def solve_dataframe(dataframe_name):  
    df_all = dataframe_name
    df_all.message = df_all.message.str.lower()
    print('removing links')
    pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    subs_url = r'url-web'
    df_all['message'] = df_all['message'].replace(to_replace = pattern_url, value = subs_url, regex = True)
    
    
    import string
    def remove_punctuation(post):
        return ''.join([l for l in post if l not in string.punctuation])
    print('removing punctuations')

    df_all['message'] = df_all['message'].apply(remove_punctuation)
    
#     print('removing stopwords')
#     df_all['message'] = df_all['message'].apply(lambda x: remove_stopword(x))

    print('applying lematization')

    df_all['message'] = df_all['message'].apply(lambda x:x.split())
    def mbti_lemma(words, lemmatizer):
        return [lemmatizer.lemmatize(word) for word in words] 
    df_all['message'] = df_all['message'].apply(mbti_lemma, args=(lemmatizer, ))


    
    df_all['message'] = df_all['message'].apply(lambda x: ' '.join(x))
    return df_all
    
    
def process_single_text(name):
    print('changing everything into lowercase')
    name = name.lower()
    print('removing links')
    for i in name.split():
        if i[:4] == 'http':
            name = name.replace(i,'url-web')
    
    print('removing punctuations')
    name = ''.join([l for l in name if l not in string.punctuation])
    #name = remove_stopword(name)
    name = name.split()

    print('applying lematization')
    lema_name = []
    for i in name:
        lema_name.append(lemmatizer.lemmatize(i))

    lema_name = ' '.join(lema_name)
    lema_name = np.array(lema_name).reshape(-1)
    return lema_name
df_all = solve_dataframe(df_all)

X_train, X_test, y_train, y_test = train_test_split(df_all.message, df_all.sentiment, test_size=0.2, random_state=27)

# Random Forest Classifier
rf = Pipeline([('tfidf', TfidfVectorizer()),
               ('clf', RandomForestClassifier(max_depth=5, 
                                              n_estimators=100))])

# Naïve Bayes:
nb = Pipeline([('tfidf', TfidfVectorizer()),
               ('clf', MultinomialNB())])

# K-NN Classifier
knn = Pipeline([('tfidf', TfidfVectorizer()),
                ('clf', KNeighborsClassifier(n_neighbors=5, 
                                             metric='minkowski', 
                                             p=2))])

# Logistic Regression
lr = Pipeline([('tfidf',TfidfVectorizer(max_df=0.9,min_df=2, ngram_range=(1,3))),
               ('clf',LogisticRegression(C=50, max_iter=1000,solver = 'sag', multi_class = 'ovr', warm_start = False))])


# Linear SVC:
lsvc = Pipeline([('tfidf', TfidfVectorizer(max_df=0.8,min_df=2, ngram_range=(1,3))),
                 ('clf', LinearSVC(C = 0.5))])
                 
# Random forest 
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Niave bayes
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# K - nearest neighbors
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Linear regression
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Linear SVC
lsvc.fit(X_train, y_train)
y_pred_lsvc = lsvc.predict(X_test)

import pickle
model_save_path = "./models/rf.pkl"
with open(model_save_path,'wb') as file:
    pickle.dump(rf,file)


model_save_path = "./models/nb.pkl"
with open(model_save_path,'wb') as file:
    pickle.dump(nb,file)


model_save_path = "./models/knn.pkl"
with open(model_save_path,'wb') as file:
    pickle.dump(knn,file)


model_save_path = "./models/lr.pkl"
with open(model_save_path,'wb') as file:
    pickle.dump(lr,file)    

model_save_path = "./models/lsvc.pkl"
with open(model_save_path,'wb') as file:
    pickle.dump(lsvc,file)
    




