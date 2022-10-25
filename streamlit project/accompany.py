import streamlit as st
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import numpy as np
import pandas as pd
import re
import warnings
import pickle
import string
# Model evaluation
from sklearn import metrics

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.corpus import stopwords, wordnet  


home = st.sidebar.checkbox('Home')
make_prediction = st.sidebar.checkbox('Make prediction', value = True)
eda = st.sidebar.checkbox('EDA')
model_performance = st.sidebar.checkbox('model performance')
report_bug = st.sidebar.checkbox('Report bug')
contact_us = st.sidebar.checkbox('Contact us')

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


model_load_path = "./models/rf.pkl"
with open(model_load_path,'rb') as file:
    rf = pickle.load(file)
    
model_load_path = "./models/nb.pkl"
with open(model_load_path,'rb') as file:
    nb = pickle.load(file)
    
model_load_path = "./models/knn.pkl"
with open(model_load_path,'rb') as file:
    knn = pickle.load(file)
    
model_load_path = "./models/lr.pkl"
with open(model_load_path,'rb') as file:
    lr = pickle.load(file)
    
model_load_path = "./models/lsvc.pkl"
with open(model_load_path,'rb') as file:
    lsvc = pickle.load(file)





st.title('Team ES3 application')


if make_prediction == True:
    st.write('***')
    st.title('Make Prediction')
    model_selected = st.selectbox('Select model',['Linear support vector classifier','Random Forest Classifier','Naive Bayes','K-Nearest Neighbours','Logistic Regression'])
    if model_selected == 'Linear support vector classifier':
        model = lsvc
    elif model_selected == 'Random Forest Classifier':
        model = rf
    elif model_selected == 'Naive Bayes':
        model = nb
    elif model_selected == 'K-Nearest Neighbours':
        model = knn
    else:
        model = lr


    var_text = st.text_area('Enter text min: 3 words, max: 500 words')
    button = st.button('classify')

    if button == True:
        var_len = len(var_text.split())
        if var_len < 3:
            st.error('You need more than Three words to classify')
        elif var_len > 500:
            st.error('You have exceeded the maximum input')
        else:
            data = process_single_text(var_text)
            data2 = model.predict(data)[0]
            if data2 == 1 :
                #st.write('Pro')
                st.success('successfully classified')
                st.subheader('Model used:' + '  '+ model_selected)
                st.subheader('Sentiment:'+ '  '+'Pro')
          
            elif data2 == 0 :
                #st.write('Neutral')
                st.success('successfully classified')
                st.subheader('Model used:' + '  '+ model_selected)
                st.subheader('Sentiment:'+'  '+'Neutral')
  
            elif data2 == -1 :
                #st.write('Anti')
                st.success('successfully classified')
                st.subheader('Model used:' + '  '+ model_selected)
                st.subheader('Sentiment:'+ '  '+'Anti')
   
            else :
                #st.write('News')
                st.success('successfully classified')
                st.subheader('Model used:' + '  '+ model_selected)
                st.subheader('Sentiment:'+ '  '+'News')





if eda == True:
	
	st.write('***')
	st.title('EXPLORATORY DATA ANALYSIS')
	
	col1,col2 = st.columns(2)
	
	col1.write('Test Data')
	col2.write('Train Data')
	col1.metric('Total Features',15919,0)
	col2.metric('Total Features',10546,-5273)


	col1,col2 = st.columns(2)
	col1.image('./image1.png')
	col2.write('''
	## Training set preview
	>1. 10% dublicate entries on the training set
	>2. 4 different labels
	>3. Hightly imbalanced
	
	''')

	test = pd.read_csv('./train.csv').groupby(by = 'sentiment').count()
	test = test.drop('tweetid',axis =1)
	test.index = ['Anti','Neutral','Pro','News']
	test['count'] = [1296,2353,8530,3640]
	test = test.drop('message',axis =1)
	
	st.subheader('Bar Graph showing count per category')
	st.write('')
	
	st.bar_chart(test)


if home == True:
    st.write('***')
    st.write('''
    ### Introduction

###### Using Twitter to measure the impact of climate change:
As the climate crisis intensifies and natural disasters become more frequent and powerful, scientists are increasingly turning to social media as a way to assess the damage and impact on a more localized scale. In our case, Twitter was useful given the geographical reach of Twitter as well as the volume and location-specific nature of tweets. The platform can be used to track how individuals feel about climate change and how they view climate change.
Social media encourages greater knowledge of climate change, mobilization of climate change activists, space for discussing the issue with others, and online discussions that frame climate change as a negative for society. Social media, however, does provide space for framing climate change skeptically and activating those with a skeptical perspective of climate change.

    ''')

    st.image('https://media.tenor.com/images/47d160eabb0927ed23827ab099ee83c3/tenor.gif')

    st.write('''
        ## Problem statement

    *Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received.*

    *With this context, EDSA is challenging you during the Classification Sprint with the task of creating a Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data.*

    *Providing an accurate and robust solution to this task gives companies access to a broad base of consumer sentiment, spanning multiple demographic and geographic categories - thus increasing their insights and informing future marketing strategies.*
        
        ''')

    st.image('https://en.reset.org/files/imagecache/sc_832x468/2018/02/27/planet_earth.jpg')
            

if report_bug == True:
	st.write('***')
	st.subheader('Report bug')
	with st.form(key='reportbug'):
		email = st.text_input('Enter email address')
		message = st.text_area('bug description')
		st.form_submit_button('report')




if contact_us == True:
    st.write('***')
    st.subheader('Contact Us Page')
    with st.form(key='contactus'):
        email = st.text_input('Enter email address')
        full_name = st.text_input('Enter full name')
        country = st.text_input('enter country')
        message = st.text_area('message')
        st.form_submit_button('submit')


if model_performance  == True: 
    st.write('***')
    st.subheader('Model Performance')
    model_name = {'Logistic Regression':76, 'Random Forest':60,'KNN':68,'Linear support Vector':76,'Naive Bayes':70}
    df = pd.DataFrame(model_name, index = [1,2,3,4,5])
    df.index = ['f1_score','Random Forest','KNN','Linear support Vector','Naive Bayes']
    df = df.loc['f1_score']
    st.line_chart(df)
    st.bar_chart(df)
