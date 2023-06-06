#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import joblib
import re
import spacy
import nltk
from nltk.corpus import stopwords



def main():
    #Import Librartes
    import streamlit as st
    from PIL import Image
    from keras.models import load_model
    #Setting Application title
    st.title('Sentiment Analyzer')
    #Setting Application description
    st.markdown("""
    :dart: This Streamlit app is developed to predict the sentiment of a product review.The application can predict if the sentiment towards a product is positive or negative.
     The application is functional for grocery product reviews.
     """)
    st.markdown("<h3></h3>", unsafe_allow_html = True)
    # Setting Application sidebar default
    image = Image.open('image.PNG')
    add_selectbox = st.sidebar.selectbox(
    "What sort of domain is your review?", ("Grocery Product","Airline Service"))
    st.sidebar.info('This app is created to predict product review sentiments')
    st.sidebar.image(image)
    text = st.text_area(label='Product Review to analyze', value="",  label_visibility="visible",placeholder='Enter your review here',key ='placeholder')
    if add_selectbox =="Grocery Product":
        data_input = preprocess(text)
        if len(text) == 0:
            st.stop()
        else:
            model_rf = joblib.load(r"amazonclf_500_rfc_nlp.sav")
            sentiment = model_rf.predict(data_input)
            probability = model_rf.predict_proba(data_input)
            
        if st.button('Predict'):
            if sentiment[0] == 1:
                st.success('This is a positive review',icon ="👍")
            else:
                st.warning('This is a negative review', icon="👎")
                
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            st.subheader('Polarity Score')
            st.write('Score:', probability)
        
def preprocess(text):
    stop = nltk.corpus.stopwords.words('english')
    text=text.lower()
    words = text.split()
    words_list=[w for w in words if (w not in stop)]
    words_clean=' '.join(words_list)
    review = [words_clean]
    vector = joblib.load(r"tfid500_vector_nlp.sav")
    tokenized_features=vector.transform(review)
    features = pd.DataFrame(data = tokenized_features.toarray(),columns = vector.get_feature_names_out())
    return(features)

            
if __name__ == '__main__':
        main()


# In[ ]:




