import streamlit as st
import numpy as np
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import load_model


st.title('IMDB Sentiment Analysis')
text = st.text_input('Enter a movie review:')

model = load_model('static\IMDB.h5')
#Define a function to predict the sentiment
def predict_sentiment(texts):
    texts = [texts]
    word_index=imdb.get_word_index()
    text = [[word_index[word] for text in texts for word in text.split() if word in word_index]]
    text = pad_sequences(text, maxlen=500)
    prediction = model.predict(text)
    if prediction > 0.5:
        return 'Positive'
    else:
        return 'Negative'

#Create the Streamlit application

if st.button('Predict'):
    sentiment = predict_sentiment(text)
    st.write('Sentiment: ', sentiment)
