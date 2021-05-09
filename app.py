# -*- coding: utf-8 -*-
"""
Created on Sun May  9 02:49:35 2021

@author: SULTAN
"""
import uvicorn
from fastapi import FastAPI
from tensorflow.keras.models import load_model
import joblib
import regex as re
import nltk
from nltk.stem import PorterStemmer
import demoji
import numpy as np 
from depression import Depression
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


# 2. Create the app object
app = FastAPI()
model = load_model('model.h5')
word_vectors= joblib.load('word_vector.pickle')

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

#Route with a single parameter, returns the parameter within a message
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To depression detection web app': f'{name}'}

@app.post('/predict')
def predict_depression(data: Depression):
    data = data.dict()
    text = data['Text']
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is",   text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not",   text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am",    text)
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # stemming
    ps = PorterStemmer()
    text = ps.stem(text)
    # remove stop word not including word 'not'
    stop_words = nltk.corpus.stopwords.words('english')
    stop_words.remove('not')
    text_list = nltk.tokenize.word_tokenize(text)
    word_list = [word for word in text_list if word not in stop_words]
    text = ' '.join(word_list)
    # remove hyperlinks
    match = r'http://\S+|https://\S+'
    if match in text:
        text = re.sub(r'http://\S+|https://\S+', '', text)
    else:
        text
    text = re.sub(r'\d', '', text) # remove digits
    text = re.sub(r'\b\w{1,2}\b'," ", text) # remove words whose length less than 2
    text = re.sub(r"[\n\t\-\\\/]", " ", text, flags = re.MULTILINE)
    text = re.sub(r" {2,}", " ", text, flags=re.MULTILINE) # remove the extra space
    #text = demoji.replace(text, '')
    text = text.lower()
    
    avg_fast_text_vector = []
    vector = np.zeros(300) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in text.split(): 
        if word in word_vectors:
            vector += word_vectors[word]
            cnt_words += 1
    if cnt_words != 0:
        vector /= cnt_words
    avg_fast_text_vector.append(vector)
    # load trained model
    prediction = model.predict(np.array(avg_fast_text_vector))
    
    if prediction[0][0]> 0.5:
        prediction = "This is a depressive post"
    else:
        prediction = "This is not a depressive post"
  
    return {'prediction': prediction}

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload