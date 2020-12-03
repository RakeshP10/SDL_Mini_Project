import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import nltk # Natural Language tool kit 
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    sample_message = 'i agree, but hard to believe he does, given his strident opposition to ordination of women. #womensordinationÃ¢Â€Â¦'
    sample_message = re.sub("[^a-zA-Z]",' ', str(sample_message))
    sample_message = sample_message.lower()
    final_features = tfidf.transform([sample_message]).toarray()
    print(final_features)
    prediction = model.predict(final_features) 

    return render_template('index.html', prediction_text='Tweet is {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)