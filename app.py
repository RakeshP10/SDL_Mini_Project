import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import nltk # Natural Language tool kit 
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=500)
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    sample_message = request.form.get('name')
    print(sample_message)
    sample_message = re.sub("[^a-zA-Z]",' ', str(sample_message))
    sample_message = sample_message.lower()
    sample_message_words = []
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    final_message = [wnl.lemmatize(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    final_features = tfidf.transform([final_message]).toarray()
    prediction = model.predict(final_features) 

    return render_template('index.html', prediction_text='Tweet is $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)