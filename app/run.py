import json
import plotly
import numpy as np
import pandas as pd

import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import sys
sys.path.append('../models')

from build_model import build_model

app = Flask(__name__)


def tokenize(text):
    '''
    Process text data (messages) into ML ready tokens
    Input:  text = DataFrame with text messages
    Output: clean_tokens = list of processed tokens
    '''
    # Detect and replace URLs
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # tokenize --> tranform sentence into list of words (tokens)
    tokens = word_tokenize(text)
    # lemmatize --> tranform words into word "stem" (e. g. is, was --> be)
    lemmatizer = WordNetLemmatizer()
    # remove unimportant "stop" words (e.g. the, this, at)
    stop_words = stopwords.words('english')
    tokens = [t for t in tokens if t not in stop_words]

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


@app.route('/')
@app.route('/index')
def index():
    '''
    Render index.html webpage and create bar charts about the training dataset with Plotly
    The webpage also receives the user's input text for the classification model
    '''
    # extract data needed for visuals
    # graph one - Distribution of Message Genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # graph two - Categories per Message
    msg_counts = df.iloc[:, 4:]
    msg_counts = msg_counts[msg_counts > 0].count(axis = 1)
    cat_counts, msgs = np.unique(msg_counts, return_counts=True)
    cat_counts = list(cat_counts)
    msgs = list(msgs)
    # graph three - Messages per Category
    class_counts = df.iloc[:, 4:]
    class_counts = class_counts[class_counts > 0].count().sort_values(ascending = False)
    class_names = [x.replace('_', ' ') for x in list(class_counts.index)]
    class_counts = list(class_counts.values)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # graph one - Distribution of Message Genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # graph two - Categories per Message
        {
            'data': [
                Bar(
                    x=cat_counts,
                    y=msgs
                )
            ],
            'layout': {
                'title': 'Categories per Message',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Categories per Message"
                }
            }
        },
        # graph three - Messages per Category
        {
            'data': [
                Bar(
                    x=class_names,
                    y=class_counts
                )
            ],
            'layout': {
                'title': 'Messages per Category',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('index.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    '''
    Render go.html webpage which handles the user query and displays the classification model results
    '''
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
