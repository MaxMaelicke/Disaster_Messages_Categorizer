# Import libraries
import sys
import re
import pandas as pd
import numpy as np
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from xgboost import XGBClassifier
import xgboost as xgb


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


def build_model():
    '''
    Build ML model with Scikit-Learn's Pipeline function and optimize model parameters with GridSearch
    By uncommenting and commenting the 'clf' lines you can use either a RandomForest Classifier or the XGBoost Classifier. Make sure to also uncomment and comment the appropriate grid search parameters.
    Output: model = model pipeline
    '''
    pipeline = Pipeline([
            ('features', FeatureUnion([
                    ('text_pipeline', Pipeline([
                            ('vect', CountVectorizer(tokenizer=tokenize)),
                            ('tfidf', TfidfTransformer())
                            ]))
                    # space for second pipeline
                    ])),
            #('clf', MultiOutputClassifier(RandomForestClassifier()))    # RandomForestClassifier
            ('clf', MultiOutputClassifier(XGBClassifier()))            # XGBoost Classifier
            ])

    # Grid search
    '''
    # RandomForest grid search params
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__max_depth': [5, 6, 7],
        'clf__estimator__min_samples_split': [3, 4, 5]
        }
    '''
    # XGBoost grid search params
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__max_depth': [5, 6, 7],
        'clf__estimator__subsample': [0.5, 0.75, 1]
        }


    # create grid search object
    model = GridSearchCV(pipeline, param_grid = parameters, cv = 3, n_jobs = -1)

    return model
