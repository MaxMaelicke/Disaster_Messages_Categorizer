# Import libraries
import sys
import re
import pandas as pd
import numpy as np
import pickle

import sqlite3
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

#from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
import xgboost as xgb

# GPU support
#from numba import jit, cuda, vectorize

'''
# load data from database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql('SELECT * FROM df', engine)
X = df['message'].values
y = df.iloc[:, 4:]
'''


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM df', engine)
    X = df['message'].values
    y = df.iloc[:, 4:]

    return X, y

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    stop_words = stopwords.words('english')
    tokens = [t for t in tokens if t not in stop_words]

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


#from build_model import build_model
def build_model():
    pipeline = Pipeline([
            ('features', FeatureUnion([
                    ('text_pipeline', Pipeline([
                            ('vect', CountVectorizer(tokenizer=tokenize, ngram_range = (1,2), max_df = 0.9)),
                            ('tfidf', TfidfTransformer(use_idf = False))
                            ]))
                    # space for second pipeline
                    ])),
            ('clf', MultiOutputClassifier(XGBClassifier(max_depth = 7, subsample = 0.75)))
            ])

    # Grid search
    '''
    # parameters for grid search
    parameters = {
        #'features__text_pipeline__vect__ngram_range': ((1, 2)),
        'features__text_pipeline__vect__max_df': (0.9, 1.0),
        #'features__text_pipeline__vect__max_features': (None),
        #'features__text_pipeline__tfidf__use_idf': (False),
        'clf__estimator__max_depth': [6, 7],
        'clf__estimator__subsample': [0.5, 0.75]
        }

    # Initial grid search params
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__max_depth': [5, 6],
        'clf__estimator__subsample': [0.5, 1]
        }

    # create grid search object
    model = GridSearchCV(pipeline, param_grid = parameters, cv = 3, n_jobs = -1)
    '''
    model = pipeline

    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print('Accuracy: {}'.format((y_pred == y_test).mean()))

    # Print classification report
    for i, col in enumerate(y_test):
        print(col)
        print(classification_report(y_test[col], y_pred[:, i]))

    #print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath = 'model.pkl'):
    with open(model_filepath, 'wb') as model_pickle:
        pickle.dump(model, model_pickle)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
