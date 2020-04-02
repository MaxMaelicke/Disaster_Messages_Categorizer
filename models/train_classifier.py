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

#from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
import xgboost as xgb


def load_data(database_filepath):
    '''
    Load DataFrame from database and create X and y DataFrames
    Input:  database_filepath = string of database filepath (e. g. 'data/DisasterResponse.db')
    Output: X = DataFrame of independent variables (in this case the messages)
            y = DataFrame of dependent variables (in this case the categories)
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM df', engine)
    X = df['message'].values
    y = df.iloc[:, 4:]

    return X, y


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


# Import outsourced build_model function
from build_model import build_model


def evaluate_model(model, X_test, y_test):
    '''
    Report accuracy and Scikit-Learn's classification report (f1 score, precision and recall) for each output category of the dataset.
    Input:  model = trained model
            X_test = DataFrame of independent variables of the test set
            y_test = DataFrame of dependent variables of the test set
    Output: Print of accuracy and classification report for each category
            Print of best parameters for the model
    '''
    y_pred = model.predict(X_test)

    print('Accuracy: {}'.format((y_pred == y_test).mean()))

    # Print classification report
    for i, col in enumerate(y_test):
        print(col)
        print(classification_report(y_test[col], y_pred[:, i]))

    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath = 'model.pkl'):
    '''
    Save model into pickle file
    Input:  model = trained model
            model_filepath = string of filepath (e. g.'model.pkl')
    '''
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
