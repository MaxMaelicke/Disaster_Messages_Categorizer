# Import Libraries
import sys
import pandas as pd
import numpy as np

import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath = 'disaster_messages.csv', categories_filepath = 'disaster_categories.csv'):
    '''
    Load message dataset and categories dataset from .csv files and create a Pandas DataFrame for each dataset.
    Input:  messages_filepath = string of 'filepath/filename.csv' of the messages file
            categories_filepath = string of 'filepath/filename.csv' of the categories file
    Output: df_messages = Messages DataFrame
            df_categories = Categories DataFrame
    '''
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)

    return df_messages, df_categories


def wrangle_data(df_messages, df_categories):
    '''
    Convert and merge Messages DataFrame and Categories DataFrame into a usable MachineLearning DataFrame
    Input:  df_messages = Messages DataFrame
            df_categories = Categories DataFrame
    Output: df = converted and merged Messages and Categories DataFrames
    '''
    # Convert df_categories into a usable ML df
    categories = df_categories['categories'].str.split(';', expand = True)
    cats = categories[0:1].values.tolist()
    category_colnames = []
    for row in cats[0]:
        category_colnames.append(row[:-2])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])
        categories[column] = categories[column].astype(int)

    df_categories = pd.concat([df_categories, categories], axis = 1)
    df_categories = df_categories.drop('categories', axis = 1)

    # Merge messages and categories DataFrames & Remove Duplicates
    df = df_messages.merge(df_categories, on = 'id')
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename = 'DisasterResponse.db'):
    '''
    Create SQLite3 Database with DataFrame
    Input:  df = DataFrame
            database_filename = database filename (e.g. DisasterResponse.db)
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('df', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df_messages, df_categories = load_data(messages_filepath, categories_filepath)
        #df = load_data(messages_filepath, categories_filepath)

        print('Wrangling and cleaning data...')
        df = wrangle_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
