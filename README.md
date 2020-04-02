# Disaster Messages Categorizer (Flask App)
### Udacity Data Scientist Nanodegree | Disaster Response Pipeline Project

This project inhabitates the code for the flask app categorizing disaster messages.
The app tranforms input text and performs a multi-output classification into 36 possible categories.
As classifier in the machine learning pipeline either the XGBoost Classifier (default) or the RandomForest Classifier can be used. The pipeline is optimized by GridSearch.  

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        Optional: In build_model.py you can choose between XGBoost and RandomForest Classifier

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Previews
Main webpage with graphs about the trainig datasets
![main webpage](/screenshots/screenshot1.png?raw=true "main webpage")

Result from the classifier
![category results](/screenshots/screenshot2.png?raw=true "results")


## Prerequisites & Used libraries

* Python 3

* Numpy
* Pandas

* Flask
* Plotly
* Sys
* JSON

* SQLite3
* SQLAlchemy

* Regex
* NLTK (Natural Language Toolkit)
* Scikit-Learn
* GridSearch
* XGBoost
* Pickle


## Author

* **Max Maelicke** - (https://github.com/MaxMaelicke)


## Acknowledgments

Many thanks to
* **Figure Eight** for the messages and categories datasets (https://www.figure-eight.com/)
* **Udacity** for providing data, templates and the project idea (https://www.udacity.com/).
