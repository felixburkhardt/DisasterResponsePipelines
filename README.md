

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)


## Installationguide <a name="installation"></a>

The code should run with Python versions 3.*. The following libraries are necessary:


1. pandas
2. sqlalchemy
3. pandas
4. numpy
5. re
6. nltk
7. pickle

## Project Motivation<a name="motivation"></a>

This project is part of the Udacity Nanodegree Program "Data Science". It includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

The project is based on a data set containing real messages that were sent during disaster events. A machine learning pipeline is used to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

## File Descriptions <a name="files"></a>

1. process_data.py:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database

2. train_classifier.py:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

3. run.py
- Builds basis for data visualisations in web app

4. data: 
- sample messages and categories datasets in csv format

5. app: 
- All files necessary to run web app

## Instructions<a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
