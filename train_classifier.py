import sys
# import libraries
import pandas as pd
import numpy as np
import re
import nltk
import pickle

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def load_data(database_filepath):
    '''Load and merge df messages and df categories
    
    Args:
    database_filpath: string. Filepath for SQLite database containing cleaned message data.
       
    Returns:
    X: df. Df containing features
    Y: df. Df containing labels
    category_names: list. List containing categories
    '''
    
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages1", engine)
    
    # Create datasets
    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    
    # Create list containing all category names
    categories = list(Y.columns.values)
    
    return X, Y, categories

def tokenize(text):
    '''
    Normalize and tokenize df.message
    
    Args:
    text: string. String containing message
       
    Returns:
    clean_tokens: list of strings. List containing normalized and lemmatized tokens
    
    '''
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # Initiate List of clean_tokens
    clean_tokens = []
    
    for tok in tokens:
        
        # lemmatize and normalize
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        
        # append clean tokens to list
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    '''MLP
    
    Args:
    None
       
    Returns:
    cv: gridsearchcv object. Gridsearchcv object that finds optimal parameters.
    '''
    
    #create pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize, min_df = 5)),
    ('tfidf', TfidfTransformer(use_idf = True)),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 20, min_samples_split = 10)))
    ])
    
    # parameters dictionary
    parameters = {'vect__min_df': [1, 5],
              'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[10, 20],
              'clf__estimator__min_samples_split':[2,5,10]}
        

    # grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters) #, scoring = scorer, verbose = 10)

    return cv

def get_eval_metrics(BasisArray, PredictedArray, col_names):
    '''
    Report of f1 score, precision and recall for eacht output category of dataset
    
    Args:
    BasisArray: array. Array containing the original labels
    PredictedArray: array. Array containing the predicted labels
    col_names: list of strings. List containing column names of PredictedArray
       
    Returns:
    data_metrics: Contains accuracy, precision, recall and F1 Score for a given set
    of BasisArray and PredictedArray labels
    
    '''
    metrics = []
    
    # Evaluate metrics
    for i in range(len(col_names)):
        # acc = correct predictions/total --> acc = ((TP+TN)/(TP+TN+FP+FN))
        accuracy = accuracy_score(BasisArray[:, i], PredictedArray[:, i])
        # prec = (TP/TP+FN)
        precision = precision_score(BasisArray[:, i], PredictedArray[:, i], average = "micro")
        # rec = (TP/TP+FN)
        recall = recall_score(BasisArray[:, i], PredictedArray[:, i], average = "micro")
        # F1 = (2*rec*prec/(rec+prec)
        f1 = f1_score(BasisArray[:, i], PredictedArray[:, i], average = "micro")
        
        # Append Scores to metrics
        metrics.append([accuracy, precision, recall, f1])
    
    # Store metrics
    metrics = np.array(metrics)
    
    data_metrics = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return data_metrics

def evaluate_model(model, X_test, Y_test, categories):
    '''returns accuracy, precision, recall and F1
    
    Args:
    model: model object. Fitted model
    X_test: df. Df containing test features
    Y_test: df. DF containing test labels
    category_names: list. List containing categories
    
    Returns:
    None
    '''
    # Predict labels --> test dataset
    Y_pred = model.predict(X_test)
    
    # Calculate and print evaluation metrics
    eval_metrics = get_eval_metrics(np.array(Y_test), Y_pred, categories)
    print(eval_metrics)


def save_model(model, model_filepath):

    '''Pickle fitted model
    
    Args:
    model: model object. Fitted model
    model_filepath: string. Saving filepath for fitted model
    
    Returns:
    None
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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