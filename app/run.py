import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data

engine = create_engine('sqlite:///../data/DisasterResponse.db')

df = pd.read_sql_table('Messages1', engine)


# load model

model = joblib.load("../models/classifier.pkl")



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    genre_percentage = round(100*genre_counts/genre_counts.sum(), 0)
    genre = list(genre_counts.index)
    
    categories_num = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()
    categories_num = categories_num.sort_values(ascending = False)
    categories = list(categories_num.index)
    
    cat_proportion = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()/len(df)        
    cat_proportion = cat_proportion.sort_values(ascending = False)

    graphs = [
        # Pie Chart showing the percentage of messages by genre
        
        {
            "data": [
              {
                "type": "pie",
                "hole": 0.1,
                "name": "Genre",
                "domain": {
                  "x": genre_percentage,
                  "y": genre
                },
                
                "textinfo": "label+value",
                "labels": genre,
                "values": genre_percentage
              }
            ],
            "layout": {
              "title": "Percentage of Messages by Genre"
            }
        },
        
        {
            'data': [
                {
                "type": "pie",
                #"hole": 0,
                "name": "Genre",
                "domain": {
                  "x": categories,
                  "y": cat_proportion
                },
                
                "labels": categories,
                "values": cat_proportion
              }
            ],

             "layout": {
             "title": "Proportion of Messages by Category"
            }
        },
        
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
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
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