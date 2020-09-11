import json
import plotly
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """ Tokenize each message and remove useless words or symbols.
    INPUT: text -- The message need to be tokenized
    OUTPUT: token -- Tokens of a message
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # remove stopwords
    tokens = [token for token in  tokens if token not in stop_words]
    # lemmatize
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disasters', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # bar graph visual of  the distribution of message genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    related_counts=df[df['related']==1].groupby('genre').count()['related']
    request_counts=df[df['request']==1].groupby('genre').count()['request']
    direct_report_counts=df[df['direct_report']==1].groupby('genre').count()['direct_report']
    # distribution visual
    category_counts=df.drop(['id','message','original','genre'], axis=1).mean()
    category_names = list(category_counts.index)


    # create visuals
    graphs = [
    # bar graph visual of  the distribution of message genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=related_counts,
                    text=related_counts,
                    textposition='auto',
                    name= 'related'

                ),
                Bar(
                    x=genre_names,
                    y=request_counts,
                    text=request_counts,
                    textposition='auto',
                    name = 'request'
                ),
                Bar(
                    x=genre_names,
                    y=direct_report_counts,
                    text=direct_report_counts,
                    textposition='auto',
                    name = 'Direct report'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                 'width': 800,
                 'height': 500,
                'yaxis': {
                    'title': "Count"
                },

                'xaxis': {
                    'title': "categories"
                },
                'barmode' : 'group'
            }
        },

        # bar graph of category distribution
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts,
                    marker=dict(color='rgb(85,201,159)',
                    line=dict(color='rgb(85,201,159)')
                    )
                   )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'titlefont': {
                      'size': 20,
                      'color': 'rgb(0,0,0)'
                             },
                      'width': 800,
                      'height': 500,
                 'font':{
                 'family': 'Raleway, sans-serif'
                         },
                'showlegend': False,
                'yaxis': {
                    'zeroline': False,
                    'gridwidth': 2
                },
                'bargap': 0.05,
                'xaxis': {
                    'title': "categories",
                    'tickangle': -45,
                    'titlefont': {
                          'size': 16,
                          'color': 'black'
                                 }
                }
            }
        },
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
